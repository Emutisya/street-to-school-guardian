import copy
from contextlib import redirect_stderr
from http.client import HTTPConnection, RemoteDisconnected
import io
import json
from pathlib import Path
import subprocess
import socket
import sys
import threading
import tempfile
import unittest
from unittest.mock import patch

from guardian.model import DATA_DIR, Retriever, audit_catalog, evaluate, evaluate_formats, load_catalog, tokens
from guardian.server import create_server
from guardian.reporting import compare_evaluation, save_evaluation
from guardian.service import RequestError, browse, compare, handoff, match, plan, related

ROOT = Path(__file__).resolve().parents[1]


class ModelTests(unittest.TestCase):
    def setUp(self):
        self.catalog = load_catalog()
        self.model = Retriever(self.catalog).fit()

    def test_fitted_descriptions_not_fixed_keyword_routes(self):
        catalog = copy.deepcopy(self.catalog[:2])
        catalog[0]["description"] = "orchard apples harvest"
        catalog[1]["description"] = "ocean coral reef"
        fitted = Retriever(catalog).fit()
        self.assertEqual(fitted.rank("coral reef")[0]["id"], catalog[1]["id"])
        catalog[0]["description"], catalog[1]["description"] = catalog[1]["description"], catalog[0]["description"]
        refitted = Retriever(catalog).fit()
        self.assertEqual(refitted.rank("coral reef")[0]["id"], catalog[0]["id"])

    def test_match_and_explanations_are_grounded(self):
        item = self.model.rank("Bus fares are expensive.")[0]
        self.assertEqual(item["id"], "transport")
        self.assertGreater(item["score"], 0)
        self.assertLessEqual(item["score"], 1)
        for word in item["matched_terms"]:
            self.assertIn(word, tokens("Bus fares are expensive."))
            self.assertIn(word, tokens(item["description"]))
        self.assertIn("not a probability", item["explanation"])

    def test_complete_contributions_account_for_score_beyond_five_terms(self):
        result = self.model.rank(self.catalog[1]["description"])[0]
        self.assertGreater(result["explanation_term_count"], 5)
        self.assertEqual(len(result["matched_terms"]), 5)
        self.assertEqual(len(result["term_contributions"]), result["explanation_term_count"])
        self.assertAlmostEqual(sum(part["contribution"] for part in result["term_contributions"]),
                               result["score"], places=6)
        for part in result["term_contributions"]:
            self.assertIn(part["term"], tokens(result["description"]))
        self.assertEqual(result["term_contributions"],
                         sorted(result["term_contributions"], key=lambda part: (-part["contribution"], part["term"])))

    def test_display_rounding_does_not_change_retrieval_order(self):
        resources = copy.deepcopy(self.catalog[:2])
        resources[0]["id"], resources[1]["id"] = "a", "z"
        model = Retriever(resources).fit()
        model.vectors = {"a": {"bus": .5000001}, "z": {"bus": .5000002}}
        with patch.object(model, "vectorize", return_value={"bus": 1}):
            ranked = model.rank("bus")
        self.assertEqual([row["id"] for row in ranked], ["z", "a"])
        self.assertEqual(ranked[0]["score"], ranked[1]["score"])

    def test_catalog_validation_and_snapshot_prevent_stale_vectors(self):
        for catalog in ([], None, [None], [self.catalog[0], self.catalog[0]]):
            with self.subTest(catalog=catalog), self.assertRaises(ValueError):
                Retriever(catalog)
        for field, value in (("description", ""), ("modes", []), ("modes", ["other"])):
            catalog = copy.deepcopy(self.catalog)
            catalog[0][field] = value
            with self.assertRaises(ValueError):
                Retriever(catalog)
        before = self.model.rank("bus fares")
        self.catalog[1]["description"] = "Changed after fitting"
        self.assertEqual(self.model.rank("bus fares"), before)

    def test_catalog_schema_blocks_personal_extensions_hidden_controls_and_unbounded_ids(self):
        for fields in ({"contact": "not allowed"}, {"eligibility": "not allowed"},
                       {"description": "bus\x00fares"}, {"title": "Hidden\u202etitle"},
                       {"id": "white space"}, {"id": "x" * 81}, {"description": "x" * 1601}):
            catalog = copy.deepcopy(self.catalog)
            catalog[0].update(fields)
            with self.assertRaises(ValueError):
                Retriever(catalog)
        self.assertEqual(Retriever(self.catalog).fit().rank("notebooks")[0]["id"], "learning-kit")

    def test_invalid_retrieval_options_and_unfitted_model_are_rejected(self):
        for text, mode, limit in ((None, "any", 3), ("bus", [], 3),
                                  ("bus", "any", True), ("bus", "any", 0)):
            with self.assertRaises(ValueError):
                self.model.rank(text, mode, limit)
        with self.assertRaisesRegex(ValueError, "Fit or load"):
            Retriever(self.catalog).rank("bus fares")

    def test_empty_description_vectors_and_unfitted_artifacts_are_rejected(self):
        catalog = copy.deepcopy(self.catalog)
        catalog[0]["description"] = "the and school"
        with self.assertRaisesRegex(ValueError, "usable retrieval terms"):
            Retriever(catalog).fit()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "unfitted.json"
            with self.assertRaisesRegex(ValueError, "before saving"):
                Retriever(self.catalog).save(path)
            self.assertFalse(path.exists())

    def test_format_filter_is_not_eligibility(self):
        result = match(self.model, {"consent": True, "text": "Printed worksheets and paper exercises.", "mode": "online"})
        self.assertEqual(result["status"], "no_match")
        self.assertIn("not a denial", result["message"])

    def test_consumers_cannot_change_format_filters_through_results(self):
        expected = self.model.rank("bus fares", "offline")
        result = self.model.rank("bus fares", "offline")
        result[0]["modes"].clear()
        self.assertEqual(self.model.rank("bus fares", "offline"), expected)

    def test_unseen_and_generic_queries_abstain(self):
        for text in ["", "quantum entanglement", "Learning.", "We want assistance."]:
            with self.subTest(text=text):
                self.assertEqual(self.model.rank(text), [])

    def test_query_diagnostics_explain_catalog_limits_without_echoing_terms(self):
        for text, mode, reason in (
            ("Learning support", "any", "no_usable_terms"),
            ("quantum entanglement", "any", "outside_vocabulary"),
            ("printed worksheets paper exercises", "online", "format_excluded_matches"),
            ("bus fares", "offline", "matched"),
        ):
            diagnostic = self.model.diagnostics(text, mode)
            self.assertEqual(diagnostic["reason"], reason)
            self.assertNotIn(text, json.dumps(diagnostic))
            self.assertEqual(diagnostic["match_count"], len(self.model.rank(text, mode, limit=12)))
        catalog = copy.deepcopy(self.catalog[:1])
        model = Retriever(catalog).fit()
        self.assertEqual(model.diagnostics("notebooks", "online")["reason"], "empty_format")
        self.assertEqual(model.diagnostics("notebooks " + " ".join(model.idf), "offline")["known_term_count"],
                         len(model.idf))

    def test_saved_model_roundtrip_and_tamper_detection(self):
        path = ROOT / "models" / "test-roundtrip.json"
        try:
            self.model.save(path)
            loaded = Retriever.load(path, self.catalog)
            self.assertEqual(loaded.rank("bus fares"), self.model.rank("bus fares"))
            saved = json.loads(path.read_text(encoding="utf-8"))
            saved["version"] = True
            path.write_text(json.dumps(saved), encoding="utf-8")
            with self.assertRaises(ValueError):
                Retriever.load(path, self.catalog)
            saved["version"] = 1
            saved["idf"]["bus"] = -1
            path.write_text(json.dumps(saved), encoding="utf-8")
            with self.assertRaises(ValueError):
                Retriever.load(path, self.catalog)
        finally:
            path.unlink(missing_ok=True)

    def test_held_out_evaluation_and_duplicate_leakage_guard(self):
        scores = evaluate(self.model)
        self.assertGreaterEqual(scores["mean_recall_at_3"], .8)
        self.assertLess(scores["challenge_hit_at_3"], scores["mean_recall_at_3"])
        evaluation = json.loads((DATA_DIR / "evaluation.json").read_text(encoding="utf-8"))
        leaked = copy.deepcopy(self.catalog)
        leaked[0]["description"] = evaluation["queries"][0]["text"]
        with self.assertRaises(ValueError):
            evaluate(Retriever(leaked).fit())

    def test_failed_save_preserves_existing_fitted_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.json"
            self.model.save(path)
            original = path.read_bytes()
            with patch("guardian.model.os.replace", side_effect=PermissionError("Isolated write failure")):
                with self.assertRaises(PermissionError):
                    self.model.save(path)
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(list(Path(directory).glob("*.tmp")), [])
            self.assertEqual(Retriever.load(path, self.catalog).rank("bus fares"), self.model.rank("bus fares"))

    def test_artifact_boolean_weights_cannot_impersonate_unit_weights(self):
        catalog = copy.deepcopy(self.catalog[:1])
        catalog[0]["description"] = "bus"
        model = Retriever(catalog).fit()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.json"
            model.save(path)
            original = json.loads(path.read_text(encoding="utf-8"))
            for field in ("idf", "vectors"):
                saved = copy.deepcopy(original)
                if field == "idf":
                    saved["idf"]["bus"] = True
                else:
                    saved["vectors"][catalog[0]["id"]]["bus"] = True
                path.write_text(json.dumps(saved), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "does not match"):
                    Retriever.load(path, catalog)

    def test_evaluation_rejects_empty_groups_and_invalid_resource_labels(self):
        original = json.loads((DATA_DIR / "evaluation.json").read_text(encoding="utf-8"))
        for mutate in (
            lambda data: data.update(queries=[]),
            lambda data: data.update(unmatched_queries=[]),
            lambda data: data["queries"][0].update(relevant=["missing"]),
            lambda data: data["queries"][0].update(relevant=["transport"] * 2),
        ):
            data = copy.deepcopy(original)
            mutate(data)
            with patch("guardian.model.json.loads", return_value=data), self.assertRaises(ValueError):
                evaluate(self.model)

    def test_benchmark_schema_rejects_contact_markers_and_private_fields_before_retrieval(self):
        original = json.loads((DATA_DIR / "evaluation.json").read_text(encoding="utf-8"))
        for mutate in (
            lambda data: data["queries"][0].update(text="contact@example.invalid"),
            lambda data: data["unmatched_queries"].append("Call 12345"),
            lambda data: data["queries"][0].update(private_notes="not allowed"),
            lambda data: data.update(family_details="not allowed"),
        ):
            data = copy.deepcopy(original); mutate(data)
            with patch("guardian.model.json.loads", return_value=data), patch.object(self.model, "rank") as rank:
                with self.assertRaises(ValueError):
                    evaluate(self.model)
                rank.assert_not_called()

    def test_evaluation_cases_expose_failures_and_track_benchmark_provenance(self):
        report = evaluate(self.model)
        self.assertEqual(report, evaluate(self.model))
        cases = report["cases"]
        self.assertEqual(len(cases), report["positive_queries"] + report["challenge_queries"] + report["unmatched_queries"])
        for case in cases:
            self.assertEqual(case["ranked"], [row["id"] for row in self.model.rank(case["text"])])
        self.assertTrue(any(case["group"] == "challenge_queries" and case["abstained"] for case in cases))
        positive = [case for case in cases if case["group"] == "queries"]
        self.assertAlmostEqual(report["mean_recall_at_3"],
                               sum(case["recall_at_3"] for case in positive) / len(positive), places=6)
        changed = json.loads((DATA_DIR / "evaluation.json").read_text(encoding="utf-8"))
        changed["description"] += " Updated synthetic annotation."
        with patch("guardian.model.json.loads", return_value=changed):
            revised = evaluate(self.model)
        self.assertEqual(revised["cases"], report["cases"])
        self.assertNotEqual(revised["benchmark_sha256"], report["benchmark_sha256"])

    def test_evaluation_resource_coverage_uses_labels_not_catalog_size(self):
        report = evaluate(self.model)
        for row in report["resource_coverage"]:
            cases = [case for case in report["cases"] if row["resource_id"] in case["relevant"]]
            self.assertEqual(row["labeled_queries"], len(cases))
            self.assertEqual(row["retrieved_at_3"], sum(row["resource_id"] in case["ranked"] for case in cases))
        data = json.loads((DATA_DIR / "evaluation.json").read_text(encoding="utf-8"))
        for query in data["queries"] + data["challenge_queries"]:
            query["relevant"] = ["meal"]
        with patch("guardian.model.json.loads", return_value=data):
            sparse = evaluate(self.model)
        self.assertEqual(len(sparse["unlabeled_resources"]), 11)
        self.assertTrue(all(row["retrieval_rate"] is None for row in sparse["resource_coverage"]
                            if row["resource_id"] != "meal"))

    def test_challenge_metrics_distinguish_hit_recall_and_answered_coverage(self):
        report = evaluate(self.model)
        for name, group in (("standard", "queries"), ("challenge", "challenge_queries")):
            rows = [row for row in report["cases"] if row["group"] == group]
            metrics = report["group_metrics"][name]
            self.assertEqual(metrics["answered_queries"], sum(not row["abstained"] for row in rows))
            self.assertAlmostEqual(metrics["mean_recall_at_3"], sum(row["recall_at_3"] for row in rows) / len(rows), places=6)
            self.assertGreaterEqual(metrics["hit_at_3"], metrics["mean_recall_at_3"])
            self.assertGreaterEqual(metrics["answered_top1_hit_rate"], metrics["top1_hit_rate"])
        with patch.object(self.model, "rank", return_value=[]):
            abstained = evaluate(self.model)
        self.assertEqual(abstained["group_metrics"]["challenge"]["coverage"], 0)
        self.assertIsNone(abstained["group_metrics"]["challenge"]["answered_top1_hit_rate"])

    def test_ranking_quality_metrics_penalize_wrong_order_and_account_for_multiple_labels(self):
        with patch.object(self.model, "rank", return_value=[{"id": "transport"}, {"id": "meal"}, {"id": "connectivity"}]):
            report = evaluate(self.model)
        row = next(case for case in report["cases"] if case["case_id"] == "queries:25")
        self.assertAlmostEqual(row["precision_returned"], 2 / 3)
        self.assertAlmostEqual(row["average_precision_at_3"], (1 + 2 / 3) / 2)
        import math
        self.assertAlmostEqual(row["ndcg_at_3"], (1 + 1 / math.log2(4)) / (1 + 1 / math.log2(3)))
        standard = [case for case in report["cases"] if case["group"] == "queries"]
        self.assertAlmostEqual(report["group_metrics"]["standard"]["map_at_3"],
                               sum(case["average_precision_at_3"] for case in standard) / len(standard), places=6)
        with patch.object(self.model, "rank", return_value=[]):
            empty = evaluate(self.model)
        self.assertEqual(empty["group_metrics"]["standard"]["map_at_3"], 0)
        self.assertEqual(empty["group_metrics"]["standard"]["ndcg_at_3"], 0)

    def test_evaluation_diagnostics_separate_vocabulary_failures_and_negative_matches(self):
        report = evaluate(self.model)
        self.assertEqual(len({row["case_id"] for row in report["cases"]}), len(report["cases"]))
        for group, counts in report["diagnostic_counts"].items():
            self.assertEqual(sum(counts.values()), sum(row["group"] == group for row in report["cases"]))
        self.assertGreater(report["diagnostic_counts"]["challenge_queries"]["outside_vocabulary"], 0)
        self.assertEqual(report["unexpected_match_cases"], [])
        original = self.model.rank
        def contaminate(text, mode="any", limit=3):
            return original("bus fares", mode, limit) if text == "Learning." else original(text, mode, limit)
        with patch.object(self.model, "rank", side_effect=contaminate):
            changed = evaluate(self.model)
        self.assertEqual(changed["unexpected_match_cases"], ["unmatched_queries:6"])

    def test_format_evaluation_excludes_unavailable_labels_from_denominators(self):
        report = evaluate_formats(self.model)
        online = report["profiles"]["online"]
        self.assertGreater(online["metrics"]["queries"]["excluded_queries"], 0)
        for row in online["cases"]:
            for rid in row["ranked"]:
                self.assertIn("online", next(resource["modes"] for resource in self.catalog if resource["id"] == rid))
            if row["status"] == "labels_outside_format":
                self.assertIsNone(row["recall_at_3"])
        ordinary = evaluate(self.model)
        self.assertEqual(report["profiles"]["any"]["metrics"]["queries"]["mean_recall_at_3"],
                         ordinary["mean_recall_at_3"])
        offline_only = Retriever([row for row in self.catalog if "offline" in row["modes"]]).fit()
        for row in offline_only.catalog:
            row["modes"] = ["offline"]
        empty = evaluate_formats(offline_only)["profiles"]["online"]["metrics"]["queries"]
        self.assertEqual(empty["evaluated_queries"], 0)
        self.assertIsNone(empty["top1_hit_rate"])

    def test_catalog_audit_detects_indistinguishable_resources_and_unexercised_words(self):
        catalog = copy.deepcopy(self.catalog)
        catalog[0]["description"] = "orchard apples"
        catalog[1]["description"] = "apples orchard"
        report = audit_catalog(Retriever(catalog).fit())
        expected = [["learning-kit", "transport"]]
        self.assertEqual(report["token_equivalent_resources"], expected)
        self.assertEqual(report["identical_vector_resources"], expected)
        self.assertIn("orchard", report["unexercised_terms"])
        self.assertEqual(report["exercised_terms"] + report["unexercised_term_count"], report["vocabulary_terms"])
        self.assertLessEqual(len(report["unexercised_terms"]), 50)
        self.assertEqual(report, audit_catalog(Retriever(catalog).fit()))

    def test_token_equivalent_queries_cannot_inflate_evaluation(self):
        data = json.loads((DATA_DIR / "evaluation.json").read_text(encoding="utf-8"))
        duplicate = copy.deepcopy(data["queries"][0])
        duplicate["text"] = duplicate["text"] + "!!!"
        data["queries"].append(duplicate)
        with patch("guardian.model.json.loads", return_value=data), self.assertRaisesRegex(ValueError, "Token-equivalent"):
            evaluate(self.model)
        resources = copy.deepcopy(self.catalog)
        original = json.loads((DATA_DIR / "evaluation.json").read_text(encoding="utf-8"))
        resources[0]["description"] = original["queries"][0]["text"] + "!!!"
        with self.assertRaisesRegex(ValueError, "Token-equivalent"):
            evaluate(Retriever(resources).fit())

    def test_reordered_bag_of_words_cannot_leak_into_evaluation(self):
        data = json.loads((DATA_DIR / "evaluation.json").read_text(encoding="utf-8"))
        duplicate = copy.deepcopy(data["queries"][0])
        duplicate["text"] = " ".join(reversed(tokens(duplicate["text"])))
        data["queries"].append(duplicate)
        with patch("guardian.model.json.loads", return_value=data), self.assertRaisesRegex(ValueError, "Token-equivalent"):
            evaluate(self.model)
        data["queries"].pop()
        resources = copy.deepcopy(self.catalog)
        resources[0]["description"] = duplicate["text"]
        with self.assertRaisesRegex(ValueError, "Token-equivalent"):
            evaluate(Retriever(resources).fit())


class ConsentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = Retriever(load_catalog()).fit()

    def test_explicit_true_required_before_any_input_processing(self):
        for consent in [None, False, "true", 1, [], {}]:
            for operation, payload in [
                (match, {"text": "Bus fares"}),
                (handoff, {"resource_id": "transport"})
            ]:
                with self.subTest(consent=consent, operation=operation.__name__):
                    with self.assertRaises(RequestError) as caught:
                        operation(self.model, {**payload, "consent": consent})
                    self.assertEqual(caught.exception.status, 403)

    def test_empty_invalid_and_identifying_input(self):
        for text in [None, False, 3, [], {}, "", " \n ", "x" * 801, "text\x00",
                     "a@example.invalid", "DOB 2015", "GPS 1.25", "https://example.invalid"]:
            with self.subTest(text=text):
                with self.assertRaises(RequestError):
                    match(self.model, {"consent": True, "text": text})
        for payload in [[], "text", None, 12]:
            with self.assertRaises(RequestError):
                match(self.model, payload)

    def test_hidden_unicode_controls_are_rejected_without_echo_but_plain_whitespace_remains(self):
        for text in ("bus\u200bfares", "bus\u202efares", "bus\ud800fares", "bus\x7ffares"):
            with self.assertRaises(RequestError) as caught:
                match(self.model, {"consent": True, "text": text})
            self.assertNotIn(text, str(caught.exception))
        result = match(self.model, {"consent": True, "text": "bus\tfares\n"})
        self.assertEqual(result["results"][0]["id"], "transport")

    def test_invalid_preferences_and_unexpected_personal_fields(self):
        for mode in ["unknown", None, {}, [], 1]:
            with self.subTest(mode=mode):
                with self.assertRaises(RequestError):
                    match(self.model, {"consent": True, "text": "bus fares", "mode": mode})
        with self.assertRaises(RequestError):
            match(self.model, {"consent": True, "text": "bus fares", "child_name": "Not accepted"})

    def test_handoff_simulation_does_not_echo_barriers(self):
        result = handoff(self.model, {"consent": True, "resource_id": "transport"})
        self.assertEqual(result["status"], "simulation_only")
        self.assertIn("Nothing was sent", result["message"])
        self.assertEqual(len(result["checklist"]), 4)
        with self.assertRaises(RequestError) as caught:
            handoff(self.model, {"consent": True, "resource_id": "missing"})
        self.assertEqual(caught.exception.status, 404)
        with self.assertRaises(RequestError):
            handoff(self.model, {"consent": True, "resource_id": "transport", "text": "bus fares"})

    def test_catalog_browsing_is_bounded_filtered_and_independent(self):
        first = browse(self.model, {"consent": True, "limit": 5})
        second = browse(self.model, {"consent": True, "offset": first["next_offset"], "limit": 5})
        self.assertFalse({r["id"] for r in first["results"]} & {r["id"] for r in second["results"]})
        self.assertEqual(first["total"], 12)
        first["results"][0]["modes"].clear()
        filtered = browse(self.model, {"consent": True, "mode": "online", "category": "Travel"})
        self.assertEqual([r["id"] for r in filtered["results"]], ["transport"])
        self.assertEqual(browse(self.model, {"consent": True, "mode": "online", "category": "Food"})["total"], 0)
        for fields in ({"consent": False}, {"limit": True}, {"limit": 13}, {"offset": -1},
                       {"category": "missing"}, {"text": "not accepted"}):
            with self.assertRaises(RequestError):
                browse(self.model, {"consent": True, **fields})

    def test_catalog_facets_count_across_other_filter_not_current_page(self):
        result = browse(self.model, {"consent": True, "mode": "online", "category": "Food", "limit": 1})
        self.assertEqual(result["total"], 0)
        self.assertEqual(result["category_counts"]["Food"], 0)
        self.assertEqual(result["category_counts"]["Travel"], 1)
        self.assertEqual(result["mode_counts"], {"any": 1, "offline": 1, "online": 0})
        all_rows = browse(self.model, {"consent": True, "limit": 1})
        self.assertEqual(sum(all_rows["category_counts"].values()), 12)
        self.assertGreater(all_rows["mode_counts"]["offline"] + all_rows["mode_counts"]["online"], 12)

    def test_comparison_is_bounded_unranked_and_uses_only_metadata(self):
        result = compare(self.model, {"consent": True, "resource_ids": ["transport", "meal"]})
        self.assertEqual([r["id"] for r in result["resources"]], ["transport", "meal"])
        self.assertEqual(result["shared_modes"], ["offline"])
        self.assertEqual(result["differing_fields"], ["category", "modes"])
        self.assertNotIn("score", result["resources"][0])
        result["resources"][0]["modes"].clear()
        self.assertEqual(len(self.model.catalog[1]["modes"]), 2)
        for ids in ([], ["transport"], ["transport"] * 2, ["missing", "meal"],
                    [None, "meal"], ["transport", "meal", "tutoring", "language"]):
            with self.assertRaises(RequestError):
                compare(self.model, {"consent": True, "resource_ids": ids})
        with self.assertRaises(RequestError):
            compare(self.model, {"consent": False, "resource_ids": ["transport", "meal"]})

    def test_plan_contains_only_selected_catalog_conversations(self):
        payload = {"consent": True, "resource_ids": ["meal", "transport"]}
        result = plan(self.model, payload)
        self.assertEqual([r["id"] for r in result["resources"]], payload["resource_ids"])
        self.assertIn("Nothing was sent", result["message"])
        self.assertEqual(len(result["verification"]), 4)
        self.assertNotIn("description", result["resources"][0])
        for extra in ({"text": "bus fares"}, {"notes": "not allowed"}, {"consent": False}):
            with self.assertRaises(RequestError):
                plan(self.model, {**payload, **extra})

    def test_related_catalog_excludes_source_and_limits_copied_results(self):
        payload = {"consent": True, "resource_id": "connectivity", "mode": "offline", "limit": 2}
        result = related(self.model, payload)
        self.assertLessEqual(len(result["results"]), 2)
        self.assertNotIn("connectivity", [row["id"] for row in result["results"]])
        self.assertTrue(all("offline" in row["modes"] for row in result["results"]))
        self.assertEqual(result["results"], [row for row in self.model.rank(self.model.catalog[3]["description"], "offline", 12)
                                            if row["id"] != "connectivity"][:2])
        for extra in ({"consent": False}, {"text": "not accepted"}, {"limit": True}, {"resource_id": "missing"}):
            with self.assertRaises(RequestError):
                related(self.model, {**payload, **extra})


class HTTPTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = Retriever(load_catalog()).fit()
        cls.server = create_server(cls.model, 0)
        cls.port = cls.server.server_port
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=5)

    def request(self, path, payload=None, headers=None, raw=None):
        connection = HTTPConnection("127.0.0.1", self.port, timeout=5)
        try:
            body = raw if raw is not None else json.dumps(payload) if payload is not None else None
            connection.request("POST" if body is not None else "GET", path, body,
                               headers or ({"Content-Type": "application/json"} if body is not None else {}))
            response = connection.getresponse()
            return response.status, dict(response.getheaders()), response.read().decode()
        finally:
            connection.close()

    def test_dashboard_and_health_are_served(self):
        status, headers, body = self.request("/")
        self.assertEqual(status, 200)
        self.assertIn("text/html", headers["Content-Type"])
        self.assertIn("Street to School Guardian", body)
        self.assertIn("--cp-bg:", body)
        self.assertIn("connect-src 'self'", headers["Content-Security-Policy"])
        self.assertEqual(json.loads(self.request("/health")[2])["resources"], 12)
        self.assertEqual(self.request("/?clawpilotTheme=light")[0], 200)

    def test_readiness_is_distinct_from_server_liveness(self):
        ready = self.request("/ready")
        self.assertEqual(ready[0], 200)
        self.assertEqual(json.loads(ready[2])["status"], "ready")
        with patch.object(self.model, "idf", {}):
            unavailable = self.request("/ready")
            self.assertEqual(unavailable[0], 503)
            self.assertIn("Fit or reload", json.loads(unavailable[2])["recovery"])
            self.assertEqual(self.request("/health")[0], 200)
        self.assertEqual(self.request("/ready")[0], 200)

    def test_http_inference_matches_direct_inference_and_no_query_echo(self):
        payload = {"consent": True, "text": "Bus fares make the journey difficult.", "mode": "offline"}
        status, headers, body = self.request("/api/match", payload)
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body), match(self.model, payload))
        self.assertNotIn(payload["text"], body)
        self.assertEqual(headers["Cache-Control"], "no-store")
        self.assertNotIn("Set-Cookie", headers)
        self.assertNotIn("Access-Control-Allow-Origin", headers)

    def test_http_consent_gates_matching_and_handoff(self):
        for path, fields in [("/api/match", {"text": "bus fares"}),
                             ("/api/handoff", {"resource_id": "transport"})]:
            status, _, _ = self.request(path, {"consent": False, **fields})
            self.assertEqual(status, 403)
        status, _, body = self.request("/api/handoff", {"consent": True, "resource_id": "transport"})
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["status"], "simulation_only")

    def test_http_catalog_browsing_requires_consent(self):
        self.assertEqual(self.request("/api/browse", {"consent": False})[0], 403)
        status, headers, body = self.request("/api/browse", {"consent": True, "mode": "online", "limit": 2})
        self.assertEqual(status, 200)
        self.assertEqual(len(json.loads(body)["results"]), 2)
        self.assertEqual(headers["Cache-Control"], "no-store")

    def test_http_comparison_preserves_selected_order(self):
        payload = {"consent": True, "resource_ids": ["meal", "transport"]}
        status, _, body = self.request("/api/compare", payload)
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body), compare(self.model, payload))

    def test_http_plan_is_stateless_and_consent_gated(self):
        payload = {"consent": True, "resource_ids": ["meal"]}
        status, headers, body = self.request("/api/plan", payload)
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body), plan(self.model, payload))
        self.assertEqual(headers["Cache-Control"], "no-store")
        self.assertEqual(self.request("/api/plan", {**payload, "consent": False})[0], 403)

    def test_http_related_catalog_requires_consent(self):
        payload = {"consent": True, "resource_id": "connectivity"}
        self.assertEqual(json.loads(self.request("/api/related", payload)[2]), related(self.model, payload))
        self.assertEqual(self.request("/api/related", {**payload, "consent": False})[0], 403)

    def test_http_no_match_and_input_errors(self):
        for raw in ('{"consent":false,"consent":true,"text":"bus fares"}',
                    json.dumps({"consent": True, "text": "bus fares"}).encode("utf-16")):
            self.assertEqual(self.request("/api/match", raw=raw)[0], 400)
        self.assertEqual(self.request("/api/match", raw="{}", headers={
            "Content-Type": "application/json", "Content-Length": "+2",
        })[0], 400)
        status, _, body = self.request("/api/match", {"consent": True, "text": "volcano fossils"})
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["status"], "no_match")
        for payload in [{"consent": True, "text": ""}, {"consent": True, "text": 3}, []]:
            self.assertEqual(self.request("/api/match", payload)[0], 400)
        self.assertEqual(self.request("/api/match", raw="{broken")[0], 400)
        self.assertEqual(self.request("/api/match", raw="[" * 1500 + "]" * 1500)[0], 400)
        self.assertEqual(self.request("/api/match", raw="not json", headers={"Content-Type": "text/plain"})[0], 415)
        self.assertEqual(self.request("/api/match", raw="x" * 4097)[0], 413)
        self.assertEqual(self.request("/missing")[0], 404)

    def test_browser_origin_host_and_url_guards(self):
        payload = {"consent": True, "text": "bus fares"}
        for headers in [{"Origin": "https://example.invalid"}, {"Origin": "null"},
                        {"Host": "attacker.invalid"}, {"Sec-Fetch-Site": "cross-site"}]:
            self.assertEqual(self.request("/api/match", payload, {"Content-Type": "application/json", **headers})[0], 403)
        self.assertEqual(self.request("/health?text=private")[0], 400)
        self.assertEqual(self.request("/api/match", payload, {
            "Content-Type": "application/json", "Origin": f"http://127.0.0.1:{self.port}"
        })[0], 200)

    def test_stalled_request_returns_timeout_without_breaking_server(self):
        connection = HTTPConnection("127.0.0.1", self.port, timeout=8)
        try:
            connection.request("POST", "/api/match", "{}", {
                "Content-Type": "application/json", "Content-Length": "100",
            })
            response = connection.getresponse()
            self.assertEqual(response.status, 408)
            self.assertIn("timed out", json.loads(response.read())["error"])
        finally:
            connection.close()
        self.assertEqual(self.request("/health")[0], 200)

    def test_client_disconnect_is_quiet_and_server_remains_healthy(self):
        logs = io.StringIO()
        with redirect_stderr(logs):
            with patch.object(self.server.RequestHandlerClass, "reply",
                              side_effect=ConnectionAbortedError("client closed")):
                with self.assertRaises(RemoteDisconnected):
                    self.request("/health")
        self.assertEqual(logs.getvalue(), "")
        self.assertEqual(self.request("/health")[0], 200)


class CLITests(unittest.TestCase):
    def run_cli(self, *args, text=""):
        return subprocess.run([sys.executable, "-m", "guardian", *args],
                              input=text, text=True, capture_output=True, cwd=ROOT, timeout=10)

    def test_cli_consent_and_anonymous_stdin_inference(self):
        denied = self.run_cli("recommend", text="bus fares")
        self.assertEqual(denied.returncode, 2)
        self.assertEqual(denied.stdout, "")
        allowed = self.run_cli("recommend", "--consent", "--mode", "offline", text="bus fares")
        self.assertEqual(allowed.returncode, 0, allowed.stderr)
        self.assertEqual(json.loads(allowed.stdout)["results"][0]["id"], "transport")
        for text in ["", "a@example.invalid"]:
            self.assertEqual(self.run_cli("recommend", "--consent", text=text).returncode, 2)

    def test_cli_browse_accepts_only_catalog_filters(self):
        self.assertEqual(self.run_cli("browse").returncode, 2)
        result = self.run_cli("browse", "--consent", "--category", "Food", "--limit", "2")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["results"][0]["id"], "meal")
        invalid = self.run_cli("browse", "--consent", "--offset", "-1")
        self.assertEqual(invalid.returncode, 2)
        self.assertEqual(invalid.stdout, "")

    def test_cli_selected_plan_and_comparison(self):
        for command in ("compare", "plan"):
            denied = self.run_cli(command, "meal", "transport")
            self.assertEqual(denied.returncode, 2)
            allowed = self.run_cli(command, "--consent", "meal", "transport")
            self.assertEqual(allowed.returncode, 0, allowed.stderr)
            self.assertIn("resources", json.loads(allowed.stdout))
        self.assertEqual(self.run_cli("plan", "--consent", "missing").returncode, 2)

    def test_cli_related_catalog_reads_no_barrier(self):
        run = self.run_cli("related", "--consent", "connectivity", "--limit", "2", text="ignored stdin")
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertEqual(json.loads(run.stdout)["source_id"], "connectivity")
        self.assertNotIn("ignored stdin", run.stdout)

    def test_occupied_dashboard_port_explains_recovery(self):
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            listener.listen()
            port = listener.getsockname()[1]
            result = self.run_cli("serve", "--port", str(port))
        self.assertEqual(result.returncode, 2)
        self.assertIn(f"port {port}", result.stderr)
        self.assertIn("--port 0", result.stderr)
        self.assertNotIn("Traceback", result.stderr)
        self.assertEqual(result.stdout, "")

    def test_cli_evaluation_is_reproducible(self):
        run = self.run_cli("evaluate")
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertEqual(json.loads(run.stdout), evaluate(Retriever(load_catalog()).fit()))

    def test_cli_format_evaluation_is_reproducible(self):
        run = self.run_cli("evaluate-formats")
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertEqual(json.loads(run.stdout), evaluate_formats(Retriever(load_catalog()).fit()))

    def test_cli_catalog_audit_is_reproducible(self):
        run = self.run_cli("audit-catalog")
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertEqual(json.loads(run.stdout), audit_catalog(Retriever(load_catalog()).fit()))

    def test_evaluation_snapshot_is_atomic_repository_local_and_cli_reproducible(self):
        path = ROOT / "models" / "snapshot-test.json"
        model = Retriever(load_catalog()).fit()
        try:
            run = self.run_cli("evaluate", "--output", str(path))
            self.assertEqual(run.returncode, 0, run.stderr)
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), json.loads(run.stdout))
            original = path.read_bytes()
            with patch("guardian.reporting.os.replace", side_effect=PermissionError):
                with self.assertRaises(PermissionError):
                    save_evaluation(model, path)
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(list(path.parent.glob(f".{path.name}.*.tmp")), [])
            with self.assertRaises(ValueError):
                save_evaluation(model, ROOT.parent / "outside-report.json")
        finally:
            path.unlink(missing_ok=True)

    def test_snapshot_comparison_recomputes_ranks_and_rejects_incompatible_inputs(self):
        path = ROOT / "models" / "comparison-test.json"
        model = Retriever(load_catalog()).fit()
        try:
            report = save_evaluation(model, path)
            unchanged = self.run_cli("compare-evaluation", str(path))
            self.assertEqual(unchanged.returncode, 0, unchanged.stderr)
            self.assertEqual(json.loads(unchanged.stdout)["changed_cases"], [])
            old = copy.deepcopy(report)
            old["cases"][0]["ranked"] = []
            old["top1_hit_rate"] = 999
            path.write_text(json.dumps(old), encoding="utf-8")
            changed = compare_evaluation(model, path)
            self.assertEqual(changed["changed_cases"][0]["case_id"], "queries:1")
            self.assertGreater(changed["metrics"]["top1_hit_rate"]["delta"], 0)
            for mutate in (
                lambda data: data.update(benchmark_sha256="different"),
                lambda data: data["cases"][0].update(text="Unapproved example"),
                lambda data: data["cases"][0].update(ranked=["missing"]),
                lambda data: data["cases"][0].update(private_notes="not accepted"),
            ):
                bad = copy.deepcopy(report); mutate(bad)
                path.write_text(json.dumps(bad), encoding="utf-8")
                with self.assertRaises(ValueError):
                    compare_evaluation(model, path)
            path.write_bytes(b"x" * 262145)
            with self.assertRaisesRegex(ValueError, "too large"):
                compare_evaluation(model, path)
        finally:
            path.unlink(missing_ok=True)

    def test_snapshot_parser_rejects_deep_duplicate_nonfinite_and_non_utf8_json_quietly(self):
        path = ROOT / "models" / "parser-test.json"
        try:
            for raw in (
                b"[" * 2000 + b"]" * 2000,
                b'{"cases":[],"cases":[]}',
                b'{"metric":NaN}',
                "{}".encode("utf-16"),
            ):
                path.write_bytes(raw)
                run = self.run_cli("compare-evaluation", str(path))
                self.assertEqual(run.returncode, 2)
                self.assertEqual(run.stdout, "")
                self.assertNotIn("Traceback", run.stderr)
        finally:
            path.unlink(missing_ok=True)

    def test_cli_inspect_validates_artifacts_and_reports_catalog_geometry(self):
        fitted = self.run_cli("inspect")
        self.assertEqual(fitted.returncode, 0, fitted.stderr)
        report = json.loads(fitted.stdout)
        self.assertEqual(report["vector_norm_range"], [1, 1])
        self.assertEqual(report["documents"], 12)
        self.assertEqual(report["source"], "in_memory_fit")
        path = ROOT / "models" / "inspect-test.json"
        try:
            Retriever(load_catalog()).fit().save(path)
            loaded = self.run_cli("inspect", "--model", str(path))
            self.assertEqual(loaded.returncode, 0, loaded.stderr)
            self.assertEqual(json.loads(loaded.stdout)["catalog_sha256"], report["catalog_sha256"])
            path.write_text("{}", encoding="utf-8")
            invalid = self.run_cli("inspect", "--model", str(path))
            self.assertEqual(invalid.returncode, 2)
            self.assertEqual(invalid.stdout, "")
        finally:
            path.unlink(missing_ok=True)

    def test_invalid_port_precedes_model_loading(self):
        for port in ("-1", "65536", "999999999999999999999"):
            with self.subTest(port=port):
                result = self.run_cli("serve", "--port", port, "--model", "models/missing-port-test.json")
                self.assertEqual(result.returncode, 2)
                self.assertIn("--port must be between 0 and 65535", result.stderr)
                self.assertIn("use 0 to choose an available port", result.stderr)
                self.assertNotIn("Traceback", result.stderr)
                self.assertEqual(result.stdout, "")


if __name__ == "__main__":
    unittest.main()
