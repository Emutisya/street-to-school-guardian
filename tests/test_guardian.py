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

from guardian.model import DATA_DIR, Retriever, evaluate, load_catalog, tokens
from guardian.server import create_server
from guardian.service import RequestError, handoff, match

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
