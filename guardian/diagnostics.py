"""Reproducible diagnostics over bundled synthetic cases only."""

from .model import evaluate
from .service import RequestError, match
from itertools import combinations


def explain_case(model, case_id, consent=False, mode="any", limit=3):
    if consent is not True:
        raise RequestError("Explicit consent is required for a synthetic case replay.", 403, "consent_required")
    benchmark = evaluate(model)
    case = next((row for row in benchmark["cases"] if row["case_id"] == case_id), None)
    if case is None:
        raise RequestError("Choose a bundled case ID from an evaluation report.", code="unknown_case")
    result = match(model, {"consent": True, "text": case["text"], "mode": mode, "limit": limit})
    allowed = {row["id"] for row in model.catalog if mode == "any" or mode in row["modes"]}
    return {"case_id": case["case_id"], "group": case["group"],
            "benchmark_sha256": benchmark["benchmark_sha256"],
            "format_relevant": sorted(set(case["relevant"]) & allowed),
            "excluded_labels": sorted(set(case["relevant"]) - allowed), "retrieval": result,
            "note": "Replays a bundled synthetic case, never arbitrary text from command arguments."}


def vocabulary_report(model):
    benchmark = evaluate(model)
    groups = {}
    for name in ("queries", "challenge_queries", "unmatched_queries"):
        cases = []
        for row in benchmark["cases"]:
            if row["group"] != name:
                continue
            diagnostic = row["diagnostics"]
            usable, known = diagnostic["usable_term_count"], diagnostic["known_term_count"]
            cases.append({"case_id": row["case_id"], "usable_terms": usable, "known_terms": known,
                          "known_fraction": known / usable if usable else None})
        usable_cases = [row for row in cases if row["known_fraction"] is not None]
        groups[name] = {"cases": cases, "zero_usable_queries": len(cases) - len(usable_cases),
                        "fully_unknown_queries": sum(row["known_terms"] == 0 for row in usable_cases),
                        "mean_known_fraction": round(sum(row["known_fraction"] for row in usable_cases) / len(usable_cases), 6)
                        if usable_cases else None}
    return {"catalog_sha256": benchmark["catalog_sha256"], "benchmark_sha256": benchmark["benchmark_sha256"],
            "groups": groups, "note": "Synthetic tokenizer coverage, not language identification, accuracy or personal-data coverage. Zero-usable cases are excluded from the fraction denominator."}


def labeled_metrics(rows):
    return {"queries": len(rows),
            "coverage": round(sum(not row["abstained"] for row in rows) / len(rows), 6) if rows else None,
            "top1_hit_rate": round(sum(row["top1_hit"] for row in rows) / len(rows), 6) if rows else None,
            "mean_recall_at_3": round(sum(row["recall_at_3"] for row in rows) / len(rows), 6) if rows else None,
            "case_ids": [row["case_id"] for row in rows]}


def cardinality_report(model):
    benchmark = evaluate(model)
    groups = {}
    for group in ("queries", "challenge_queries"):
        rows = [row for row in benchmark["cases"] if row["group"] == group]
        groups[group] = {
            "single_label": labeled_metrics([row for row in rows if len(row["relevant"]) == 1]),
            "multiple_labels": labeled_metrics([row for row in rows if len(row["relevant"]) > 1])
        }
    return {"catalog_sha256": benchmark["catalog_sha256"], "benchmark_sha256": benchmark["benchmark_sha256"],
            "groups": groups, "note": "Slices are synthetic annotation cardinality, not child or family profiles. Empty slices have null metrics; standard and challenge cases remain separate."}


def pair_report(model):
    benchmark = evaluate(model)
    tested = {}
    for row in benchmark["cases"]:
        for pair in combinations(sorted(row["relevant"]), 2):
            entry = tested.setdefault(pair, {"resource_ids": list(pair), "standard_labels": 0,
                                             "challenge_labels": 0, "retrieved_together_queries": 0})
            entry["standard_labels" if row["group"] == "queries" else "challenge_labels"] += 1
            entry["retrieved_together_queries"] += int(set(pair) <= set(row["ranked"]))
    possible = set(combinations(sorted(row["id"] for row in model.catalog), 2))
    untested = sorted(possible - set(tested))
    return {
        "catalog_sha256": benchmark["catalog_sha256"], "benchmark_sha256": benchmark["benchmark_sha256"],
        "tested_pairs": [tested[pair] for pair in sorted(tested)],
        "possible_catalog_pairs": len(possible), "untested_pair_count": len(untested),
        "untested_pair_preview": [list(pair) for pair in untested[:20]],
        "omitted_untested_pairs": max(0, len(untested) - 20),
        "note": "Synthetic editorial co-label audit, not family co-occurrence data. Not every mathematical pair is meaningful or requires a benchmark case."
    }


def cutoff_report(model):
    benchmark = evaluate(model)
    ranked = {row["case_id"]: [item["id"] for item in model.rank(row["text"], limit=6)]
              for row in benchmark["cases"]}
    profiles = []
    for limit in range(1, 7):
        groups = {}
        for group in ("queries", "challenge_queries"):
            rows = [row for row in benchmark["cases"] if row["group"] == group]
            ids = [ranked[row["case_id"]][:limit] for row in rows]
            hits = [len(set(found) & set(row["relevant"])) for row, found in zip(rows, ids)]
            groups[group] = {
                "queries": len(rows), "hit_at_limit": round(sum(hit > 0 for hit in hits) / len(rows), 6),
                "mean_recall_at_limit": round(sum(hit / len(row["relevant"]) for row, hit in zip(rows, hits)) / len(rows), 6),
                "mean_precision_returned": round(sum(hit / len(found) if found else 0 for hit, found in zip(hits, ids)) / len(rows), 6),
                "mean_options": round(sum(len(found) for found in ids) / len(rows), 6)
            }
        negatives = [row for row in benchmark["cases"] if row["group"] == "unmatched_queries"]
        profiles.append({"limit": limit, "groups": groups,
                         "unmatched_rejection_rate": round(sum(not ranked[row["case_id"]][:limit] for row in negatives) / len(negatives), 6)})
    return {"catalog_sha256": benchmark["catalog_sha256"], "benchmark_sha256": benchmark["benchmark_sha256"],
            "threshold": benchmark["threshold"], "profiles": profiles,
            "note": "Fixed-threshold synthetic review-list diagnostics. This does not select a threshold, optimize on labels or measure real service usefulness."}


def resource_balance_report(model):
    benchmark = evaluate(model)
    rows = benchmark["resource_coverage"]
    def rates(resources):
        labeled = [row for row in resources if row["labeled_queries"]]
        total = sum(row["labeled_queries"] for row in labeled)
        hits = sum(row["retrieved_at_3"] for row in labeled)
        return {"labeled_resources": len(labeled), "unlabeled_resources": len(resources) - len(labeled),
                "labels": total, "retrieved_labels": hits,
                "micro_retrieval_rate": round(hits / total, 6) if total else None,
                "macro_resource_retrieval_rate": round(sum(row["retrieved_at_3"] / row["labeled_queries"] for row in labeled) / len(labeled), 6)
                if labeled else None}
    return {
        "catalog_sha256": benchmark["catalog_sha256"], "benchmark_sha256": benchmark["benchmark_sha256"],
        "overall": rates(rows),
        "categories": {category: rates([row for row in rows if row["category"] == category])
                       for category in sorted({row["category"] for row in rows})},
        "note": "Synthetic standard and challenge label balance. Macro weights labeled resources equally; micro weights annotations. Unlabeled resources are excluded, not scored. No fairness or population-representativeness claim."
    }
