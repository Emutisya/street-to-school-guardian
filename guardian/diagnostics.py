"""Reproducible diagnostics over bundled synthetic cases only."""

from .model import evaluate
from .service import RequestError, match


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
