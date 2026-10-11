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
