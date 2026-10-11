"""Repository-local, synthetic benchmark artifacts. Never store live requests."""

import json
import os
from pathlib import Path
import tempfile

from .model import evaluate

ROOT = Path(__file__).resolve().parents[1]


def local_path(path):
    result = Path(path).resolve()
    if not result.is_relative_to(ROOT):
        raise ValueError("Use a repository-local synthetic report path.")
    return result


def save_evaluation(model, path):
    report = evaluate(model)
    path = local_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="\n", dir=path.parent,
                                         prefix=f".{path.name}.", suffix=".tmp", delete=False) as output:
            temporary = Path(output.name)
            output.write(json.dumps(report, sort_keys=True, indent=2) + "\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return report


def compare_evaluation(model, path):
    current = evaluate(model)
    with local_path(path).open("rb") as source:
        raw = source.read(262145)
    if len(raw) > 262144:
        raise ValueError("Synthetic report is too large.")
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate snapshot fields.")
            result[key] = value
        return result
    def reject_constant(_value):
        raise ValueError("Nonfinite JSON constants are not accepted.")
    try:
        previous = json.loads(raw.decode("utf-8"), object_pairs_hook=unique,
                              parse_constant=reject_constant)
    except (ValueError, RecursionError):
        raise ValueError("Use complete strict UTF-8 JSON with unique fields and finite numbers.") from None
    if (not isinstance(previous, dict) or set(previous) != set(current)
            or any(previous[field] != current[field] for field in
                   ("catalog_sha256", "benchmark_sha256", "threshold", "note"))
            or not isinstance(previous.get("cases"), list)
            or len(previous["cases"]) != len(current["cases"])):
        raise ValueError("Compare reports from the same synthetic catalog, benchmark and threshold.")
    allowed = {row["id"] for row in model.catalog}
    changed = []
    pairs = []
    for old, new in zip(previous["cases"], current["cases"]):
        if (not isinstance(old, dict) or set(old) != set(new)
                or any(old[field] != new[field] for field in ("case_id", "group", "text", "relevant"))
                or not isinstance(old["ranked"], list) or len(old["ranked"]) > 3
                or any(not isinstance(rid, str) or rid not in allowed for rid in old["ranked"])
                or len(set(old["ranked"])) != len(old["ranked"])):
            raise ValueError("Snapshot case identities and ranks must match the synthetic benchmark schema.")
        pairs.append((old, new))
        if old["ranked"] != new["ranked"]:
            changed.append({"case_id": new["case_id"], "previous": old["ranked"], "current": new["ranked"]})
    def metrics(cases):
        positives = [row for row in cases if row["relevant"]]
        negatives = [row for row in cases if not row["relevant"]]
        return {
            "top1_hit_rate": sum(bool(row["ranked"] and row["ranked"][0] in row["relevant"]) for row in positives) / len(positives),
            "mean_recall_at_3": sum(len(set(row["ranked"]) & set(row["relevant"])) / len(row["relevant"]) for row in positives) / len(positives),
            "mrr_at_3": sum(next((1 / (index + 1) for index, rid in enumerate(row["ranked"]) if rid in row["relevant"]), 0) for row in positives) / len(positives),
            "unmatched_rejection_rate": sum(not row["ranked"] for row in negatives) / len(negatives)
        }
    before, after = metrics([old for old, _ in pairs]), metrics([new for _, new in pairs])
    group_changes = {}
    for group in ("queries", "challenge_queries"):
        subset = [(old, new) for old, new in pairs if new["group"] == group]
        def positive_metrics(rows):
            return {
                "top1_hit_rate": sum(bool(row["ranked"] and row["ranked"][0] in row["relevant"]) for row in rows) / len(rows),
                "mean_recall_at_3": sum(len(set(row["ranked"]) & set(row["relevant"])) / len(row["relevant"]) for row in rows) / len(rows)
            }
        old_metrics = positive_metrics([old for old, _ in subset])
        new_metrics = positive_metrics([new for _, new in subset])
        recovered, regressed = [], []
        for old, new in subset:
            was_hit = bool(old["ranked"] and old["ranked"][0] in old["relevant"])
            now_hit = bool(new["ranked"] and new["ranked"][0] in new["relevant"])
            if now_hit and not was_hit:
                recovered.append(new["case_id"])
            if was_hit and not now_hit:
                regressed.append(new["case_id"])
        group_changes[group] = {
            "queries": len(subset), "recovered_top1_cases": recovered, "regressed_top1_cases": regressed,
            "metrics": {name: {"previous": round(old_metrics[name], 6), "current": round(new_metrics[name], 6),
                               "delta": round(new_metrics[name] - old_metrics[name], 6)} for name in old_metrics}
        }
    return {
        "catalog_sha256": current["catalog_sha256"], "benchmark_sha256": current["benchmark_sha256"],
        "changed_cases": changed,
        "group_changes": group_changes,
        "metrics": {name: {"previous": round(before[name], 6), "current": round(after[name], 6),
                           "delta": round(after[name] - before[name], 6)} for name in before},
        "note": "All labeled standard and challenge cases combined. Metrics recomputed from validated ranks; stored aggregates ignored. Synthetic comparisons only."
    }
