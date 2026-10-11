"""A small, fitted TF-IDF cosine retriever using Python's standard library."""

from collections import Counter
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile

DATA_DIR = Path(__file__).parent / "data"
MIN_SCORE = 0.12
STOPWORDS = frozenset(
    "a an and are as at be both but by can cannot could do for from get has have "
    "help how i in is it its let make makes more my need needs of on or our than "
    "that the their them there these this to us want we what when which with would "
    "you your learning support school study lessons studying difficult without".split()
)


def tokens(text):
    return [word for word in re.findall(r"[a-z]+", text.lower())
            if len(word) > 1 and word not in STOPWORDS]


def load_catalog():
    return json.loads((DATA_DIR / "resources.json").read_text(encoding="utf-8"))


def fingerprint(catalog):
    raw = json.dumps(catalog, sort_keys=True, ensure_ascii=True).encode()
    return hashlib.sha256(raw).hexdigest()


class Retriever:
    """Fit only catalog descriptions; query labels never enter training."""

    def __init__(self, catalog):
        if not isinstance(catalog, list) or not catalog:
            raise ValueError("Provide a nonempty resource catalog.")
        seen = set()
        for row in catalog:
            if not isinstance(row, dict):
                raise ValueError("Each catalog resource must be an object.")
            for field in ("id", "title", "category", "description", "language", "cost", "availability", "handoff"):
                if not isinstance(row.get(field), str) or not row[field].strip():
                    raise ValueError(f"Catalog resource {field} must be nonempty text.")
            if row["id"] in seen:
                raise ValueError("Catalog resource IDs must be unique.")
            seen.add(row["id"])
            modes = row.get("modes")
            if (not isinstance(modes, list) or not modes
                    or any(mode not in ("offline", "online") for mode in modes)
                    or len(modes) != len(set(modes))):
                raise ValueError("Resource modes must be unique offline or online values.")
        self.catalog = deepcopy(catalog)
        self.idf = {}
        self.vectors = {}

    def fit(self):
        counts = {r["id"]: Counter(tokens(r["description"])) for r in self.catalog}
        if any(not count for count in counts.values()):
            raise ValueError("Every resource description must contain usable retrieval terms.")
        frequency = Counter(word for doc in counts.values() for word in doc)
        size = len(counts)
        self.idf = {word: math.log((1 + size) / (1 + n)) + 1
                    for word, n in sorted(frequency.items())}
        self.vectors = {rid: self.vectorize(count) for rid, count in counts.items()}
        return self

    def vectorize(self, words):
        count = words if isinstance(words, Counter) else Counter(words)
        weighted = {word: (1 + math.log(n)) * self.idf[word]
                    for word, n in count.items() if word in self.idf}
        norm = math.sqrt(sum(value * value for value in weighted.values()))
        return {word: value / norm for word, value in weighted.items()} if norm else {}

    def rank(self, text, mode="any", limit=3):
        if not isinstance(text, str):
            raise ValueError("Retrieval text must be a string.")
        if mode not in ("any", "offline", "online"):
            raise ValueError("Mode must be any, offline or online.")
        if type(limit) is not int or limit < 1:
            raise ValueError("Retrieval limit must be a positive integer.")
        if not self.idf or set(self.vectors) != {row["id"] for row in self.catalog}:
            raise ValueError("Fit or load the model before retrieval.")
        query = self.vectorize(tokens(text))
        results = []
        for resource in self.catalog:
            if mode != "any" and mode not in resource["modes"]:
                continue
            document = self.vectors[resource["id"]]
            contributions = {word: value * document.get(word, 0)
                             for word, value in query.items() if word in document}
            score = sum(contributions.values())
            if score < MIN_SCORE:
                continue
            ordered = sorted(contributions, key=lambda w: (-contributions[w], w))
            terms = ordered[:5]
            results.append((score, {
                **deepcopy(resource), "score": round(score, 6), "matched_terms": terms,
                "term_contributions": [{"term": term, "contribution": round(contributions[term], 12)}
                                       for term in ordered],
                "explanation_term_count": len(ordered),
                "explanation": "Shared description terms: " + ", ".join(terms)
                    + ". This similarity is not a probability or an assessment of a person."
            }))
        results.sort(key=lambda item: (-item[0], item[1]["id"]))
        return [item for _, item in results[:limit]]

    def save(self, path):
        if not self.idf or set(self.vectors) != {row["id"] for row in self.catalog}:
            raise ValueError("Fit or load the model before saving an artifact.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        content = json.dumps({
            "version": 1, "catalog_sha256": fingerprint(self.catalog),
            "idf": self.idf, "vectors": self.vectors
        }, sort_keys=True, indent=2) + "\n"
        temporary = None
        try:
            with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                             prefix=f".{path.name}.", suffix=".tmp", delete=False) as output:
                temporary = Path(output.name)
                output.write(content)
            os.replace(temporary, path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def diagnostics(self, text, mode="any"):
        ranked = self.rank(text, mode, limit=len(self.catalog))
        words = set(tokens(text))
        known = words.intersection(self.idf)
        candidates = sum(mode == "any" or mode in row["modes"] for row in self.catalog)
        if ranked:
            reason = "matched"
        elif not words:
            reason = "no_usable_terms"
        elif not known:
            reason = "outside_vocabulary"
        elif not candidates:
            reason = "empty_format"
        elif mode != "any" and self.rank(text, "any", limit=len(self.catalog)):
            reason = "format_excluded_matches"
        else:
            reason = "below_similarity_threshold"
        return {
            "reason": reason, "usable_term_count": len(words), "known_term_count": len(known),
            "candidate_count": candidates, "match_count": len(ranked),
            "note": "Counts describe the synthetic catalog and tokenizer, never a person. No input terms are returned."
        }

    def summary(self):
        self.rank("")
        return {
            "algorithm": "catalog-only TF-IDF cosine", "artifact_version": 1,
            "catalog_sha256": fingerprint(self.catalog), "documents": len(self.catalog),
            "vocabulary_size": len(self.idf), "threshold": MIN_SCORE,
            "tokenizer": "English ASCII words, lowercased; stopwords and single letters removed",
            "format_counts": {mode: sum(mode in row["modes"] for row in self.catalog)
                              for mode in ("offline", "online")},
            "category_counts": dict(sorted(Counter(row["category"] for row in self.catalog).items())),
            "vector_norm_range": [
                round(min(math.sqrt(sum(v * v for v in vector.values())) for vector in self.vectors.values()), 12),
                round(max(math.sqrt(sum(v * v for v in vector.values())) for vector in self.vectors.values()), 12)
            ],
            "note": "Synthetic resource weights only. No personal data, field accuracy or eligibility model."
        }

    @classmethod
    def load(cls, path, catalog):
        saved = json.loads(Path(path).read_text(encoding="utf-8"))
        # This small catalog makes exact re-fit validation inexpensive and detects
        # stale, corrupt or tampered artifacts without trusting persisted weights.
        expected = cls(catalog).fit()
        if (not isinstance(saved, dict) or type(saved.get("version")) is not int
                or not isinstance(saved.get("idf"), dict)
                or any(type(weight) not in (int, float) for weight in saved["idf"].values())
                or not isinstance(saved.get("vectors"), dict)
                or any(not isinstance(vector, dict)
                       or any(type(weight) not in (int, float) for weight in vector.values())
                       for vector in saved["vectors"].values())
                or saved != {
            "version": 1, "catalog_sha256": fingerprint(catalog),
            "idf": expected.idf, "vectors": expected.vectors
        }):
            raise ValueError("Model artifact does not match this catalog and implementation. Retrain.")
        expected.idf = saved["idf"]
        expected.vectors = saved["vectors"]
        return expected


def evaluate(model):
    evaluation = json.loads((DATA_DIR / "evaluation.json").read_text(encoding="utf-8"))
    if not isinstance(evaluation, dict):
        raise ValueError("Evaluation must be an object of query groups.")
    for group in ("queries", "challenge_queries", "unmatched_queries"):
        if not isinstance(evaluation.get(group), list) or not evaluation[group]:
            raise ValueError(f"Evaluation {group} must contain at least one query.")
    allowed = {row["id"] for row in model.catalog}
    for item in evaluation["queries"] + evaluation["challenge_queries"]:
        if not isinstance(item, dict) or not isinstance(item.get("text"), str) or not item["text"].strip():
            raise ValueError("Evaluation queries must contain nonempty text.")
        relevant = item.get("relevant")
        if (not isinstance(relevant, list) or not relevant
                or any(not isinstance(identifier, str) for identifier in relevant)
                or len(set(relevant)) != len(relevant) or not set(relevant) <= allowed):
            raise ValueError("Evaluation relevance labels must be unique catalog resource IDs.")
    if any(not isinstance(text, str) or not text.strip() for text in evaluation["unmatched_queries"]):
        raise ValueError("Unmatched evaluation queries must be nonempty text.")
    training_texts = {r["description"].strip().lower() for r in model.catalog}
    test_texts = [q["text"].strip().lower() for q in evaluation["queries"]]
    test_texts += [q["text"].strip().lower() for q in evaluation["challenge_queries"]]
    test_texts += [q.strip().lower() for q in evaluation["unmatched_queries"]]
    if len(test_texts) != len(set(test_texts)) or training_texts.intersection(test_texts):
        raise ValueError("Exact train/test overlap or duplicate evaluation query.")
    canonical = lambda text: tuple(sorted(Counter(tokens(text)).items()))
    canonical_queries = [canonical(text) for text in test_texts]
    if len(canonical_queries) != len(set(canonical_queries)):
        raise ValueError("Token-equivalent duplicate evaluation query.")
    if {query for query in canonical_queries if query} & {canonical(text) for text in training_texts}:
        raise ValueError("Token-equivalent training/evaluation overlap.")
    cases = []
    for group in ("queries", "challenge_queries", "unmatched_queries"):
        for index, item in enumerate(evaluation[group], 1):
            text = item if group == "unmatched_queries" else item["text"]
            relevant = set() if group == "unmatched_queries" else set(item["relevant"])
            ids = [row["id"] for row in model.rank(text, limit=3)]
            cases.append({
                "case_id": f"{group}:{index}",
                "group": group, "text": text, "relevant": sorted(relevant), "ranked": ids,
                "diagnostics": model.diagnostics(text),
                "abstained": not ids,
                "top1_hit": bool(ids and ids[0] in relevant) if relevant else None,
                "recall_at_3": len(relevant.intersection(ids)) / len(relevant) if relevant else None,
                "reciprocal_rank": next((1 / (i + 1) for i, rid in enumerate(ids)
                                        if rid in relevant), 0) if relevant else None,
            })
    positive = [row for row in cases if row["group"] == "queries"]
    negatives = [row for row in cases if row["group"] == "unmatched_queries"]
    challenges = [row for row in cases if row["group"] == "challenge_queries"]
    coverage = []
    for resource in sorted(model.catalog, key=lambda row: row["id"]):
        labeled = [row for row in positive + challenges if resource["id"] in row["relevant"]]
        hits = sum(resource["id"] in row["ranked"] for row in labeled)
        coverage.append({
            "resource_id": resource["id"], "category": resource["category"],
            "standard_labels": sum(row["group"] == "queries" for row in labeled),
            "challenge_labels": sum(row["group"] == "challenge_queries" for row in labeled),
            "retrieved_at_3": hits, "labeled_queries": len(labeled),
            "retrieval_rate": round(hits / len(labeled), 6) if labeled else None
        })
    group_metrics = {}
    for name, group in (("standard", positive), ("challenge", challenges)):
        answered = [row for row in group if not row["abstained"]]
        group_metrics[name] = {
            "queries": len(group), "answered_queries": len(answered),
            "coverage": round(len(answered) / len(group), 6),
            "top1_hit_rate": round(sum(row["top1_hit"] for row in group) / len(group), 6),
            "hit_at_3": round(sum(row["recall_at_3"] > 0 for row in group) / len(group), 6),
            "mean_recall_at_3": round(sum(row["recall_at_3"] for row in group) / len(group), 6),
            "mrr_at_3": round(sum(row["reciprocal_rank"] for row in group) / len(group), 6),
            "answered_top1_hit_rate": round(sum(row["top1_hit"] for row in answered) / len(answered), 6)
                if answered else None
        }
    diagnostic_counts = {
        group: dict(sorted(Counter(row["diagnostics"]["reason"] for row in cases
                                   if row["group"] == group).items()))
        for group in ("queries", "challenge_queries", "unmatched_queries")
    }
    return {
        "positive_queries": len(positive), "unmatched_queries": len(negatives),
        "top1_hit_rate": round(sum(row["top1_hit"] for row in positive) / len(positive), 6),
        "mean_recall_at_3": round(sum(row["recall_at_3"] for row in positive) / len(positive), 6),
        "mrr_at_3": round(sum(row["reciprocal_rank"] for row in positive) / len(positive), 6),
        "unmatched_rejection_rate": round(sum(row["abstained"] for row in negatives) / len(negatives), 6),
        "challenge_queries": len(challenges),
        "challenge_hit_at_3": round(sum(row["recall_at_3"] > 0 for row in challenges) / len(challenges), 6),
        "challenge_abstention_rate": round(sum(row["abstained"] for row in challenges) / len(challenges), 6),
        "threshold": MIN_SCORE, "catalog_sha256": fingerprint(model.catalog),
        "benchmark_sha256": fingerprint(evaluation), "cases": cases,
        "resource_coverage": coverage,
        "unlabeled_resources": [row["resource_id"] for row in coverage if not row["labeled_queries"]],
        "group_metrics": group_metrics,
        "diagnostic_counts": diagnostic_counts,
        "unexpected_match_cases": [row["case_id"] for row in negatives if not row["abstained"]],
        "note": "Small synthetic editorial benchmark, not field accuracy or impact."
    }


def evaluate_formats(model):
    benchmark = evaluate(model)
    profiles = {}
    for mode in ("any", "offline", "online"):
        allowed = {row["id"] for row in model.catalog if mode == "any" or mode in row["modes"]}
        cases = []
        for row in benchmark["cases"]:
            relevant = set(row["relevant"]) & allowed
            ranked = [item["id"] for item in model.rank(row["text"], mode)]
            excluded = bool(row["relevant"] and not relevant)
            cases.append({
                "case_id": row["case_id"], "group": row["group"],
                "format_relevant": sorted(relevant), "ranked": ranked,
                "excluded_labels": sorted(set(row["relevant"]) - allowed),
                "status": "labels_outside_format" if excluded else "evaluated",
                "top1_hit": bool(ranked and ranked[0] in relevant) if relevant else None,
                "recall_at_3": len(relevant & set(ranked)) / len(relevant) if relevant else None
            })
        metrics = {}
        for group in ("queries", "challenge_queries"):
            rows = [row for row in cases if row["group"] == group]
            eligible = [row for row in rows if row["format_relevant"]]
            metrics[group] = {
                "queries": len(rows), "evaluated_queries": len(eligible),
                "excluded_queries": len(rows) - len(eligible),
                "top1_hit_rate": round(sum(row["top1_hit"] for row in eligible) / len(eligible), 6) if eligible else None,
                "mean_recall_at_3": round(sum(row["recall_at_3"] for row in eligible) / len(eligible), 6) if eligible else None
            }
        negatives = [row for row in cases if row["group"] == "unmatched_queries"]
        profiles[mode] = {"candidate_count": len(allowed), "metrics": metrics, "cases": cases,
                          "unmatched_rejection_rate": round(sum(not row["ranked"] for row in negatives) / len(negatives), 6)}
    return {"catalog_sha256": benchmark["catalog_sha256"], "benchmark_sha256": benchmark["benchmark_sha256"],
            "profiles": profiles,
            "note": "Format diagnostics on fixed synthetic labels. Excluded labels are catalog constraints, never family eligibility."}
