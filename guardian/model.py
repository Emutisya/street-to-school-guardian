"""A small, fitted TF-IDF cosine retriever using Python's standard library."""

from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re

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
        self.catalog = catalog
        self.idf = {}
        self.vectors = {}

    def fit(self):
        counts = {r["id"]: Counter(tokens(r["description"])) for r in self.catalog}
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
            terms = sorted(contributions, key=lambda w: (-contributions[w], w))[:5]
            results.append({
                **resource, "score": round(score, 6), "matched_terms": terms,
                "explanation": "Shared description terms: " + ", ".join(terms)
                    + ". This similarity is not a probability or an assessment of a person."
            })
        results.sort(key=lambda item: (-item["score"], item["id"]))
        return results[:limit]

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "version": 1, "catalog_sha256": fingerprint(self.catalog),
            "idf": self.idf, "vectors": self.vectors
        }, sort_keys=True, indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path, catalog):
        saved = json.loads(Path(path).read_text(encoding="utf-8"))
        # This small catalog makes exact re-fit validation inexpensive and detects
        # stale, corrupt or tampered artifacts without trusting persisted weights.
        expected = cls(catalog).fit()
        if saved != {
            "version": 1, "catalog_sha256": fingerprint(catalog),
            "idf": expected.idf, "vectors": expected.vectors
        }:
            raise ValueError("Model artifact does not match this catalog and implementation. Retrain.")
        expected.idf = saved["idf"]
        expected.vectors = saved["vectors"]
        return expected


def evaluate(model):
    evaluation = json.loads((DATA_DIR / "evaluation.json").read_text(encoding="utf-8"))
    training_texts = {r["description"].strip().lower() for r in model.catalog}
    test_texts = [q["text"].strip().lower() for q in evaluation["queries"]]
    test_texts += [q["text"].strip().lower() for q in evaluation["challenge_queries"]]
    test_texts += [q.strip().lower() for q in evaluation["unmatched_queries"]]
    if len(test_texts) != len(set(test_texts)) or training_texts.intersection(test_texts):
        raise ValueError("Exact train/test overlap or duplicate evaluation query.")
    hits, recalls, reciprocals = [], [], []
    for item in evaluation["queries"]:
        ranked = model.rank(item["text"], limit=3)
        ids = [r["id"] for r in ranked]
        relevant = set(item["relevant"])
        hits.append(bool(ids and ids[0] in relevant))
        recalls.append(len(relevant.intersection(ids)) / len(relevant))
        reciprocals.append(next((1 / (i + 1) for i, rid in enumerate(ids)
                                 if rid in relevant), 0))
    negatives = evaluation["unmatched_queries"]
    rejected = sum(not model.rank(q) for q in negatives)
    challenges = evaluation["challenge_queries"]
    challenge_hits = sum(bool(set(q["relevant"]).intersection(
        r["id"] for r in model.rank(q["text"], limit=3))) for q in challenges)
    challenge_abstentions = sum(not model.rank(q["text"]) for q in challenges)
    return {
        "positive_queries": len(hits), "unmatched_queries": len(negatives),
        "top1_hit_rate": round(sum(hits) / len(hits), 6),
        "mean_recall_at_3": round(sum(recalls) / len(recalls), 6),
        "mrr_at_3": round(sum(reciprocals) / len(reciprocals), 6),
        "unmatched_rejection_rate": round(rejected / len(negatives), 6),
        "challenge_queries": len(challenges),
        "challenge_hit_at_3": round(challenge_hits / len(challenges), 6),
        "challenge_abstention_rate": round(challenge_abstentions / len(challenges), 6),
        "threshold": MIN_SCORE, "catalog_sha256": fingerprint(model.catalog),
        "note": "Small synthetic editorial benchmark, not field accuracy or impact."
    }
