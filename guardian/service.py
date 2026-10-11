"""Validation and consent gates shared by HTTP and CLI."""

from copy import deepcopy
import re
import unicodedata


class RequestError(ValueError):
    def __init__(self, message, status=400, code="invalid_request"):
        super().__init__(message)
        self.status = status
        self.code = code


def consent_fields(payload, allowed):
    if not isinstance(payload, dict):
        raise RequestError("Send a JSON object.", code="invalid_payload")
    if payload.get("consent") is not True:
        raise RequestError("Consent is required before processing or simulating a handoff.", 403, "consent_required")
    if set(payload) - allowed - {"consent"}:
        raise RequestError("Unexpected fields. Do not send personal details.", code="unexpected_fields")


def browse(model, payload):
    """Browse fictional metadata, never a person or a free-text search."""
    consent_fields(payload, {"mode", "category", "offset", "limit", "sort"})
    mode = payload.get("mode", "any")
    if not isinstance(mode, str) or mode not in {"any", "offline", "online"}:
        raise RequestError("Mode must be any, offline or online.")
    categories = sorted({row["category"] for row in model.catalog})
    category = payload.get("category")
    if category is not None and (not isinstance(category, str) or category not in categories):
        raise RequestError("Choose a category from the fictional catalog.")
    offset, limit = payload.get("offset", 0), payload.get("limit", 6)
    if type(offset) is not int or offset < 0 or offset > len(model.catalog):
        raise RequestError("Offset must be a catalog position.")
    if type(limit) is not int or not 1 <= limit <= 12:
        raise RequestError("Limit must be an integer from 1 through 12.")
    sort = payload.get("sort", "id")
    if not isinstance(sort, str) or sort not in {"id", "title", "category"}:
        raise RequestError("Sort must be id, title or category; no relevance ranking is implied.")
    rows = sorted((row for row in model.catalog
                   if (mode == "any" or mode in row["modes"])
                   and (category is None or row["category"] == category)),
                  key=lambda row: (row[sort].casefold(), row["id"]))
    end = offset + limit
    return {
        "status": "catalog_only", "results": deepcopy(rows[offset:end]),
        "total": len(rows), "offset": offset,
        "sort": sort,
        "next_offset": end if end < len(rows) else None,
        "previous_offset": max(0, offset - limit) if offset else None,
        "categories": categories,
        "category_counts": {name: sum(row["category"] == name and
                                     (mode == "any" or mode in row["modes"]) for row in model.catalog)
                            for name in categories},
        "mode_counts": {name: sum((category is None or row["category"] == category)
                                 and (name == "any" or name in row["modes"]) for row in model.catalog)
                        for name in ("any", "offline", "online")},
        "message": "Fictional catalog browsing, not recommendations or verified availability. No description is processed."
    }


def selected_resources(model, payload, minimum=1):
    consent_fields(payload, {"resource_ids"})
    ids = payload.get("resource_ids")
    if (not isinstance(ids, list) or not minimum <= len(ids) <= 3
            or any(not isinstance(rid, str) for rid in ids) or len(set(ids)) != len(ids)):
        raise RequestError(f"Choose {minimum} through 3 distinct demo resource IDs.")
    catalog = {row["id"]: row for row in model.catalog}
    if any(rid not in catalog for rid in ids):
        raise RequestError("Unknown demo resource.", 404)
    return [deepcopy(catalog[rid]) for rid in ids]


def compare(model, payload):
    rows = selected_resources(model, payload, minimum=2)
    fields = ("category", "modes", "language", "cost", "availability")
    return {
        "status": "comparison_only", "resources": rows,
        "shared_modes": sorted(set.intersection(*(set(row["modes"]) for row in rows))),
        "differing_fields": [field for field in fields
                             if any(row[field] != rows[0][field] for row in rows[1:])],
        "message": "Compare fictional descriptions, not people. No option is preferred, verified or eligible by this comparison."
    }


def plan(model, payload):
    rows = selected_resources(model, payload)
    return {
        "status": "planning_only",
        "message": "Anonymous fictional conversation plan. Nothing was sent, saved, booked or referred. A human must verify every real next step.",
        "resources": [{"id": row["id"], "title": row["title"],
                       "conversation": row["handoff"]} for row in rows],
        "verification": [
            "Check that a real service exists and is safe, accessible and currently available.",
            "Ask about actual costs, formats, language and terms directly with a verified human.",
            "Decide whether to proceed; disclose only minimum necessary information outside this demo.",
            "Decline or withdraw at any time. This plan makes no family or eligibility decisions."
        ],
        "privacy": "No barrier text, names, contact details, routes or private notes belong in this plan."
    }


def related(model, payload):
    consent_fields(payload, {"resource_id", "mode", "limit"})
    rid = validate({"consent": True, "resource_id": payload.get("resource_id")}, "handoff")
    source = next((row for row in model.catalog if row["id"] == rid), None)
    if source is None:
        raise RequestError("Unknown demo resource.", 404)
    mode, limit = payload.get("mode", "any"), payload.get("limit", 3)
    if not isinstance(mode, str) or mode not in {"any", "offline", "online"}:
        raise RequestError("Mode must be any, offline or online.")
    if type(limit) is not int or not 1 <= limit <= 3:
        raise RequestError("Choose one through three related fictional descriptions.")
    rows = [row for row in model.rank(source["description"], mode, len(model.catalog)) if row["id"] != rid]
    return {"status": "catalog_similarity_only", "source_id": rid, "source_title": source["title"],
            "results": rows[:limit],
            "message": "Similar fictional catalog descriptions only. Your barrier is not processed; similarity is not suitability, availability or a referral."}


def resource_detail(model, payload):
    consent_fields(payload, {"resource_id"})
    rows = selected_resources(model, {"consent": True, "resource_ids": [payload.get("resource_id")]})
    return {"status": "catalog_only", "resource": rows[0],
            "message": "One fictional catalog entry for human review. No verified availability, personal suitability or referral is implied."}


def validate(payload, action="match"):
    if not isinstance(payload, dict):
        raise RequestError("Send a JSON object.", code="invalid_payload")
    if payload.get("consent") is not True:
        raise RequestError("Consent is required before processing or simulating a handoff.", 403, "consent_required")
    allowed = {"consent", "resource_id"} if action == "handoff" else {"consent", "text", "mode", "limit"}
    if set(payload) - allowed:
        raise RequestError("Unexpected fields. Do not send personal details.", code="unexpected_fields")
    if action == "handoff":
        rid = payload.get("resource_id")
        if not isinstance(rid, str) or not rid or len(rid) > 80:
            raise RequestError("Choose a demo resource.")
        return rid
    text = payload.get("text")
    if not isinstance(text, str) or not text.strip():
        raise RequestError("Describe a barrier without identifying anyone.", code="invalid_text")
    if len(text) > 800:
        raise RequestError("Use at most 800 characters and no identifying details.", code="text_too_long")
    if any(unicodedata.category(char) in {"Cc", "Cf", "Cs"} and char not in "\n\r\t" for char in text):
        raise RequestError("Remove hidden, directional or control characters from the description.", code="hidden_controls")
    # A conservative backstop, not anonymization: names cannot reliably be detected.
    if re.search(r"[@\d]|https?://|www\.", text, re.IGNORECASE):
        raise RequestError("Remove numbers, dates, coordinates, links and contact details. Use general words only.", code="identifying_markers")
    mode = payload.get("mode", "any")
    if not isinstance(mode, str) or mode not in {"any", "offline", "online"}:
        raise RequestError("Mode must be any, offline or online.")
    limit = payload.get("limit", 3)
    if type(limit) is not int or not 1 <= limit <= 6:
        raise RequestError("Choose one through six fictional ideas.")
    return text.strip(), mode


def match(model, payload):
    text, mode = validate(payload)
    ranked = model.rank(text, mode, payload.get("limit", 3))
    return {
        "results": ranked, "status": "matched" if ranked else "no_match",
        "diagnostics": model.diagnostics(text, mode),
        "message": "Fictional options for a human conversation; no referral has been sent."
        if ranked else "No close demo match. Try a specific general barrier, relax the format preference, or discuss options with a trusted human. This is not a denial of support.",
        "score_note": "TF-IDF cosine similarity, not a probability, need score or eligibility decision."
    }


def explain(model, payload):
    text, mode = validate(payload)
    return model.trace(text, mode, payload.get("limit", 3))


def handoff(model, payload):
    rid = validate(payload, action="handoff")
    resource = next((r for r in model.catalog if r["id"] == rid), None)
    if resource is None:
        raise RequestError("Unknown demo resource.", 404)
    return {
        "status": "simulation_only", "title": resource["title"],
        "message": "Nothing was sent, saved or booked. There is no real service or recipient.",
        "checklist": [
            resource["handoff"],
            "A human must verify availability, safety, accessibility, costs and actual terms.",
            "Choose whether to proceed and what minimum information to share directly with a verified service.",
            "You may decline or withdraw at any time; this tool makes no decisions about families."
        ]
    }
