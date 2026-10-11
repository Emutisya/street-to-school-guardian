"""Validation and consent gates shared by HTTP and CLI."""

from copy import deepcopy
import re


class RequestError(ValueError):
    def __init__(self, message, status=400):
        super().__init__(message)
        self.status = status


def consent_fields(payload, allowed):
    if not isinstance(payload, dict):
        raise RequestError("Send a JSON object.")
    if payload.get("consent") is not True:
        raise RequestError("Consent is required before processing or simulating a handoff.", 403)
    if set(payload) - allowed - {"consent"}:
        raise RequestError("Unexpected fields. Do not send personal details.")


def browse(model, payload):
    """Browse fictional metadata, never a person or a free-text search."""
    consent_fields(payload, {"mode", "category", "offset", "limit"})
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
    rows = sorted((row for row in model.catalog
                   if (mode == "any" or mode in row["modes"])
                   and (category is None or row["category"] == category)), key=lambda row: row["id"])
    end = offset + limit
    return {
        "status": "catalog_only", "results": deepcopy(rows[offset:end]),
        "total": len(rows), "offset": offset,
        "next_offset": end if end < len(rows) else None,
        "categories": categories,
        "message": "Fictional catalog browsing, not recommendations or verified availability. No description is processed."
    }


def validate(payload, action="match"):
    if not isinstance(payload, dict):
        raise RequestError("Send a JSON object.")
    if payload.get("consent") is not True:
        raise RequestError("Consent is required before processing or simulating a handoff.", 403)
    allowed = {"consent", "resource_id"} if action == "handoff" else {"consent", "text", "mode"}
    if set(payload) - allowed:
        raise RequestError("Unexpected fields. Do not send personal details.")
    if action == "handoff":
        rid = payload.get("resource_id")
        if not isinstance(rid, str) or not rid or len(rid) > 80:
            raise RequestError("Choose a demo resource.")
        return rid
    text = payload.get("text")
    if not isinstance(text, str) or not text.strip():
        raise RequestError("Describe a barrier without identifying anyone.")
    if len(text) > 800:
        raise RequestError("Use at most 800 characters and no identifying details.")
    if any(ord(char) < 32 and char not in "\n\r\t" for char in text):
        raise RequestError("Remove control characters from the description.")
    # A conservative backstop, not anonymization: names cannot reliably be detected.
    if re.search(r"[@\d]|https?://|www\.", text, re.IGNORECASE):
        raise RequestError("Remove numbers, dates, coordinates, links and contact details. Use general words only.")
    mode = payload.get("mode", "any")
    if not isinstance(mode, str) or mode not in {"any", "offline", "online"}:
        raise RequestError("Mode must be any, offline or online.")
    return text.strip(), mode


def match(model, payload):
    text, mode = validate(payload)
    ranked = model.rank(text, mode)
    return {
        "results": ranked, "status": "matched" if ranked else "no_match",
        "message": "Fictional options for a human conversation; no referral has been sent."
        if ranked else "No close demo match. Try a specific general barrier, relax the format preference, or discuss options with a trusted human. This is not a denial of support.",
        "score_note": "TF-IDF cosine similarity, not a probability, need score or eligibility decision."
    }


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
