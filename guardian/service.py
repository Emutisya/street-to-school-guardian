"""Validation and consent gates shared by HTTP and CLI."""

import re


class RequestError(ValueError):
    def __init__(self, message, status=400):
        super().__init__(message)
        self.status = status


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
