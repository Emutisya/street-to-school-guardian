"""Strict bounded framing for non-personal model artifacts."""

import json
from pathlib import Path


def read_artifact(path):
    with Path(path).open("rb") as source:
        raw = source.read(1048577)
    if len(raw) > 1048576:
        raise ValueError("Model artifact exceeds the one MiB bound.")
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate artifact fields.")
            result[key] = value
        return result
    def reject(_value):
        raise ValueError("Nonfinite artifact numbers.")
    try:
        return json.loads(raw.decode("utf-8"), object_pairs_hook=unique, parse_constant=reject)
    except (ValueError, RecursionError):
        raise ValueError("Use bounded strict UTF-8 artifact JSON with unique fields and finite numbers.") from None


def check_artifact(path, catalog):
    from .model import Retriever, fingerprint
    try:
        saved = read_artifact(path)
    except OSError:
        code = "unreadable"
    except ValueError:
        code = "invalid_framing"
    else:
        if not isinstance(saved, dict) or type(saved.get("version")) is not int:
            code = "invalid_schema"
        elif saved["version"] != 1:
            code = "unsupported_version"
        elif saved.get("catalog_sha256") != fingerprint(catalog):
            code = "catalog_mismatch"
        else:
            try:
                Retriever.load(path, catalog)
            except (ValueError, OSError):
                code = "weights_or_schema_mismatch"
            else:
                code = "valid"
    return {"status": "valid" if code == "valid" else "invalid", "code": code,
            "recovery": "No action required." if code == "valid" else "Check the repository-local path and regenerate with python -m guardian train.",
            "note": "Compatibility check by exact refit, not authenticity, performance or field validation. Artifact contents are not echoed."}
