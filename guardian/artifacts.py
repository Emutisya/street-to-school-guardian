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
