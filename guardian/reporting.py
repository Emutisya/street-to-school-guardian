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
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return report
