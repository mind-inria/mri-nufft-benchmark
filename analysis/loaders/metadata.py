"""Load hardware/software metadata, keyed by run_id."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_metadata(results_dir: Path) -> dict[str, dict[str, Any]]:
    metadata_dir = Path(results_dir) / "metadata"
    result: dict[str, dict[str, Any]] = {}
    for path in metadata_dir.glob("run_*.yaml"):
        with open(path) as f:
            data = yaml.safe_load(f)
        result[data["run_id"]] = data
    return result
