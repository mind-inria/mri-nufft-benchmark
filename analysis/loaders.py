"""Load raw benchmark results and their run metadata.

Reads every ``benchmark_results/raw/run_*.parquet`` file and every
``benchmark_results/metadata/run_*.yaml`` file. Raw files are append-only,
one per run - this only ever concatenates/collects them, it never mutates
them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl
import yaml


def load_raw(results_dir: Path) -> pl.DataFrame:
    files = sorted((Path(results_dir) / "raw").glob("run_*.parquet"))
    if not files:
        return pl.DataFrame()
    # "diagonal_relaxed" unions columns by name (filling gaps with null) and
    # upcasts mismatched dtypes - unlike "vertical_relaxed", it doesn't
    # require every file to have identical column order, which run_*.parquet
    # files written by different suites/code versions won't.
    return pl.concat([pl.read_parquet(f) for f in files], how="diagonal_relaxed")


def load_metadata(results_dir: Path) -> dict[str, dict[str, Any]]:
    metadata_dir = Path(results_dir) / "metadata"
    result: dict[str, dict[str, Any]] = {}
    for path in metadata_dir.glob("run_*.yaml"):
        with open(path) as f:
            data = yaml.safe_load(f)
        result[data["run_id"]] = data
    return result
