"""Load raw benchmark results.

Reads every ``benchmark_results/raw/run_*.parquet`` file. Raw files are
append-only, one per run - this just concatenates them, it never mutates
them.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl


def load_raw(results_dir: Path) -> pl.DataFrame:
    files = sorted((Path(results_dir) / "raw").glob("run_*.parquet"))
    if not files:
        return pl.DataFrame()
    return pl.concat([pl.read_parquet(f) for f in files], how="vertical_relaxed")
