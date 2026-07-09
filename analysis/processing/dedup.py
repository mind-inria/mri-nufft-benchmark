"""Keep only the latest run's replicates for each (backend, action, scenario)
combination in the raw data.

Raw parquet files are append-only (see loaders/raw.py) and every one of them
is loaded on every report build - rerunning a scenario doesn't overwrite the
earlier run's file, it adds a new one. Without this step, build_summary
would blend replicates from every historical run of the same combination
into one median/quantile, so a bad run (e.g. one contaminated by the GPU
memory leak fixed in release_gpu_memory) keeps skewing the numbers forever
even after a clean rerun. Each hydra job (one run_id) only ever produces
rows for a single combination, so "latest run_id per combination" is
equivalent to "latest rerun of that exact job".
"""

from __future__ import annotations

import polars as pl


def latest_run_ids(raw: pl.DataFrame, group_cols: list[str]) -> set[str]:
    """The run_id of the most recent run for each distinct group_cols combination.

    Filtering by run_id membership (rather than joining raw back on
    group_cols) sidesteps a polars footgun: several group_cols are
    legitimately null together for a whole scenario family (e.g. nz/ny for
    2D trajectories), and a join's default null != null semantics would
    otherwise silently drop every row of those groups instead of matching
    them to their own latest run.
    """
    if raw.is_empty():
        return set()

    latest = (
        raw.group_by([*group_cols, "run_id"])
        .agg(pl.col("timestamp").max().alias("run_timestamp"))
        .sort("run_timestamp", descending=True)
        .group_by(group_cols)
        .agg(pl.col("run_id").first().alias("run_id"))
    )
    return set(latest["run_id"].to_list())


def keep_latest_runs(raw: pl.DataFrame, group_cols: list[str]) -> pl.DataFrame:
    if raw.is_empty():
        return raw
    return raw.filter(pl.col("run_id").is_in(latest_run_ids(raw, group_cols)))
