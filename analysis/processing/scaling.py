"""Build the scaling dataset: runtime/memory vs image size or coil count.

Derived from the summary table, restricted to action=forward. nx-scaling
covers every trajectory family with more than one scenario (see
analysis.scenarios), holding ncoils=1 fixed. ncoils-scaling is restricted to
NCOILS_SCALING_TRAJECTORY_IDS - one reference image size per family - for
the same reason in reverse: every trajectory in that family has its own
ncoils variants, so without holding nx fixed the ncoils-scaling line would
overlay several different-nx coil-sweeps as one series per backend.
"""

from __future__ import annotations

import polars as pl

from analysis.scenarios import (
    NCOILS_SCALING_TRAJECTORY_IDS,
    SOS_3D_FAMILY,
    SPIRAL_2D_FAMILY,
)

_SCALING_FAMILIES = SPIRAL_2D_FAMILY | SOS_3D_FAMILY

_SCALING_SCHEMA = {
    "backend": pl.Utf8,
    "trajectory_id": pl.Utf8,
    "input_location": pl.Utf8,
    "variable": pl.Utf8,
    "value": pl.Int64,
    "runtime_median_ms": pl.Float64,
    "peak_gpu_allocated_mb": pl.Float64,
}


def build_scaling(summary: pl.DataFrame) -> pl.DataFrame:
    if summary.is_empty():
        return pl.DataFrame(schema=_SCALING_SCHEMA)

    # input_location is kept (not restricted to "host") so cuda backends'
    # GPU-resident-input (input_location=device) points ride along in the
    # same dataset - report.py plots them as a dashed line per backend
    # instead of silently mixing them into the host line's points.
    base_filter = (pl.col("action") == "forward") & pl.col("trajectory_id").is_in(
        _SCALING_FAMILIES
    )
    # runtime and peak_gpu_allocated_mb never coexist on the same summary
    # row - benchmark-suite rows carry runtime with memory columns null,
    # memory-suite rows carry memory with runtime columns null (see
    # summary.py). Join the two suites back together on the scenario key so
    # a scaling row can report both.
    scenario_key = ["backend", "trajectory_id", "nx", "ncoils", "input_location"]
    runtime_df = summary.filter(base_filter & (pl.col("suite") == "benchmark")).select(
        *scenario_key, "runtime_median_ms"
    )
    memory_df = summary.filter(base_filter & (pl.col("suite") == "memory")).select(
        *scenario_key, "peak_gpu_allocated_mb"
    )
    df = runtime_df.join(memory_df, on=scenario_key, how="full", coalesce=True)
    if df.is_empty():
        return pl.DataFrame(schema=_SCALING_SCHEMA)

    # trajectory_id disambiguates same-(backend, variable, value) points
    # across families - e.g. ncoils=1 exists for both the 2D and 3D
    # multi-coil trajectories, and nx values happen not to collide across
    # families today but shouldn't be relied on to keep not colliding.
    common = [
        "backend",
        "trajectory_id",
        "input_location",
        "runtime_median_ms",
        "peak_gpu_allocated_mb",
    ]

    nx_rows = df.filter(pl.col("ncoils") == 1).select(
        *[pl.col(c) for c in common],
        pl.lit("nx").alias("variable"),
        pl.col("nx").alias("value"),
    )
    ncoils_rows = df.filter(
        pl.col("trajectory_id").is_in(NCOILS_SCALING_TRAJECTORY_IDS)
    ).select(
        *[pl.col(c) for c in common],
        pl.lit("ncoils").alias("variable"),
        pl.col("ncoils").alias("value"),
    )
    return pl.concat([nx_rows, ncoils_rows], how="vertical_relaxed")
