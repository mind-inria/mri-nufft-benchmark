"""Build the scaling dataset: runtime/memory vs image size or coil count.

Derived from the summary table, restricted to action=forward and the
spiral_2d_* trajectory family - the only family with more than one scenario
row in the default scenario list (see README; 3D scaling isn't produced
until a second 3D scenario is added).
"""

from __future__ import annotations

import polars as pl

_SPIRAL_2D_FAMILY = {"spiral_2d_128_standard", "spiral_2d_256_standard"}

_SCALING_SCHEMA = {
    "backend": pl.Utf8,
    "variable": pl.Utf8,
    "value": pl.Int64,
    "runtime_median_ms": pl.Float64,
    "peak_gpu_allocated_mb": pl.Float64,
}


def build_scaling(summary: pl.DataFrame) -> pl.DataFrame:
    if summary.is_empty():
        return pl.DataFrame(schema=_SCALING_SCHEMA)

    df = summary.filter(
        (pl.col("suite") == "benchmark")
        & (pl.col("action") == "forward")
        & pl.col("trajectory_id").is_in(_SPIRAL_2D_FAMILY)
    )
    if df.is_empty():
        return pl.DataFrame(schema=_SCALING_SCHEMA)

    common = ["backend", "runtime_median_ms", "peak_gpu_allocated_mb"]

    nx_rows = df.filter(pl.col("ncoils") == 1).select(
        *[pl.col(c) for c in common], pl.lit("nx").alias("variable"), pl.col("nx").alias("value")
    )
    ncoils_rows = df.filter(pl.col("trajectory_id") == "spiral_2d_256_standard").select(
        *[pl.col(c) for c in common],
        pl.lit("ncoils").alias("variable"),
        pl.col("ncoils").alias("value"),
    )
    return pl.concat([nx_rows, ncoils_rows], how="vertical_relaxed")
