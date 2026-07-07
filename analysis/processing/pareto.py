"""Build the Pareto (runtime vs accuracy) dataset from the summary table.

A point is Pareto-optimal if no other point (for the same action - runtime
magnitudes aren't comparable across forward/adjoint/normal_operator/
operator_init) is both faster and more accurate. Runs that failed the
accuracy validation gate are excluded, per README.
"""

from __future__ import annotations

import polars as pl

_PARETO_COLS = ["backend", "action", "runtime_median_ms", "throughput_median", "relative_l2_error"]


def build_pareto(summary: pl.DataFrame) -> pl.DataFrame:
    if summary.is_empty():
        return pl.DataFrame(schema={**dict.fromkeys(_PARETO_COLS, pl.Float64), "is_pareto_optimal": pl.Boolean})

    df = summary.filter(
        (pl.col("suite") == "benchmark")
        & (pl.col("validation_passed") == True)  # noqa: E712
        & pl.col("relative_l2_error").is_not_null()
    ).select(_PARETO_COLS)

    if df.is_empty():
        return df.with_columns(pl.lit(False).alias("is_pareto_optimal"))

    rows = df.to_dicts()
    flags = []
    for i, row in enumerate(rows):
        dominated = any(
            j != i
            and other["action"] == row["action"]
            and other["runtime_median_ms"] <= row["runtime_median_ms"]
            and other["relative_l2_error"] <= row["relative_l2_error"]
            and (
                other["runtime_median_ms"] < row["runtime_median_ms"]
                or other["relative_l2_error"] < row["relative_l2_error"]
            )
            for j, other in enumerate(rows)
        )
        flags.append(not dominated)

    return df.with_columns(pl.Series("is_pareto_optimal", flags))
