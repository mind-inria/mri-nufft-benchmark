"""Build the summary table from raw measurements.

Aggregated separately per suite (benchmark vs memory) - a runtime computed
across both timing runs and memory-profiler-overhead runs would be
meaningless (see README).
"""

from __future__ import annotations

import polars as pl

GROUP_COLS = [
    "suite",
    "backend",
    "backend_version",
    "action",
    "input_location",
    "ndim",
    "nx",
    "ny",
    "nz",
    "ncoils",
    "smaps",
    "trajectory_id",
    "n_samples",
]

# input_location is excluded alongside action: operator_init rows are always
# input_location=host, but must still join persistent_gpu_mb onto
# forward/adjoint rows regardless of their input_location.
_SCENARIO_COLS = [c for c in GROUP_COLS if c not in ("action", "input_location")]

_NULL_ACCURACY_COLS = (
    "relative_l2_error",
    "relative_linf_error",
    "adjointness_error",
    "validation_passed",
)
_NULL_RUNTIME_COLS = (
    "runtime_median_ms",
    "runtime_mad_ms",
    "runtime_p5_ms",
    "runtime_p95_ms",
    "throughput_median",
)
_NULL_MEMORY_COLS = (
    "peak_cpu_rss_mb",
    "peak_gpu_allocated_mb",
    "peak_gpu_driver_mb",
    "persistent_gpu_mb",
)


def _with_null_columns(df: pl.DataFrame, columns: tuple[str, ...]) -> pl.DataFrame:
    return df.with_columns([pl.lit(None).alias(c) for c in columns])


def _summarize_benchmark(df: pl.DataFrame) -> pl.DataFrame:
    summary = df.group_by(GROUP_COLS).agg(
        runtime_median_ms=pl.col("runtime_ms").median(),
        runtime_mad_ms=(pl.col("runtime_ms") - pl.col("runtime_ms").median()).abs().median(),
        runtime_p5_ms=pl.col("runtime_ms").quantile(0.05),
        runtime_p95_ms=pl.col("runtime_ms").quantile(0.95),
        relative_l2_error=pl.col("relative_l2_error").first(),
        relative_linf_error=pl.col("relative_linf_error").first(),
        adjointness_error=pl.col("adjointness_error").first(),
        validation_passed=pl.col("validation_passed").first(),
        n_replicates=pl.len(),
    )
    summary = summary.with_columns(
        (pl.col("n_samples") * pl.col("ncoils") / (pl.col("runtime_median_ms") / 1000.0)).alias(
            "throughput_median"
        )
    )
    return _with_null_columns(summary, _NULL_MEMORY_COLS)


def _summarize_memory(df: pl.DataFrame) -> pl.DataFrame:
    summary = df.group_by(GROUP_COLS).agg(
        peak_cpu_rss_mb=pl.col("cpu_rss_mb").max(),
        peak_gpu_allocated_mb=pl.col("gpu_allocated_mb").max(),
        peak_gpu_driver_mb=pl.col("gpu_driver_mb").max(),
        n_replicates=pl.len(),
    )

    # Persistent GPU memory = the operator_init action's own peak allocation
    # (memory still resident right after construction, before any NUFFT
    # call). The raw schema only records one memory snapshot per
    # (action, replicate) - there's no separate "right after init" probe for
    # every other action - so this is joined back from the operator_init row
    # for the same backend/scenario onto every other action's row.
    persistent = summary.filter(pl.col("action") == "operator_init").select(
        [*_SCENARIO_COLS, pl.col("peak_gpu_allocated_mb").alias("persistent_gpu_mb")]
    )
    summary = summary.join(persistent, on=_SCENARIO_COLS, how="left")

    return _with_null_columns(summary, _NULL_ACCURACY_COLS + _NULL_RUNTIME_COLS)


def build_summary(raw: pl.DataFrame) -> pl.DataFrame:
    if raw.is_empty():
        return raw

    parts = []
    benchmark_rows = raw.filter(pl.col("suite") == "benchmark")
    if not benchmark_rows.is_empty():
        parts.append(_summarize_benchmark(benchmark_rows))

    memory_rows = raw.filter(pl.col("suite") == "memory")
    if not memory_rows.is_empty():
        parts.append(_summarize_memory(memory_rows))

    if not parts:
        return raw.clear()
    return pl.concat(parts, how="diagonal_relaxed")
