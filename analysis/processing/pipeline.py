"""run_pipeline: raw parquet files -> processed/{summary,pareto,scaling}.parquet.

Processed files are derived from raw data and may be overwritten - only
benchmark_results/raw/ is append-only.
"""

from __future__ import annotations

from pathlib import Path

from analysis.loaders.raw import load_raw
from analysis.processing.pareto import build_pareto
from analysis.processing.scaling import build_scaling
from analysis.processing.summary import build_summary


def run_pipeline(results_dir: Path) -> None:
    results_dir = Path(results_dir)
    processed_dir = results_dir / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    raw = load_raw(results_dir)
    summary = build_summary(raw)
    pareto = build_pareto(summary)
    scaling = build_scaling(summary)

    summary.write_parquet(processed_dir / "summary.parquet")
    pareto.write_parquet(processed_dir / "pareto.parquet")
    scaling.write_parquet(processed_dir / "scaling.parquet")
