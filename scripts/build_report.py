"""Regenerate processed data + report.html from benchmark_results/raw/."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from analysis.processing.pipeline import run_pipeline  # noqa: E402
from analysis.report import build_report  # noqa: E402

RESULTS_DIR = REPO_ROOT / "benchmark_results"


def main() -> None:
    run_pipeline(RESULTS_DIR)
    report_path = RESULTS_DIR / "report.html"
    build_report(RESULTS_DIR / "processed", report_path)
    print(f"wrote {report_path}")


if __name__ == "__main__":
    main()
