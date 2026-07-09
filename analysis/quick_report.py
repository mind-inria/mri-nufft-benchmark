"""Standalone, single-scroll report for the quick backend comparison.

Unlike analysis/report.py's tabbed dashboard (built for the full scenario
matrix, with memory/GPU-resident/scaling tabs), the quick comparison only
ever has runtime+accuracy data for a couple of scenarios - not enough to
justify tab chrome, so this just stacks the runtime figure, speedup table,
and run details on one page. Reuses the same figure/table builders as the
full report so the numbers/labels stay consistent between the two.
"""

from __future__ import annotations

from pathlib import Path

import plotly.io as pio
import polars as pl

from analysis.figures.breakdown import runtime_by_family_figure
from analysis.figures.tables import details_html, speedup_table

_PAGE_CSS = """
<style>
body { font-family: sans-serif; margin: 1.5rem; }
h1 { margin-bottom: 0.25rem; }
h2 { font-size: 1.1rem; margin: 1.5rem 0 0.5rem; }
p { margin: 0.3rem 0; }
.details-table { border-collapse: collapse; margin-bottom: 1rem; }
.details-table th, .details-table td { border: 1px solid #ccc; padding: 4px 10px; font-size: 0.85rem; text-align: left; }
.details-table th { background: #f0f0f0; }
</style>
"""


def build_quick_report(processed_dir: Path, output_path: Path) -> None:
    processed_dir = Path(processed_dir)
    summary = pl.read_parquet(processed_dir / "summary.parquet")

    runtime_html = pio.to_html(
        runtime_by_family_figure(summary), include_plotlyjs=True, full_html=False
    )
    speedup_html = speedup_table(summary)
    details = details_html(processed_dir, summary)

    html = (
        "<html><head><title>mri-nufft-benchmark quick report</title>"
        + _PAGE_CSS
        + "</head><body>"
        + "<h1>mri-nufft-benchmark quick report</h1>"
        + "<p>Fast backend comparison over a small scenario subset - see "
        + "how-to.md's Quick backend comparison section. Not a substitute "
        + "for the full scenario matrix's report.html.</p>"
        + "<h2>Runtime</h2>"
        + runtime_html
        + "<h2>Speedup</h2>"
        + speedup_html
        + details
        + "</body></html>"
    )

    Path(output_path).write_text(html)
