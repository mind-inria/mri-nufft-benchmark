"""Build the single self-contained report.html from processed/*.parquet.

Plotly is embedded inline (not via CDN) so the report opens with no network
access, matching README's "single self-contained report.html".
"""

from __future__ import annotations

from pathlib import Path

import plotly.graph_objects as go
import plotly.io as pio
import polars as pl

_TABS_CSS = """
<style>
body { font-family: sans-serif; margin: 1.5rem; }
.tab-buttons button { padding: 8px 16px; margin-right: 4px; cursor: pointer; }
.tab-buttons button.active { font-weight: bold; text-decoration: underline; }
.tab-panel { display: none; }
.tab-panel.active { display: block; }
</style>
"""

_TABS_JS = """
<script>
function showTab(name) {
  document.querySelectorAll('.tab-panel').forEach((el) => el.classList.remove('active'));
  document.querySelectorAll('.tab-buttons button').forEach((el) => el.classList.remove('active'));
  document.getElementById('tab-' + name).classList.add('active');
  document.getElementById('btn-' + name).classList.add('active');
}
</script>
"""


def _empty_figure(message: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=message, showarrow=False, font=dict(size=16))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return fig


def _pareto_figure(pareto: pl.DataFrame) -> go.Figure:
    if pareto.is_empty():
        return _empty_figure("No benchmark-suite data yet")

    fig = go.Figure()
    for backend in sorted(pareto["backend"].unique()):
        sub = pareto.filter(pl.col("backend") == backend)
        fig.add_trace(
            go.Scatter(
                x=sub["relative_l2_error"],
                y=sub["runtime_median_ms"],
                mode="markers",
                name=backend,
                marker=dict(
                    symbol=["star" if p else "circle" for p in sub["is_pareto_optimal"]],
                    size=11,
                ),
                text=sub["action"],
                hovertemplate="%{text}<br>l2=%{x}<br>runtime=%{y}ms",
            )
        )
    fig.update_xaxes(type="log", title="Relative L2 error")
    fig.update_yaxes(type="log", title="Runtime (median, ms)")
    fig.update_layout(title="Runtime vs accuracy (Pareto frontier starred)")
    return fig


def _scaling_figure(scaling: pl.DataFrame, variable: str, y: str, title: str) -> go.Figure:
    sub_all = scaling.filter(pl.col("variable") == variable)
    if sub_all.is_empty():
        return _empty_figure("No data yet")

    fig = go.Figure()
    for backend in sorted(sub_all["backend"].unique()):
        sub = sub_all.filter(pl.col("backend") == backend).sort("value")
        fig.add_trace(go.Scatter(x=sub["value"], y=sub[y], mode="lines+markers", name=backend))
    fig.update_xaxes(type="log", title=variable)
    fig.update_yaxes(type="log", title=y)
    fig.update_layout(title=title)
    return fig


def _action_breakdown_figure(summary: pl.DataFrame) -> go.Figure:
    if summary.is_empty():
        return _empty_figure("No data yet")

    df = summary.filter(
        (pl.col("suite") == "benchmark")
        & (pl.col("trajectory_id") == "spiral_2d_256_standard")
        & (pl.col("ncoils") == 8)
    )
    if df.is_empty():
        return _empty_figure("No data yet for the canonical 2D-M/8-coil scenario")

    fig = go.Figure()
    for backend in sorted(df["backend"].unique()):
        sub = df.filter(pl.col("backend") == backend).sort("action")
        fig.add_trace(go.Bar(x=sub["action"], y=sub["runtime_median_ms"], name=backend))
    fig.update_layout(
        title="Action breakdown (2D-M, 8 coils, spiral_2d_256_standard)",
        barmode="group",
        yaxis_title="Runtime (median, ms)",
    )
    return fig


def build_report(processed_dir: Path, output_path: Path) -> None:
    processed_dir = Path(processed_dir)
    summary = pl.read_parquet(processed_dir / "summary.parquet")
    pareto = pl.read_parquet(processed_dir / "pareto.parquet")
    scaling = pl.read_parquet(processed_dir / "scaling.parquet")

    figures = {
        "pareto": _pareto_figure(pareto),
        "runtime_scaling": _scaling_figure(
            scaling, "nx", "runtime_median_ms", "Runtime vs image size"
        ),
        "memory_scaling": _scaling_figure(
            scaling, "nx", "peak_gpu_allocated_mb", "Peak GPU memory vs image size"
        ),
        "action_breakdown": _action_breakdown_figure(summary),
    }

    buttons = ["<div class='tab-buttons'>"]
    panels = []
    plotlyjs_embedded = False
    for i, (name, fig) in enumerate(figures.items()):
        active = " active" if i == 0 else ""
        label = name.replace("_", " ").title()
        buttons.append(
            f"<button id='btn-{name}' class='{active.strip()}' "
            f"onclick=\"showTab('{name}')\">{label}</button>"
        )
        fig_html = pio.to_html(
            fig, include_plotlyjs=(not plotlyjs_embedded), full_html=False
        )
        plotlyjs_embedded = True
        panels.append(f"<div id='tab-{name}' class='tab-panel{active}'>{fig_html}</div>")
    buttons.append("</div>")

    html = (
        "<html><head><title>mri-nufft-benchmark report</title>"
        + _TABS_CSS
        + "</head><body><h1>mri-nufft-benchmark report</h1>"
        + "".join(buttons)
        + "".join(panels)
        + _TABS_JS
        + "</body></html>"
    )

    Path(output_path).write_text(html)
