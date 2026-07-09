"""Low-level chart-building helpers shared by more than one figure builder.

Kept separate from the figure builders themselves (breakdown.py,
scaling.py) so a change to a shared visual convention (e.g. outlier capping,
CPU/GPU divider shading) only needs one edit instead of touching every
figure that uses it.
"""

from __future__ import annotations

import plotly.graph_objects as go
import polars as pl

from analysis.scenarios import BACKEND_DEVICES, scenario_label

QUALITATIVE_COLORS = [
    "#636efa",
    "#ef553b",
    "#00cc96",
    "#ab63fa",
    "#ffa15a",
    "#19d3f3",
    "#ff6692",
]

ACTIONS = ["forward", "adjoint", "normal_operator", "operator_init"]

# Grouped-bar spacing shared by every bar figure - the default Plotly gaps
# leave the bars looking thin at report scale.
BAR_LAYOUT = dict(bargap=0.1, bargroupgap=0.05)


def empty_figure(message: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=message, showarrow=False, font=dict(size=16))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return fig


def row_spacing(n_rows: int) -> float:
    """Vertical gap between subplot rows, as a fraction of plot height.

    Plotly requires vertical_spacing <= 1 / (n_rows - 1); a fixed small
    value keeps rows close together instead of the gap growing whenever a
    figure happens to have few rows.
    """
    if n_rows <= 1:
        return 0.0
    return min(0.03, 0.9 / (n_rows - 1))


def axis_cap(values: list[float | None]) -> float | None:
    """Cap for a linear axis dominated by a single outlier.

    A lone bar that's much larger than the runner-up (e.g. a pure-Python
    NUDFT reference implementation, or a 3D scenario dwarfing every 2D one)
    would otherwise compress every other bar to invisibility on a linear
    axis. Capping and labelling that one bar with its true value (see
    ``breakdown.metric_by_family_figure``) keeps the rest of the chart
    readable, at the cost of the outlier's bar no longer being to scale.
    Compared against the runner-up (not the smallest value) so a panel with
    many genuinely large bars clustered together isn't mistaken for a single
    outlier.
    """
    vals = sorted(v for v in values if v is not None)
    if len(vals) < 2:
        return None
    largest, median = vals[-1], vals[len(vals) // 2]
    if median > 0 and largest > 10 * median:
        return median * 10
    return None


def format_value(v: float) -> str:
    """Format a metric value in fixed-point - scientific notation is
    unreadable at a glance for the outlier callouts this feeds."""
    av = abs(v)
    if av >= 100:
        return f"{v:,.0f}"
    if av >= 1:
        return f"{v:,.2f}"
    if av >= 0.001:
        return f"{v:.4f}"
    return f"{v:.6f}"


def text_color(hex_color: str) -> str:
    """Black or white, whichever contrasts better against ``hex_color``."""
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i : i + 2], 16) for i in (0, 2, 4))
    luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255
    return "black" if luminance > 0.6 else "white"


def clipped_marker(color: str, overflowing: list[bool]) -> dict:
    """Marker for a bar trace where ``overflowing`` entries are truncated to
    the axis cap - hatched to signal "this bar doesn't reach its true value,
    read the number instead"."""
    return dict(
        color=color,
        pattern=dict(
            shape=["/" if over else "" for over in overflowing],
            fgcolor="rgba(255,255,255,0.55)",
            fgopacity=1,
            size=6,
            solidity=0.3,
        ),
    )


def add_device_divider(fig: go.Figure, *, row: int, col: int, backends: list[str]) -> None:
    """Shade the GPU-backend rows of a horizontal bar panel as a group divider.

    ``backends`` is expected pre-sorted CPU-first (order_backends_by_device),
    so the GPU group is always the trailing contiguous slice of the category
    axis - a shaded background band over just that slice reads as a divider
    without needing to compute an exact line position between two specific
    categories.
    """
    cpu_count = sum(1 for b in backends if BACKEND_DEVICES.get(b) != "cuda")
    if cpu_count == 0 or cpu_count == len(backends):
        return
    fig.add_hrect(
        y0=cpu_count - 0.5,
        y1=len(backends) - 0.5,
        row=row,
        col=col,
        fillcolor="rgba(128,128,128,0.15)",
        line_width=0,
        layer="below",
    )


def scenario_column(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        pl.struct(["trajectory_id", "ncoils"])
        .map_elements(
            lambda s: scenario_label(s["trajectory_id"], s["ncoils"]),
            return_dtype=pl.Utf8,
        )
        .alias("scenario")
    )


def error_bar(
    sub: pl.DataFrame, backends: list[str], col: str, error_cols: tuple[str, str] | None
) -> dict[str, list[float]] | None:
    if error_cols is None or col not in sub.columns:
        return None
    p5_col, p95_col = error_cols
    if p5_col not in sub.columns:
        return None
    lo_by_backend = dict(zip(*sub[["backend", p5_col]]))
    hi_by_backend = dict(zip(*sub[["backend", p95_col]]))
    median_by_backend = dict(zip(*sub[["backend", col]]))
    plus, minus = [], []
    for b in backends:
        median, lo, hi = (
            median_by_backend.get(b),
            lo_by_backend.get(b),
            hi_by_backend.get(b),
        )
        if median is None or lo is None or hi is None:
            plus.append(0.0)
            minus.append(0.0)
        else:
            plus.append(max(hi - median, 0.0))
            minus.append(max(median - lo, 0.0))
    return {"plus": plus, "minus": minus}
