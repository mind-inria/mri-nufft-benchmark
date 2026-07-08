"""Build the single self-contained report.html from processed/*.parquet.

Plotly is embedded inline (not via CDN) so the report opens with no network
access, matching README's "single self-contained report.html".
"""

from __future__ import annotations

from html import escape as _esc
from pathlib import Path

import plotly.graph_objects as go
import plotly.io as pio
import polars as pl
import yaml
from plotly.subplots import make_subplots

from analysis.loaders.metadata import load_metadata
from analysis.scenarios import (
    BACKEND_DEVICES,
    SCENARIO_ORDER,
    SOS_3D_FAMILY,
    SPIRAL_2D_FAMILY,
    order_backends_by_device,
    scenario_label,
    trajectory_label,
)

_CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs"

_TABS_CSS = """
<style>
body { font-family: sans-serif; margin: 1.5rem; }
.tab-buttons { display: flex; flex-wrap: wrap; align-items: flex-end; gap: 0 1.5rem; margin-bottom: 0.5rem; }
.tab-group { display: flex; flex-wrap: wrap; align-items: center; gap: 4px; padding: 6px 10px 6px 0; border-right: 1px solid #ccc; }
.tab-group:last-child { border-right: none; }
.tab-group-label { font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.04em; color: #666; margin-right: 6px; }
.tab-buttons button { padding: 8px 16px; cursor: pointer; }
.tab-buttons button.active { font-weight: bold; text-decoration: underline; }
.tab-panel { display: none; }
.tab-panel.active { display: block; }
.details h2 { font-size: 1.1rem; margin: 1.5rem 0 0.5rem; }
.details p { margin: 0.3rem 0; }
.details-table { border-collapse: collapse; margin-bottom: 1rem; }
.details-table th, .details-table td { border: 1px solid #ccc; padding: 4px 10px; font-size: 0.85rem; text-align: left; }
.details-table th { background: #f0f0f0; }
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


_QUALITATIVE_COLORS = [
    "#636efa",
    "#ef553b",
    "#00cc96",
    "#ab63fa",
    "#ffa15a",
    "#19d3f3",
    "#ff6692",
]

_ACTIONS = ["forward", "adjoint", "normal_operator", "operator_init"]

# Grouped-bar spacing shared by every bar figure - the default Plotly gaps
# leave the bars looking thin at report scale.
_BAR_LAYOUT = dict(bargap=0.1, bargroupgap=0.05)


def _row_spacing(n_rows: int) -> float:
    """Vertical gap between subplot rows, as a fraction of plot height.

    Plotly requires vertical_spacing <= 1 / (n_rows - 1); a fixed small
    value keeps rows close together instead of the gap growing whenever a
    figure happens to have few rows.
    """
    if n_rows <= 1:
        return 0.0
    return min(0.03, 0.9 / (n_rows - 1))


def _axis_cap(values: list[float | None]) -> float | None:
    """Cap for a linear axis dominated by a single outlier.

    A lone bar that's much larger than the runner-up (e.g. a pure-Python
    NUDFT reference implementation, or a 3D scenario dwarfing every 2D one)
    would otherwise compress every other bar to invisibility on a linear
    axis. Capping and labelling that one bar with its true value (see
    ``_metric_breakdown_figure``) keeps the rest of the chart readable, at
    the cost of the outlier's bar no longer being to scale. Compared against
    the runner-up (not the smallest value) so a panel with many genuinely
    large bars clustered together isn't mistaken for a single outlier.
    """
    vals = sorted(v for v in values if v is not None)
    if len(vals) < 2:
        return None
    largest, median = vals[-1], vals[len(vals) // 2]
    if median > 0 and largest > 10 * median:
        return median * 10
    return None


def _format_value(v: float) -> str:
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


def _text_color(hex_color: str) -> str:
    """Black or white, whichever contrasts better against ``hex_color``."""
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i : i + 2], 16) for i in (0, 2, 4))
    luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255
    return "black" if luminance > 0.6 else "white"


def _clipped_marker(color: str, overflowing: list[bool]) -> dict:
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


def _add_device_divider(
    fig: go.Figure, *, row: int, col: int, backends: list[str]
) -> None:
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


def _scenario_column(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        pl.struct(["trajectory_id", "ncoils"])
        .map_elements(
            lambda s: scenario_label(s["trajectory_id"], s["ncoils"]),
            return_dtype=pl.Utf8,
        )
        .alias("scenario")
    )


# Runtime uncertainty is shown as an asymmetric error bar spanning the
# [p5, p95] replicate range around the median - wider and more honest about
# skew than a symmetric +/-MAD bar would be. Only the runtime metric has a
# per-replicate distribution; peak memory is a single max over replicates
# (see summary.py), so it has no equivalent uncertainty to show.
_RUNTIME_ERROR_COLS = ("runtime_p5_ms", "runtime_p95_ms")


def _error_bar(
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


def _metric_breakdown_figure(
    summary: pl.DataFrame,
    *,
    col: str,
    label: str,
    suite: str,
    error_cols: tuple[str, str] | None = None,
    x_unit: str | None = None,
) -> go.Figure:
    if summary.is_empty():
        return _empty_figure("No data yet")

    # host-only: cufinufft/gpunufft also have input_location=device rows for
    # forward/adjoint, which would otherwise double up bars per action.
    df = _scenario_column(
        summary.filter(
            (pl.col("input_location") == "host") & (pl.col("suite") == suite)
        )
    )
    if df.is_empty():
        return _empty_figure(f"No {suite}-suite data yet")

    present = set(df["scenario"].unique())
    scenarios = [s for s in SCENARIO_ORDER if s in present] + sorted(
        present - set(SCENARIO_ORDER)
    )
    backends = order_backends_by_device(df["backend"].unique().to_list())
    colors = {
        s: _QUALITATIVE_COLORS[i % len(_QUALITATIVE_COLORS)]
        for i, s in enumerate(scenarios)
    }
    actions = [a for a in _ACTIONS if a in set(df["action"].unique())]

    fig = make_subplots(
        rows=len(actions),
        cols=1,
        row_titles=actions,
        vertical_spacing=_row_spacing(len(actions)),
    )
    for r, action in enumerate(actions, start=1):
        sub = df.filter(pl.col("action") == action)
        if sub.is_empty():
            continue
        cap = _axis_cap(sub[col].to_list())
        for scenario in scenarios:
            scenario_sub = sub.filter(pl.col("scenario") == scenario)
            values_by_backend = dict(zip(*scenario_sub[["backend", col]]))
            xs = [values_by_backend.get(b) for b in backends]
            overflowing = [cap is not None and v is not None and v > cap for v in xs]
            error = _error_bar(scenario_sub, backends, col, error_cols)
            if error is not None and any(overflowing):
                # A capped bar's x is the cap, not its true value - an error
                # bar computed from the true value would be anchored at the
                # wrong position. The bar's text label already carries the
                # true value, so just drop the error bar for that one.
                error = {
                    "plus": [
                        0.0 if over else p
                        for over, p in zip(overflowing, error["plus"])
                    ],
                    "minus": [
                        0.0 if over else m
                        for over, m in zip(overflowing, error["minus"])
                    ],
                }
            fig.add_trace(
                go.Bar(
                    x=[cap if over else v for v, over in zip(xs, overflowing)],
                    y=backends,
                    orientation="h",
                    name=scenario,
                    marker=_clipped_marker(colors[scenario], overflowing),
                    legendgroup=scenario,
                    showlegend=(r == 1),
                    text=[
                        _format_value(v) if over else ""
                        for v, over in zip(xs, overflowing)
                    ],
                    textposition="inside",
                    insidetextanchor="end",
                    textfont=dict(color=_text_color(colors[scenario])),
                    error_x=(
                        None
                        if error is None
                        else dict(
                            type="data",
                            symmetric=False,
                            array=error["plus"],
                            arrayminus=error["minus"],
                            visible=True,
                            thickness=1.5,
                            width=3,
                        )
                    ),
                ),
                row=r,
                col=1,
            )
        fig.update_yaxes(
            autorange="reversed",
            categoryorder="array",
            categoryarray=backends,
            row=r,
            col=1,
        )
        if cap is not None:
            fig.update_xaxes(range=[0, cap], row=r, col=1)
        _add_device_divider(fig, row=r, col=1, backends=backends)

    if x_unit is not None:
        fig.update_xaxes(title_text=x_unit, row=len(actions), col=1)

    fig.update_layout(
        barmode="group",
        **_BAR_LAYOUT,
        height=480 * len(actions),
        width=1600,
        font=dict(size=15),
        title=dict(
            text=f"{label} (per scenario, per backend, per action, host input)",
            font=dict(size=22),
        ),
        legend=dict(font=dict(size=14)),
    )
    fig.update_annotations(font_size=18)
    return fig


def _input_location_figure(
    summary: pl.DataFrame,
    *,
    col: str,
    label: str,
    suite: str,
    error_cols: tuple[str, str] | None = None,
    x_unit: str | None = None,
) -> go.Figure:
    """Host vs GPU-resident (input_location=device) comparison.

    Only forward/adjoint carry device rows (see config.py's validation), and
    only for the cuda-capable backends that actually ran with
    input_location=device - other backends are dropped so every panel pairs
    like with like instead of showing a host-only bar next to nothing.
    """
    if summary.is_empty():
        return _empty_figure("No data yet")

    df = _scenario_column(
        summary.filter(
            (pl.col("suite") == suite) & pl.col("action").is_in(["forward", "adjoint"])
        )
    )
    device_backends = set(
        df.filter(pl.col("input_location") == "device")["backend"].unique()
    )
    if not device_backends:
        return _empty_figure(f"No GPU-resident input {suite}-suite data yet")
    df = df.filter(pl.col("backend").is_in(device_backends))

    present = set(df["scenario"].unique())
    scenarios = [s for s in SCENARIO_ORDER if s in present] + sorted(
        present - set(SCENARIO_ORDER)
    )
    backends = order_backends_by_device(df["backend"].unique().to_list())
    actions = [a for a in ("forward", "adjoint") if a in set(df["action"].unique())]
    locations = ["host", "device"]
    loc_colors = dict(zip(locations, _QUALITATIVE_COLORS))

    fig = make_subplots(
        rows=len(actions),
        cols=len(scenarios),
        row_titles=actions,
        column_titles=scenarios,
        shared_yaxes=True,
        vertical_spacing=_row_spacing(len(actions)),
    )
    for r, action in enumerate(actions, start=1):
        for c, scenario in enumerate(scenarios, start=1):
            sub = df.filter(
                (pl.col("action") == action) & (pl.col("scenario") == scenario)
            )
            if sub.is_empty():
                continue
            for loc in locations:
                loc_sub = sub.filter(pl.col("input_location") == loc)
                values_by_backend = dict(zip(*loc_sub[["backend", col]]))
                xs = [values_by_backend.get(b) for b in backends]
                error = _error_bar(loc_sub, backends, col, error_cols)
                fig.add_trace(
                    go.Bar(
                        x=xs,
                        y=backends,
                        orientation="h",
                        name=loc,
                        marker_color=loc_colors[loc],
                        legendgroup=loc,
                        showlegend=(r == 1 and c == 1),
                        error_x=(
                            None
                            if error is None
                            else dict(
                                type="data",
                                symmetric=False,
                                array=error["plus"],
                                arrayminus=error["minus"],
                                visible=True,
                                thickness=1.5,
                                width=3,
                            )
                        ),
                    ),
                    row=r,
                    col=c,
                )
            fig.update_yaxes(
                autorange="reversed",
                categoryorder="array",
                categoryarray=backends,
                row=r,
                col=c,
            )
            _add_device_divider(fig, row=r, col=c, backends=backends)

    if x_unit is not None:
        for c in range(1, len(scenarios) + 1):
            fig.update_xaxes(title_text=x_unit, row=len(actions), col=c)

    fig.update_layout(
        barmode="group",
        **_BAR_LAYOUT,
        height=420 * len(actions),
        width=max(420 * len(scenarios), 700),
        font=dict(size=15),
        title=dict(
            text=f"{label}: host vs GPU-resident input (forward/adjoint, cuda backends)",
            font=dict(size=22),
        ),
        legend=dict(font=dict(size=14)),
    )
    fig.update_annotations(font_size=16)
    return fig


# (family label, trajectory_ids) - the two families in the scaling dataset
# (see analysis.processing.scaling), used as the figure's column facets.
_SCALING_FAMILIES = [
    ("2D spiral", SPIRAL_2D_FAMILY),
    ("3D stack-of-spirals", SOS_3D_FAMILY),
]
# (variable, row label) - the figure's row facets.
_SCALING_VARIABLES = [("nx", "Image size (nx)"), ("ncoils", "Coil count")]


def _scaling_figure(
    scaling: pl.DataFrame,
    *,
    col: str,
    label: str,
    log_y: bool,
    y_unit: str | None = None,
) -> go.Figure:
    if scaling.is_empty():
        return _empty_figure("No data yet")

    df = scaling.filter(pl.col(col).is_not_null())
    if df.is_empty():
        return _empty_figure(f"No {label.lower()} data yet")

    present_trajectories = set(df["trajectory_id"].unique())
    families = [
        (flabel, fids)
        for flabel, fids in _SCALING_FAMILIES
        if fids & present_trajectories
    ]
    present_variables = set(df["variable"].unique())
    variables = [
        (v, vlabel) for v, vlabel in _SCALING_VARIABLES if v in present_variables
    ]
    if not families or not variables:
        return _empty_figure(f"No {label.lower()} data yet")

    backends = order_backends_by_device(df["backend"].unique().to_list())
    colors = {
        b: _QUALITATIVE_COLORS[i % len(_QUALITATIVE_COLORS)]
        for i, b in enumerate(backends)
    }

    fig = make_subplots(
        rows=len(variables),
        cols=len(families),
        row_titles=[vlabel for _, vlabel in variables],
        column_titles=[flabel for flabel, _ in families],
        vertical_spacing=_row_spacing(len(variables)),
    )
    for r, (variable, _) in enumerate(variables, start=1):
        for c, (_, family_ids) in enumerate(families, start=1):
            sub = df.filter(
                (pl.col("variable") == variable)
                & pl.col("trajectory_id").is_in(family_ids)
            )
            if sub.is_empty():
                continue
            for backend in backends:
                b_sub = sub.filter(pl.col("backend") == backend).sort("value")
                if b_sub.is_empty():
                    continue
                fig.add_trace(
                    go.Scatter(
                        x=b_sub["value"].to_list(),
                        y=b_sub[col].to_list(),
                        mode="lines+markers",
                        marker=dict(size=9),
                        line=dict(width=2),
                        name=backend,
                        marker_color=colors[backend],
                        legendgroup=backend,
                        showlegend=(r == 1 and c == 1),
                    ),
                    row=r,
                    col=c,
                )
            # Log x with explicit tickvals (the actual scenario values, e.g.
            # ncoils=1/8/32) rather than default log-decade ticks - the
            # values themselves are the point of a scaling plot.
            xvals = sorted(sub["value"].unique())
            fig.update_xaxes(
                type="log",
                tickvals=xvals,
                ticktext=[str(v) for v in xvals],
                row=r,
                col=c,
            )
            if log_y:
                fig.update_yaxes(type="log", row=r, col=c)

    if y_unit is not None:
        for r in range(1, len(variables) + 1):
            fig.update_yaxes(title_text=y_unit, row=r, col=1)

    fig.update_layout(
        height=420 * len(variables),
        width=max(520 * len(families), 700),
        font=dict(size=15),
        title=dict(text=f"{label} scaling (forward, host input)", font=dict(size=22)),
        legend=dict(font=dict(size=14)),
    )
    fig.update_annotations(font_size=16)
    return fig


# Coil-count-independent scenario families for the runtime-per-coil figure -
# fixed display order; anything else present falls back to alphabetical.
_RUNTIME_FAMILY_ORDER = ["2D-S", "2D-M", "2D-L", "3D-S", "3D-M", "3D-L"]
_NCOILS_ORDER = [1, 8, 32]
_NCOILS_COLORS = dict(zip(_NCOILS_ORDER, _QUALITATIVE_COLORS))


def _runtime_by_family_figure(summary: pl.DataFrame) -> go.Figure:
    """Runtime per coil, faceted by (action, trajectory family).

    Dividing by ncoils normalizes the embarrassingly-parallel per-coil NUFFT
    cost, so the 1/8/32-coil variants of the same trajectory become
    comparable bars within one panel instead of needing a separate scenario
    column each - collapsing the scenario axis down to the four base
    trajectory families (2D-S, 2D-M, 3D-S, 3D-M).
    """
    if summary.is_empty():
        return _empty_figure("No data yet")

    df = summary.filter(
        (pl.col("input_location") == "host") & (pl.col("suite") == "benchmark")
    )
    if df.is_empty():
        return _empty_figure("No benchmark-suite data yet")

    df = df.with_columns(
        (pl.col("runtime_median_ms") / pl.col("ncoils")).alias("runtime_per_coil_ms"),
        (pl.col("runtime_p5_ms") / pl.col("ncoils")).alias("runtime_per_coil_p5_ms"),
        (pl.col("runtime_p95_ms") / pl.col("ncoils")).alias("runtime_per_coil_p95_ms"),
        pl.col("trajectory_id")
        .map_elements(trajectory_label, return_dtype=pl.Utf8)
        .alias("family"),
    )
    col, error_cols = (
        "runtime_per_coil_ms",
        ("runtime_per_coil_p5_ms", "runtime_per_coil_p95_ms"),
    )

    present_families = set(df["family"].unique())
    families = [f for f in _RUNTIME_FAMILY_ORDER if f in present_families] + sorted(
        present_families - set(_RUNTIME_FAMILY_ORDER)
    )
    backends = order_backends_by_device(df["backend"].unique().to_list())
    actions = [a for a in _ACTIONS if a in set(df["action"].unique())]

    fig = make_subplots(
        rows=len(actions),
        cols=len(families),
        row_titles=actions,
        column_titles=families,
        shared_yaxes=True,
        vertical_spacing=_row_spacing(len(actions)),
    )
    legend_shown: set[int] = set()
    for r, action in enumerate(actions, start=1):
        for c, family in enumerate(families, start=1):
            sub = df.filter((pl.col("action") == action) & (pl.col("family") == family))
            if sub.is_empty():
                continue
            cap = _axis_cap(sub[col].to_list())
            for ncoils in sorted(sub["ncoils"].unique()):
                ncoils_sub = sub.filter(pl.col("ncoils") == ncoils)
                values_by_backend = dict(zip(*ncoils_sub[["backend", col]]))
                xs = [values_by_backend.get(b) for b in backends]
                overflowing = [
                    cap is not None and v is not None and v > cap for v in xs
                ]
                error = _error_bar(ncoils_sub, backends, col, error_cols)
                if error is not None and any(overflowing):
                    error = {
                        "plus": [
                            0.0 if over else p
                            for over, p in zip(overflowing, error["plus"])
                        ],
                        "minus": [
                            0.0 if over else m
                            for over, m in zip(overflowing, error["minus"])
                        ],
                    }
                show = ncoils not in legend_shown
                legend_shown.add(ncoils)
                ncoils_color = _NCOILS_COLORS.get(ncoils, _QUALITATIVE_COLORS[-1])
                fig.add_trace(
                    go.Bar(
                        x=[cap if over else v for v, over in zip(xs, overflowing)],
                        y=backends,
                        orientation="h",
                        name=f"{ncoils} coil{'s' if ncoils != 1 else ''}",
                        marker=_clipped_marker(ncoils_color, overflowing),
                        legendgroup=str(ncoils),
                        showlegend=show,
                        text=[
                            _format_value(v) if over else ""
                            for v, over in zip(xs, overflowing)
                        ],
                        textposition="inside",
                        insidetextanchor="end",
                        textfont=dict(color=_text_color(ncoils_color)),
                        error_x=(
                            None
                            if error is None
                            else dict(
                                type="data",
                                symmetric=False,
                                array=error["plus"],
                                arrayminus=error["minus"],
                                visible=True,
                                thickness=1.5,
                                width=3,
                            )
                        ),
                    ),
                    row=r,
                    col=c,
                )
            fig.update_yaxes(
                autorange="reversed",
                categoryorder="array",
                categoryarray=backends,
                row=r,
                col=c,
            )
            if cap is not None:
                fig.update_xaxes(range=[0, cap], row=r, col=c)
            _add_device_divider(fig, row=r, col=c, backends=backends)

    for c in range(1, len(families) + 1):
        fig.update_xaxes(title_text="ms / coil", row=len(actions), col=c)

    fig.update_layout(
        barmode="group",
        **_BAR_LAYOUT,
        height=420 * len(actions),
        width=max(420 * len(families), 700),
        font=dict(size=15),
        title=dict(
            text="Runtime per coil (ms) (per scenario family, per backend, per "
            "action, host input)",
            font=dict(size=22),
        ),
        legend=dict(font=dict(size=14)),
    )
    fig.update_annotations(font_size=16)
    return fig


def _read_yaml_dir(subdir: str) -> dict[str, dict]:
    out = {}
    for path in sorted((_CONFIGS_DIR / subdir).glob("*.yaml")):
        with open(path) as f:
            out[path.stem] = yaml.safe_load(f) or {}
    return out


def _table(headers: list[str], rows: list[list]) -> str:
    head = "".join(f"<th>{_esc(h)}</th>" for h in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{_esc(str(c))}</td>" for c in row) + "</tr>"
        for row in rows
    )
    return f"<table class='details-table'><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def _hardware_table(processed_dir: Path) -> str:
    metadata = load_metadata(processed_dir.parent)
    seen: dict[str, dict] = {}
    for data in metadata.values():
        hid = data.get("hardware_id")
        if hid and hid not in seen:
            seen[hid] = data
    if not seen:
        return "<p>No metadata yet</p>"
    rows = []
    for hid, data in sorted(seen.items()):
        hw, sw = data.get("hardware", {}), data.get("software", {})
        rows.append(
            [
                hid,
                hw.get("cpu", "-"),
                f"{hw.get('ram_gb', '-')} GB",
                hw.get("gpu") or "-",
                hw.get("cuda_version") or "-",
                hw.get("driver_version") or "-",
                sw.get("os", "-"),
                sw.get("python_version", "-"),
                sw.get("mri_nufft_version", "-"),
                (sw.get("git_sha") or "-")[:12],
            ]
        )
    return _table(
        [
            "Hardware ID",
            "CPU",
            "RAM",
            "GPU",
            "CUDA",
            "Driver",
            "OS",
            "Python",
            "mri-nufft",
            "git sha",
        ],
        rows,
    )


def _backend_table(summary: pl.DataFrame) -> str:
    configs = _read_yaml_dir("backend")
    versions: dict[str, set[str]] = {}
    if not summary.is_empty():
        for row in summary.select("backend", "backend_version").unique().to_dicts():
            versions.setdefault(row["backend"], set()).add(row["backend_version"])
    rows = []
    for name in order_backends_by_device(list(configs)):
        cfg = configs[name]
        version = ", ".join(sorted(versions.get(name, {"-"})))
        params = (
            ", ".join(f"{k}={v}" for k, v in (cfg.get("parameters") or {}).items())
            or "-"
        )
        rows.append([name, cfg.get("device", "-"), cfg.get("framework", "-"), version, params])
    return _table(["Backend", "Device", "Framework", "Version", "Parameters"], rows)


def _scenario_table(summary: pl.DataFrame) -> str:
    if summary.is_empty():
        return "<p>No data yet</p>"
    trajectories = _read_yaml_dir("trajectory")
    combos = (
        summary.select(
            "trajectory_id", "ndim", "nx", "ny", "nz", "ncoils", "smaps", "n_samples"
        )
        .unique()
        .to_dicts()
    )

    def sort_key(c: dict) -> tuple[int, str]:
        label = scenario_label(c["trajectory_id"], c["ncoils"])
        rank = SCENARIO_ORDER.index(label) if label in SCENARIO_ORDER else len(
            SCENARIO_ORDER
        )
        return (rank, label)

    combos.sort(key=sort_key)
    rows = []
    for c in combos:
        traj = trajectories.get(c["trajectory_id"], {})
        size = " x ".join(str(d) for d in (c["nx"], c["ny"], c["nz"]) if d)
        rows.append(
            [
                scenario_label(c["trajectory_id"], c["ncoils"]),
                c["trajectory_id"],
                traj.get("trajectory_type", "-"),
                c["ndim"],
                size,
                c["ncoils"],
                "yes" if c["smaps"] else "no",
                f"{c['n_samples']:,}",
                traj.get("n_shots", "-"),
                traj.get("density", "-"),
            ]
        )
    return _table(
        [
            "Scenario",
            "Trajectory ID",
            "Type",
            "ndim",
            "Image size",
            "Coils",
            "Smaps",
            "Samples",
            "Shots",
            "Density",
        ],
        rows,
    )


def _benchmark_suite_table() -> str:
    configs = _read_yaml_dir("benchmark")
    rows = [
        [
            name,
            cfg.get("warmup_min_iters", "-"),
            cfg.get("warmup_min_seconds", "-"),
            cfg.get("min_runtime_budget_s", "-"),
            cfg.get("max_repetitions", "-"),
        ]
        for name, cfg in sorted(configs.items())
    ]
    return _table(
        [
            "Suite",
            "Warmup min iters",
            "Warmup min seconds",
            "Min runtime budget (s)",
            "Max repetitions",
        ],
        rows,
    )


def _simple_list_note(subdir: str, title: str) -> str:
    items = ", ".join(sorted(_read_yaml_dir(subdir)))
    return f"<p><b>{_esc(title)}:</b> {_esc(items)}</p>"


def _reference_note() -> str:
    with open(_CONFIGS_DIR / "run.yaml") as f:
        run_cfg = yaml.safe_load(f)
    ref = run_cfg.get("reference", {})
    params = ", ".join(f"{k}={v}" for k, v in (ref.get("parameters") or {}).items())
    return (
        "<p>Accuracy metrics (relative L2/Linf error, adjointness) are computed "
        f"against backend <b>{_esc(str(ref.get('backend')))}</b> ({_esc(params)}).</p>"
    )


def _details_html(processed_dir: Path, summary: pl.DataFrame) -> str:
    sections = [
        ("Hardware & software", _hardware_table(processed_dir)),
        ("Backends", _backend_table(summary)),
        ("Scenarios", _scenario_table(summary)),
        ("Benchmark suites", _benchmark_suite_table()),
        (
            "Actions & input locations",
            _simple_list_note("action", "Actions")
            + _simple_list_note("input_location", "Input locations"),
        ),
        ("Accuracy reference", _reference_note()),
    ]
    parts = ["<div class='details'>"]
    for heading, body in sections:
        parts.append(f"<h2>{_esc(heading)}</h2>{body}")
    parts.append("</div>")
    return "".join(parts)


def build_report(processed_dir: Path, output_path: Path) -> None:
    processed_dir = Path(processed_dir)
    summary = pl.read_parquet(processed_dir / "summary.parquet")
    scaling = pl.read_parquet(processed_dir / "scaling.parquet")

    figures = {
        "runtime": _runtime_by_family_figure(summary),
        "gpu_memory": _metric_breakdown_figure(
            summary,
            col="peak_gpu_allocated_mb",
            label="Peak GPU mem (MB)",
            suite="memory",
        ),
        "cpu_memory": _metric_breakdown_figure(
            summary, col="peak_cpu_rss_mb", label="Peak CPU mem (MB)", suite="memory"
        ),
        "gpu_resident_runtime": _input_location_figure(
            summary,
            col="runtime_median_ms",
            label="Runtime (ms)",
            suite="benchmark",
            error_cols=_RUNTIME_ERROR_COLS,
            x_unit="ms",
        ),
        "gpu_resident_memory": _input_location_figure(
            summary,
            col="peak_gpu_allocated_mb",
            label="Peak GPU mem (MB)",
            suite="memory",
        ),
        "scaling_runtime": _scaling_figure(
            scaling,
            col="runtime_median_ms",
            label="Runtime (ms)",
            log_y=True,
            y_unit="ms",
        ),
        "scaling_memory": _scaling_figure(
            scaling, col="peak_gpu_allocated_mb", label="Peak GPU mem (MB)", log_y=False
        ),
        "details": _details_html(processed_dir, summary),
    }

    # Grouped so the tab bar reads as sections, not one flat row of 7+
    # same-looking buttons - each group is its own visual cluster.
    groups = [
        ("Details", ["details"]),
        ("Per-Scenario Breakdown", ["runtime", "gpu_memory", "cpu_memory"]),
        ("GPU-Resident Input", ["gpu_resident_runtime", "gpu_resident_memory"]),
        ("Scaling", ["scaling_runtime", "scaling_memory"]),
    ]

    buttons = ["<div class='tab-buttons'>"]
    panels = []
    plotlyjs_embedded = False
    first = True
    for group_label, names in groups:
        buttons.append(
            f"<div class='tab-group'><span class='tab-group-label'>{group_label}</span>"
        )
        for name in names:
            fig_or_html = figures[name]
            active = " active" if first else ""
            label = name.replace("_", " ").title()
            buttons.append(
                f"<button id='btn-{name}' class='{active.strip()}' "
                f"onclick=\"showTab('{name}')\">{label}</button>"
            )
            # The details tab is plain HTML (config/hardware tables), not a
            # Plotly figure - only figures need pio.to_html + the shared
            # plotly.js bundle.
            if isinstance(fig_or_html, go.Figure):
                fig_html = pio.to_html(
                    fig_or_html,
                    include_plotlyjs=(not plotlyjs_embedded),
                    full_html=False,
                )
                plotlyjs_embedded = True
            else:
                fig_html = fig_or_html
            panels.append(
                f"<div id='tab-{name}' class='tab-panel{active}'>{fig_html}</div>"
            )
            first = False
        buttons.append("</div>")
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
