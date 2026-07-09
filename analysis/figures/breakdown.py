"""Per-scenario runtime/memory breakdown figures.

``metric_by_family_figure`` (and its ``runtime_by_family_figure`` wrapper)
facets a metric by trajectory family x action, grouped by ncoils - the
report's main "Runtime"/"GPU mem"/"CPU mem" tabs. ``input_location_figure``
is the separate host-vs-GPU-resident-input comparison used by the
"GPU-Resident Input" tab.
"""

from __future__ import annotations

import plotly.graph_objects as go
import polars as pl
from plotly.subplots import make_subplots

from analysis.figures.common import (
    ACTIONS,
    BAR_LAYOUT,
    QUALITATIVE_COLORS,
    add_device_divider,
    axis_cap,
    clipped_marker,
    empty_figure,
    error_bar,
    format_value,
    row_spacing,
    scenario_column,
    text_color,
)
from analysis.scenarios import SCENARIO_ORDER, order_backends_by_device, trajectory_label

# Coil-count-independent scenario families for the runtime-per-coil figure -
# fixed display order; anything else present falls back to alphabetical.
_RUNTIME_FAMILY_ORDER = ["2D-S", "2D-M", "2D-L", "3D-S", "3D-M", "3D-L"]
_NCOILS_ORDER = [1, 8, 32]
_NCOILS_COLORS = dict(zip(_NCOILS_ORDER, QUALITATIVE_COLORS))

# Fixed color for the GPU-resident-input overlay bar, deliberately outside
# QUALITATIVE_COLORS - it means the same thing (device-resident input) in
# every ncoils group, unlike the ncoils colors it's inset into, so it must
# not be mistaken for one of them.
_DEVICE_OVERLAY_COLOR = "#2b2b2b"
_DEVICE_OVERLAY_WIDTH_FRACTION = 0.45


def runtime_by_family_figure(summary: pl.DataFrame) -> go.Figure:
    return metric_by_family_figure(
        summary,
        col="runtime_median_ms",
        error_cols=("runtime_p5_ms", "runtime_p95_ms"),
        label="Runtime per coil (ms)",
        suite="benchmark",
        x_unit="ms / coil",
        per_coil=True,
        device_overlay=True,
    )


def metric_by_family_figure(
    summary: pl.DataFrame,
    *,
    col: str,
    label: str,
    suite: str,
    error_cols: tuple[str, str] | None = None,
    x_unit: str | None = None,
    per_coil: bool = False,
    device_overlay: bool = False,
) -> go.Figure:
    """A metric faceted by (action, trajectory family), grouped by ncoils.

    Collapses the scenario axis down to the four base trajectory families
    (2D-S, 2D-M, 3D-S, 3D-M) with the 1/8/32-coil variants of each family
    shown as adjacent grouped bars within one panel, instead of needing a
    separate scenario column each.

    When ``per_coil`` is set (runtime), values are divided by ncoils first -
    this normalizes the embarrassingly-parallel per-coil NUFFT cost so the
    ncoils variants become directly comparable. Memory does not scale this
    way, so callers for memory metrics leave this off.

    When ``device_overlay`` is set (runtime only) and cuda backends also ran
    forward/adjoint with input_location=device, that GPU-resident value is
    drawn as a narrower bar inset in the same slot as its host-input bar
    (same offsetgroup, smaller width) rather than as a separate panel - a
    compact "bar within a bar" that reads directly as "this is how much of
    the host bar is transfer overhead".
    """
    if summary.is_empty():
        return empty_figure("No data yet")

    all_df = summary.filter(pl.col("suite") == suite)
    if all_df.is_empty():
        return empty_figure(f"No {suite}-suite data yet")

    if per_coil:
        with_cols = [(pl.col(col) / pl.col("ncoils")).alias("_value")]
        if error_cols is not None:
            lo_col, hi_col = error_cols
            with_cols += [
                (pl.col(lo_col) / pl.col("ncoils")).alias("_value_lo"),
                (pl.col(hi_col) / pl.col("ncoils")).alias("_value_hi"),
            ]
            error_cols = ("_value_lo", "_value_hi")
        all_df = all_df.with_columns(*with_cols)
        col = "_value"

    all_df = all_df.with_columns(
        pl.col("trajectory_id")
        .map_elements(trajectory_label, return_dtype=pl.Utf8)
        .alias("family"),
    )

    df = all_df.filter(pl.col("input_location") == "host")
    if df.is_empty():
        return empty_figure(f"No {suite}-suite data yet")
    device_df = (
        all_df.filter(pl.col("input_location") == "device")
        if device_overlay
        else all_df.clear()
    )

    present_families = set(df["family"].unique())
    families = [f for f in _RUNTIME_FAMILY_ORDER if f in present_families] + sorted(
        present_families - set(_RUNTIME_FAMILY_ORDER)
    )
    backends = order_backends_by_device(df["backend"].unique().to_list())
    actions = [a for a in ACTIONS if a in set(df["action"].unique())]

    fig = make_subplots(
        rows=len(actions),
        cols=len(families),
        row_titles=actions,
        column_titles=families,
        shared_yaxes=True,
        vertical_spacing=row_spacing(len(actions)),
    )
    legend_shown: set[int] = set()
    device_legend_shown = False
    for r, action in enumerate(actions, start=1):
        for c, family in enumerate(families, start=1):
            sub = df.filter((pl.col("action") == action) & (pl.col("family") == family))
            if sub.is_empty():
                continue
            device_sub_ac = device_df.filter(
                (pl.col("action") == action) & (pl.col("family") == family)
            )
            cap = axis_cap(sub[col].to_list())
            ncoils_values = sorted(sub["ncoils"].unique())
            # Explicit widths (rather than Plotly's auto-sizing) so the
            # device overlay bar can be given the exact same offsetgroup as
            # its host bar and reliably render centered inside it.
            bar_width = 0.8 / max(len(ncoils_values), 1)
            device_width = bar_width * _DEVICE_OVERLAY_WIDTH_FRACTION
            for ncoils in ncoils_values:
                ncoils_sub = sub.filter(pl.col("ncoils") == ncoils)
                values_by_backend = dict(zip(*ncoils_sub[["backend", col]]))
                xs = [values_by_backend.get(b) for b in backends]
                overflowing = [
                    cap is not None and v is not None and v > cap for v in xs
                ]
                error = error_bar(ncoils_sub, backends, col, error_cols)
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
                ncoils_color = _NCOILS_COLORS.get(ncoils, QUALITATIVE_COLORS[-1])
                fig.add_trace(
                    go.Bar(
                        x=[cap if over else v for v, over in zip(xs, overflowing)],
                        y=backends,
                        orientation="h",
                        name=f"{ncoils} coil{'s' if ncoils != 1 else ''}",
                        marker=clipped_marker(ncoils_color, overflowing),
                        legendgroup=str(ncoils),
                        showlegend=show,
                        offsetgroup=str(ncoils),
                        width=bar_width,
                        text=[
                            format_value(v) if over else ""
                            for v, over in zip(xs, overflowing)
                        ],
                        textposition="inside",
                        insidetextanchor="end",
                        textfont=dict(color=text_color(ncoils_color)),
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

                device_ncoils_sub = device_sub_ac.filter(pl.col("ncoils") == ncoils)
                if device_ncoils_sub.is_empty():
                    continue
                dev_values_by_backend = dict(
                    zip(*device_ncoils_sub[["backend", col]])
                )
                dev_xs = [dev_values_by_backend.get(b) for b in backends]
                if all(v is None for v in dev_xs):
                    continue
                dev_xs = [
                    cap if (cap is not None and v is not None and v > cap) else v
                    for v in dev_xs
                ]
                dev_error = error_bar(device_ncoils_sub, backends, col, error_cols)
                fig.add_trace(
                    go.Bar(
                        x=dev_xs,
                        y=backends,
                        orientation="h",
                        name="GPU-resident input",
                        marker=dict(
                            color=_DEVICE_OVERLAY_COLOR,
                            line=dict(color="white", width=0.5),
                        ),
                        legendgroup="device_resident",
                        showlegend=not device_legend_shown,
                        offsetgroup=str(ncoils),
                        width=device_width,
                        error_x=(
                            None
                            if dev_error is None
                            else dict(
                                type="data",
                                symmetric=False,
                                array=dev_error["plus"],
                                arrayminus=dev_error["minus"],
                                visible=True,
                                thickness=1,
                                width=2,
                            )
                        ),
                    ),
                    row=r,
                    col=c,
                )
                device_legend_shown = True
            fig.update_yaxes(
                autorange="reversed",
                categoryorder="array",
                categoryarray=backends,
                row=r,
                col=c,
            )
            if cap is not None:
                fig.update_xaxes(range=[0, cap], row=r, col=c)
            add_device_divider(fig, row=r, col=c, backends=backends)

    if x_unit is not None:
        for c in range(1, len(families) + 1):
            fig.update_xaxes(title_text=x_unit, row=len(actions), col=c)

    title = f"{label} (per scenario family, per backend, per action, host input"
    title += (
        "; narrow dark bar = GPU-resident input, where available)"
        if device_overlay
        else ")"
    )
    fig.update_layout(
        barmode="group",
        **BAR_LAYOUT,
        height=420 * len(actions),
        width=max(420 * len(families), 700),
        font=dict(size=15),
        title=dict(text=title, font=dict(size=22)),
        legend=dict(font=dict(size=14)),
        meta={"backend_filter": "y"},
    )
    fig.update_annotations(font_size=16)
    return fig


def input_location_figure(
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
        return empty_figure("No data yet")

    df = scenario_column(
        summary.filter(
            (pl.col("suite") == suite) & pl.col("action").is_in(["forward", "adjoint"])
        )
    )
    device_backends = set(
        df.filter(pl.col("input_location") == "device")["backend"].unique()
    )
    if not device_backends:
        return empty_figure(f"No GPU-resident input {suite}-suite data yet")
    df = df.filter(pl.col("backend").is_in(device_backends))

    present = set(df["scenario"].unique())
    scenarios = [s for s in SCENARIO_ORDER if s in present] + sorted(
        present - set(SCENARIO_ORDER)
    )
    backends = order_backends_by_device(df["backend"].unique().to_list())
    actions = [a for a in ("forward", "adjoint") if a in set(df["action"].unique())]
    locations = ["host", "device"]
    loc_colors = dict(zip(locations, QUALITATIVE_COLORS))

    fig = make_subplots(
        rows=len(actions),
        cols=len(scenarios),
        row_titles=actions,
        column_titles=scenarios,
        shared_yaxes=True,
        vertical_spacing=row_spacing(len(actions)),
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
                error = error_bar(loc_sub, backends, col, error_cols)
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
            add_device_divider(fig, row=r, col=c, backends=backends)

    if x_unit is not None:
        for c in range(1, len(scenarios) + 1):
            fig.update_xaxes(title_text=x_unit, row=len(actions), col=c)

    fig.update_layout(
        barmode="group",
        **BAR_LAYOUT,
        height=420 * len(actions),
        width=max(420 * len(scenarios), 700),
        font=dict(size=15),
        title=dict(
            text=f"{label}: host vs GPU-resident input (forward/adjoint, cuda backends)",
            font=dict(size=22),
        ),
        legend=dict(font=dict(size=14)),
        meta={"backend_filter": "y"},
    )
    fig.update_annotations(font_size=16)
    return fig
