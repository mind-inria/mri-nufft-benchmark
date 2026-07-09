"""Scaling figure: runtime/memory vs image size or coil count.

Draws the ``analysis.processing.scaling`` dataset as one line per backend,
faceted by trajectory family (columns) and scaling variable (rows).
"""

from __future__ import annotations

import plotly.graph_objects as go
import polars as pl
from plotly.subplots import make_subplots

from analysis.figures.common import QUALITATIVE_COLORS, empty_figure, row_spacing
from analysis.scenarios import SOS_3D_FAMILY, SPIRAL_2D_FAMILY, order_backends_by_device

# (family label, trajectory_ids) - the two families in the scaling dataset
# (see analysis.processing.scaling), used as the figure's column facets.
_SCALING_FAMILIES = [
    ("2D spiral", SPIRAL_2D_FAMILY),
    ("3D stack-of-spirals", SOS_3D_FAMILY),
]
# (variable, row label) - the figure's row facets.
_SCALING_VARIABLES = [("nx", "Image size (nx)"), ("ncoils", "Coil count")]


def scaling_figure(
    scaling: pl.DataFrame,
    *,
    col: str,
    label: str,
    log_y: bool,
    y_unit: str | None = None,
) -> go.Figure:
    if scaling.is_empty():
        return empty_figure("No data yet")

    df = scaling.filter(pl.col(col).is_not_null())
    if df.is_empty():
        return empty_figure(f"No {label.lower()} data yet")

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
        return empty_figure(f"No {label.lower()} data yet")

    backends = order_backends_by_device(df["backend"].unique().to_list())
    colors = {
        b: QUALITATIVE_COLORS[i % len(QUALITATIVE_COLORS)]
        for i, b in enumerate(backends)
    }
    # host is always present; device (GPU-resident input) only exists for
    # cuda backends that ran with input_location=device (see scaling.py) -
    # drawn as a dashed line in the same color as that backend's solid host
    # line, so the two read as "one backend, two input paths" rather than
    # two unrelated series.
    locations = [loc for loc in ("host", "device") if loc in set(df["input_location"].unique())]

    fig = make_subplots(
        rows=len(variables),
        cols=len(families),
        row_titles=[vlabel for _, vlabel in variables],
        column_titles=[flabel for flabel, _ in families],
        vertical_spacing=row_spacing(len(variables)),
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
                for loc in locations:
                    b_sub = sub.filter(
                        (pl.col("backend") == backend) & (pl.col("input_location") == loc)
                    ).sort("value")
                    if b_sub.is_empty():
                        continue
                    fig.add_trace(
                        go.Scatter(
                            x=b_sub["value"].to_list(),
                            y=b_sub[col].to_list(),
                            mode="lines+markers",
                            marker=dict(size=9, symbol="circle" if loc == "host" else "diamond"),
                            line=dict(width=2, dash="solid" if loc == "host" else "dash"),
                            name=backend if loc == "host" else f"{backend} (device)",
                            marker_color=colors[backend],
                            legendgroup=backend,
                            showlegend=(r == 1 and c == 1),
                            # trace-level meta (distinct from the figure's
                            # layout.meta filter tag) carries the bare
                            # backend name so the client-side filter can
                            # match both this trace's and its host
                            # counterpart's differently-labeled legend entry.
                            meta=backend,
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
        title=dict(
            text=f"{label} scaling (forward; dashed = GPU-resident input, where available)",
            font=dict(size=22),
        ),
        legend=dict(font=dict(size=14)),
        meta={"backend_filter": "trace_name"},
    )
    fig.update_annotations(font_size=16)
    return fig
