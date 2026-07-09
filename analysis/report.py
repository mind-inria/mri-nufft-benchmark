"""Build the single self-contained report.html from processed/*.parquet.

Plotly is embedded inline (not via CDN) so the report opens with no network
access, matching README's "single self-contained report.html". Figure and
table building live in analysis.figures.* - this module only assembles them
into the tabbed page (CSS/JS chrome + tab/panel wiring).
"""

from __future__ import annotations

from html import escape as _esc
from pathlib import Path

import plotly.graph_objects as go
import plotly.io as pio
import polars as pl

from analysis.figures.breakdown import (
    input_location_figure,
    metric_by_family_figure,
    runtime_by_family_figure,
)
from analysis.figures.scaling import scaling_figure
from analysis.figures.tables import details_html, speedup_table
from analysis.scenarios import BACKEND_DEVICES, order_backends_by_device

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
.backend-filter { display: flex; flex-wrap: wrap; align-items: center; gap: 4px 12px; margin-bottom: 1rem; padding: 6px 10px; border: 1px solid #ccc; border-radius: 4px; }
.backend-check-label { font-size: 0.85rem; white-space: nowrap; }
.backend-filter-actions { margin-left: auto; display: flex; gap: 6px; }
.backend-filter-actions button { padding: 3px 10px; font-size: 0.8rem; cursor: pointer; }
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

# Filtering strategy per figure, read from each go.Figure's layout.meta at
# render time (see the figure builders' fig.update_layout(meta=...) calls):
#  - "y": bar traces where trace.y holds one backend name per element (all
#    the per-scenario/per-action breakdown charts) - filtering nulls out
#    x/text/error at the excluded positions rather than removing them, so
#    the category axis layout never has to be recomputed.
#  - "trace_name": one trace per (backend, input_location), keyed by
#    trace.meta (the scaling line charts) - filtering just toggles trace
#    visibility. Falls back to trace.name for any other figure that reuses
#    this mode without setting per-trace meta.
# Original (unfiltered) trace data is snapshotted once per plot on first use
# and reused on every later toggle, since Plotly.restyle mutates gd.data.
#
# Everything below is batched into exactly one Plotly.restyle call per
# figure (covering every trace at once via the third "trace indices"
# argument) rather than one call per trace - a single multi-trace figure can
# have 100+ traces, and restyle triggers a full redraw each time it's
# called, so one-call-per-trace was locking up the tab on every filter
# change (and even on page load, before this rewrote the DOMContentLoaded
# handler to skip work entirely when every backend is checked).
_FILTER_JS = """
<script>
function applyBackendFilter() {
  var selected = new Set(
    Array.from(document.querySelectorAll('.backend-check:checked')).map((cb) => cb.value)
  );
  var allChecked =
    document.querySelectorAll('.backend-check:checked').length ===
    document.querySelectorAll('.backend-check').length;

  window._origTraceData = window._origTraceData || {};
  document.querySelectorAll('.js-plotly-plot').forEach((gd) => {
    var meta = gd.layout && gd.layout.meta;
    if (!meta || !meta.backend_filter) return;
    if (!window._origTraceData[gd.id]) {
      window._origTraceData[gd.id] = JSON.parse(JSON.stringify(gd.data));
    }
    // Nothing to filter yet (default/"All" state) - the snapshot above is
    // enough to make later toggles cheap, skip the redraw round-trip.
    if (allChecked) return;

    var orig = window._origTraceData[gd.id];
    var traceIdx = orig.map((_, i) => i);

    if (meta.backend_filter === 'trace_name') {
      var visible = orig.map((t) => selected.has(t.meta || t.name));
      Plotly.restyle(gd, { visible: visible }, traceIdx);
      return;
    }

    var filterArr = (arr, mask) => (arr ? arr.map((v, j) => (mask[j] ? v : null)) : arr);
    var xs = [], texts = [], errArrays = [], errArraysMinus = [];
    var hasText = false, hasError = false;
    orig.forEach((trace) => {
      var mask = (trace.y || []).map((b) => selected.has(b));
      xs.push(filterArr(trace.x, mask));
      if (trace.text) {
        hasText = true;
        texts.push(filterArr(trace.text, mask));
      } else {
        texts.push(trace.text);
      }
      if (trace.error_x) {
        hasError = true;
        errArrays.push(filterArr(trace.error_x.array, mask));
        errArraysMinus.push(filterArr(trace.error_x.arrayminus, mask));
      } else {
        errArrays.push(undefined);
        errArraysMinus.push(undefined);
      }
    });
    var update = { x: xs };
    if (hasText) update.text = texts;
    if (hasError) {
      update['error_x.array'] = errArrays;
      update['error_x.arrayminus'] = errArraysMinus;
    }
    Plotly.restyle(gd, update, traceIdx);
  });

  document
    .querySelectorAll("table.speedup-table th[data-backend], table.speedup-table td[data-backend]")
    .forEach((el) => {
      el.style.display = selected.has(el.dataset.backend) ? '' : 'none';
    });
}

function setBackendFilter(mode) {
  document.querySelectorAll('.backend-check').forEach((cb) => {
    if (mode === 'all') cb.checked = true;
    else if (mode === 'gpu') cb.checked = cb.dataset.device === 'cuda';
    else if (mode === 'cpu') cb.checked = cb.dataset.device !== 'cuda';
  });
  applyBackendFilter();
}

document.addEventListener('DOMContentLoaded', applyBackendFilter);
</script>
"""


def _backend_filter_bar(backends: list[str]) -> str:
    if not backends:
        return ""
    checkboxes = "".join(
        f'<label class="backend-check-label"><input type="checkbox" class="backend-check" '
        f'value="{_esc(b)}" data-device="{_esc(BACKEND_DEVICES.get(b, "cpu"))}" checked '
        f'onchange="applyBackendFilter()"> {_esc(b)}</label>'
        for b in backends
    )
    return (
        "<div class='backend-filter'>"
        "<span class='tab-group-label'>Backends</span>"
        f"{checkboxes}"
        "<span class='backend-filter-actions'>"
        "<button type='button' onclick=\"setBackendFilter('all')\">All</button>"
        "<button type='button' onclick=\"setBackendFilter('gpu')\">GPU only</button>"
        "<button type='button' onclick=\"setBackendFilter('cpu')\">CPU only</button>"
        "</span></div>"
    )


def build_report(processed_dir: Path, output_path: Path) -> None:
    processed_dir = Path(processed_dir)
    summary = pl.read_parquet(processed_dir / "summary.parquet")
    scaling = pl.read_parquet(processed_dir / "scaling.parquet")

    figures = {
        "runtime": runtime_by_family_figure(summary),
        "speedup": speedup_table(summary),
        "gpu_memory": metric_by_family_figure(
            summary,
            col="peak_gpu_allocated_mb",
            label="Peak GPU mem (MB)",
            suite="memory",
            x_unit="MB",
        ),
        "cpu_memory": metric_by_family_figure(
            summary,
            col="peak_cpu_rss_mb",
            label="Peak CPU mem (MB)",
            suite="memory",
            x_unit="MB",
        ),
        "gpu_resident_memory": input_location_figure(
            summary,
            col="peak_gpu_allocated_mb",
            label="Peak GPU mem (MB)",
            suite="memory",
        ),
        "scaling_runtime": scaling_figure(
            scaling,
            col="runtime_median_ms",
            label="Runtime (ms)",
            log_y=True,
            y_unit="ms",
        ),
        "scaling_memory": scaling_figure(
            scaling, col="peak_gpu_allocated_mb", label="Peak GPU mem (MB)", log_y=False
        ),
        "details": details_html(processed_dir, summary),
    }

    # Grouped so the tab bar reads as sections, not one flat row of 7+
    # same-looking buttons - each group is its own visual cluster.
    groups = [
        ("Details", ["details"]),
        ("Per-Scenario Breakdown", ["runtime", "speedup", "gpu_memory", "cpu_memory"]),
        ("GPU-Resident Input", ["gpu_resident_memory"]),
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

    all_backends = (
        order_backends_by_device(summary["backend"].unique().to_list())
        if not summary.is_empty()
        else []
    )

    html = (
        "<html><head><title>mri-nufft-benchmark report</title>"
        + _TABS_CSS
        + "</head><body><h1>mri-nufft-benchmark report</h1>"
        + _backend_filter_bar(all_backends)
        + "".join(buttons)
        + "".join(panels)
        + _TABS_JS
        + _FILTER_JS
        + "</body></html>"
    )

    Path(output_path).write_text(html)
