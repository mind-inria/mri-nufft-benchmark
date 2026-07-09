"""HTML detail tables: hardware/software, backends, scenarios, speedup.

These render plain HTML (not Plotly figures) for the report's "Details" tab
and the per-action speedup table - config/metadata lookups and small polars
aggregations turned directly into ``<table>`` markup.
"""

from __future__ import annotations

from html import escape as _esc
from pathlib import Path

import polars as pl
import yaml

from analysis.figures.common import ACTIONS, scenario_column
from analysis.loaders import load_metadata
from analysis.scenarios import SCENARIO_ORDER, order_backends_by_device, scenario_label

_CONFIGS_DIR = Path(__file__).resolve().parent.parent.parent / "configs"


def read_yaml_dir(subdir: str) -> dict[str, dict]:
    out = {}
    for path in sorted((_CONFIGS_DIR / subdir).glob("*.yaml")):
        with open(path) as f:
            out[path.stem] = yaml.safe_load(f) or {}
    return out


def table(headers: list[str], rows: list[list]) -> str:
    head = "".join(f"<th>{_esc(h)}</th>" for h in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{_esc(str(c))}</td>" for c in row) + "</tr>"
        for row in rows
    )
    return f"<table class='details-table'><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def backend_table_html(scenario_header: str, backends: list[str], rows: list[list]) -> str:
    """Like table(), but tags each backend's header/data cells with
    data-backend so the report's client-side filter bar can hide columns
    for backends the user unchecked."""
    head = f"<th>{_esc(scenario_header)}</th>" + "".join(
        f'<th data-backend="{_esc(b)}">{_esc(b)}</th>' for b in backends
    )
    body = "".join(
        "<tr><td>"
        + _esc(str(row[0]))
        + "</td>"
        + "".join(
            f'<td data-backend="{_esc(b)}">{_esc(str(v))}</td>'
            for b, v in zip(backends, row[1:])
        )
        + "</tr>"
        for row in rows
    )
    return (
        f"<table class='details-table speedup-table'><thead><tr>{head}</tr></thead>"
        f"<tbody>{body}</tbody></table>"
    )


def hardware_table(processed_dir: Path) -> str:
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
    return table(
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


def backend_table(summary: pl.DataFrame) -> str:
    configs = read_yaml_dir("backend")
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
    return table(["Backend", "Device", "Framework", "Version", "Parameters"], rows)


def scenario_table(summary: pl.DataFrame) -> str:
    if summary.is_empty():
        return "<p>No data yet</p>"
    trajectories = read_yaml_dir("trajectory")
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
    return table(
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


def benchmark_suite_table() -> str:
    configs = read_yaml_dir("benchmark")
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
    return table(
        [
            "Suite",
            "Warmup min iters",
            "Warmup min seconds",
            "Min runtime budget (s)",
            "Max repetitions",
        ],
        rows,
    )


def simple_list_note(subdir: str, title: str) -> str:
    items = ", ".join(sorted(read_yaml_dir(subdir)))
    return f"<p><b>{_esc(title)}:</b> {_esc(items)}</p>"


def reference_config() -> dict:
    with open(_CONFIGS_DIR / "run.yaml") as f:
        run_cfg = yaml.safe_load(f)
    return run_cfg.get("reference", {})


def reference_note() -> str:
    ref = reference_config()
    params = ", ".join(f"{k}={v}" for k, v in (ref.get("parameters") or {}).items())
    return (
        "<p>Accuracy metrics (relative L2/Linf error, adjointness) are computed "
        f"against backend <b>{_esc(str(ref.get('backend')))}</b> ({_esc(params)}).</p>"
    )


def speedup_table(summary: pl.DataFrame) -> str:
    """Per-scenario, per-backend, per-action speedup vs the reference backend.

    Uses the same reference backend as the accuracy metrics (see
    reference_note / configs/run.yaml's ``reference.backend``) so every
    number in the report is "compared against X" for a single, consistent X.
    Host-input runtime is the primary speedup; where a backend also ran with
    input_location=device (GPU-resident input, see the Runtime tab's overlay
    bars), that speedup is appended in parentheses in the same cell instead
    of a separate table, since it's the same (scenario, backend, action)
    triple with one number swapped.
    """
    if summary.is_empty():
        return "<p>No data yet</p>"

    df = summary.filter(
        (pl.col("suite") == "benchmark") & pl.col("runtime_median_ms").is_not_null()
    )
    if df.is_empty():
        return "<p>No benchmark-suite data yet</p>"
    df = scenario_column(df)

    reference = reference_config().get("backend")
    if not reference or reference not in set(df["backend"].unique()):
        return "<p>Reference backend has no benchmark-suite data yet</p>"

    backends = order_backends_by_device(df["backend"].unique().to_list())
    present_scenarios = set(df["scenario"].unique())
    scenarios = [s for s in SCENARIO_ORDER if s in present_scenarios] + sorted(
        present_scenarios - set(SCENARIO_ORDER)
    )
    actions = [a for a in ACTIONS if a in set(df["action"].unique())]

    parts = [
        "<p>Speedup relative to <b>"
        + _esc(reference)
        + "</b> (reference runtime / backend runtime, host input); value in "
        "parentheses is the GPU-resident-input speedup, where measured.</p>"
    ]
    for action in actions:
        action_df = df.filter(pl.col("action") == action)
        host_df = action_df.filter(pl.col("input_location") == "host")
        device_df = action_df.filter(pl.col("input_location") == "device")
        ref_by_scenario = dict(
            zip(*host_df.filter(pl.col("backend") == reference)[
                ["scenario", "runtime_median_ms"]
            ])
        )

        rows = []
        for scenario in scenarios:
            ref_time = ref_by_scenario.get(scenario)
            scenario_host = host_df.filter(pl.col("scenario") == scenario)
            scenario_device = device_df.filter(pl.col("scenario") == scenario)
            host_by_backend = dict(
                zip(*scenario_host[["backend", "runtime_median_ms"]])
            )
            device_by_backend = dict(
                zip(*scenario_device[["backend", "runtime_median_ms"]])
            )
            row = [scenario]
            for backend in backends:
                host_time = host_by_backend.get(backend)
                if ref_time is None or not host_time:
                    row.append("-")
                    continue
                cell = f"{ref_time / host_time:.2f}x"
                device_time = device_by_backend.get(backend)
                if device_time:
                    cell += f" ({ref_time / device_time:.2f}x device)"
                row.append(cell)
            rows.append(row)

        parts.append(
            f"<h3>{_esc(action)}</h3>" + backend_table_html("Scenario", backends, rows)
        )

    return "".join(parts)


def details_html(processed_dir: Path, summary: pl.DataFrame) -> str:
    sections = [
        ("Hardware & software", hardware_table(processed_dir)),
        ("Backends", backend_table(summary)),
        ("Scenarios", scenario_table(summary)),
        ("Benchmark suites", benchmark_suite_table()),
        (
            "Actions & input locations",
            simple_list_note("action", "Actions")
            + simple_list_note("input_location", "Input locations"),
        ),
        ("Accuracy reference", reference_note()),
    ]
    parts = ["<div class='details'>"]
    for heading, body in sections:
        parts.append(f"<h2>{_esc(heading)}</h2>{body}")
    parts.append("</div>")
    return "".join(parts)
