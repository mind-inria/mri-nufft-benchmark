"""Hydra entry point. Ties together backends/, infra/, runner.py and suites/
into one run: setup -> warmup+timed reps -> measure/validate once -> write.
"""

from __future__ import annotations

import importlib.metadata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import hydra
from omegaconf import DictConfig, OmegaConf

from benchmark.config import BenchmarkConfig, register_configs
from benchmark.infra.hardware import collect_hardware_info
from benchmark.infra.memory import MemoryProbe
from benchmark.infra.synchronize import release_gpu_memory
from benchmark.infra.writer import ResultWriter
from benchmark.runner import BenchmarkRunner
from benchmark.suites.benchmark import BenchmarkSuiteImpl
from benchmark.suites.memory import MemorySuiteImpl

register_configs()

# Our backend name -> pip distribution name, for backends where they differ
# (e.g. the "pynfft" backend binds to the pyNFFT3 package, and
# "torchkbnufft-cpu"/"torchkbnufft-gpu" both bind to the single
# "torchkbnufft" package - see benchmark/backends/registry.py). Falls back
# to the backend name itself.
_PACKAGE_DISTRIBUTION_NAMES: dict[str, str] = {
    "pynfft": "pyNFFT3",
    "torchkbnufft-cpu": "torchkbnufft",
    "torchkbnufft-gpu": "torchkbnufft",
}


def _base_row(
    config: BenchmarkConfig, replicate: int, runtime_ms: float, git_sha: str | None
) -> dict[str, Any]:
    image_size = config.mri_setup.image_size
    distribution_name = _PACKAGE_DISTRIBUTION_NAMES.get(
        config.backend.name, config.backend.name
    )
    return {
        "suite": config.suite,
        "backend": config.backend.name,
        "backend_version": importlib.metadata.version(distribution_name),
        "ndim": config.mri_setup.ndim,
        "nx": image_size[0],
        "ny": image_size[1] if len(image_size) > 1 else None,
        "nz": image_size[2] if len(image_size) > 2 else None,
        "ncoils": config.mri_setup.ncoils,
        "smaps": config.mri_setup.smaps,
        "trajectory_id": config.trajectory.trajectory_id,
        "trajectory_type": config.trajectory.trajectory_type,
        "n_samples": config.trajectory.n_samples,
        "n_shots": config.trajectory.n_shots,
        "density": config.trajectory.density,
        "action": config.action.name,
        "input_location": config.input_location.name,
        "replicate": replicate,
        "runtime_ms": runtime_ms,
        "git_sha": git_sha,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def _collect_rows(
    config: BenchmarkConfig,
    git_sha: str | None,
    runner: BenchmarkRunner,
    run_fn: Callable[[], Any],
    row_extra: Callable[[], dict[str, Any]],
) -> tuple[Any, list[dict[str, Any]]]:
    """Run warmup+timed reps, building one result row per repetition.

    ``row_extra()`` is called once per repetition (after its timed call) to
    fill in suite-specific fields on top of ``_base_row``'s scenario/timing
    fields. Returns (last_result, rows) so a benchmark-suite caller can run
    validate() once on the final rep's result.
    """
    rows: list[dict[str, Any]] = []

    def on_rep(_result: Any, elapsed_ms: float) -> None:
        row = _base_row(config, len(rows), elapsed_ms, git_sha)
        row.update(row_extra())
        rows.append(row)

    last_result = runner.run(run_fn, config.benchmark, on_rep=on_rep)
    return last_result, rows


def _run_benchmark_suite(
    config: BenchmarkConfig, repo_root: Path, git_sha: str | None
) -> list[dict[str, Any]]:
    suite = BenchmarkSuiteImpl()
    state = suite.setup(config, repo_root)
    runner = BenchmarkRunner(config.backend.framework, config.backend.device)

    last_result, rows = _collect_rows(
        config,
        git_sha,
        runner,
        run_fn=lambda: suite.run(state),
        row_extra=lambda: {
            "cpu_rss_mb": None,
            "gpu_allocated_mb": None,
            "gpu_driver_mb": None,
        },
    )

    validation = suite.validate(last_result, state, config)
    reference_backend = (
        None if config.action.name == "operator_init" else config.reference.backend
    )
    for row in rows:
        row.update(
            {
                "reference_backend": reference_backend,
                "relative_l2_error": validation.relative_l2_error,
                "relative_linf_error": validation.relative_linf_error,
                "adjointness_error": validation.adjointness_error,
                "validation_passed": validation.passed,
            }
        )
    return rows


def _run_memory_suite(
    config: BenchmarkConfig, repo_root: Path, git_sha: str | None
) -> list[dict[str, Any]]:
    suite = MemorySuiteImpl()
    state = suite.setup(config, repo_root)
    runner = BenchmarkRunner(config.backend.framework, config.backend.device)
    probe = MemoryProbe(config.backend.framework, config.backend.device)

    def run_fn() -> Any:
        probe.start()
        return suite.run(state)

    def row_extra() -> dict[str, Any]:
        mem = probe.stop()
        # Key order must match _run_benchmark_suite's row_extra - raw
        # parquet files are concatenated across runs by column position, not
        # just name (see analysis/loaders/raw.py), so the two suites must
        # agree on schema layout.
        return {
            "cpu_rss_mb": mem["cpu_rss_mb"],
            "gpu_allocated_mb": mem["gpu_allocated_mb"],
            "gpu_driver_mb": mem["gpu_driver_mb"],
            "reference_backend": None,
            "relative_l2_error": None,
            "relative_linf_error": None,
            "adjointness_error": None,
            "validation_passed": None,
        }

    _, rows = _collect_rows(config, git_sha, runner, run_fn=run_fn, row_extra=row_extra)
    return rows


@hydra.main(config_path="configs", config_name="run", version_base=None)
def main(cfg: DictConfig) -> None:
    config: BenchmarkConfig = OmegaConf.to_object(cfg)
    repo_root = Path(hydra.utils.get_original_cwd())

    hardware_info = collect_hardware_info(repo_root)
    git_sha = hardware_info["software"]["git_sha"]

    if config.suite == "benchmark":
        rows = _run_benchmark_suite(config, repo_root, git_sha)
    else:
        rows = _run_memory_suite(config, repo_root, git_sha)

    writer = ResultWriter(repo_root / "benchmark_results")
    run_id = writer.write(
        rows, OmegaConf.to_container(cfg, resolve=True), hardware_info
    )
    print(f"wrote {len(rows)} rows for run_id={run_id}")

    # This job's operator/state have gone out of scope by now (they only
    # lived inside _run_*_suite's locals) - safe to hand the freed blocks
    # back to the driver before Hydra's multirun launcher runs the next job
    # in this same process. See release_gpu_memory's docstring.
    release_gpu_memory(config.backend.framework, config.backend.device)


if __name__ == "__main__":
    main()
