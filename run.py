"""Hydra entry point. Ties together backends/, infra/, runner.py and suites/
into one run: setup -> warmup+timed reps -> measure/validate once -> write.
"""

from __future__ import annotations

import importlib.metadata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import hydra
from omegaconf import DictConfig, OmegaConf

from benchmark.config import BenchmarkConfig, register_configs
from benchmark.infra.hardware import collect_hardware_info
from benchmark.infra.memory import MemoryProbe
from benchmark.infra.writer import ResultWriter
from benchmark.runner import BenchmarkRunner
from benchmark.suites.benchmark import BenchmarkSuiteImpl, uses_ndft_reference
from benchmark.suites.memory import MemorySuiteImpl

register_configs()


def _base_row(config: BenchmarkConfig, replicate: int, runtime_ms: float, git_sha: str | None) -> dict[str, Any]:
    image_size = config.mri_setup.image_size
    return {
        "suite": config.suite,
        "backend": config.backend.name,
        "backend_version": importlib.metadata.version(config.backend.name),
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
        "replicate": replicate,
        "runtime_ms": runtime_ms,
        "git_sha": git_sha,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def _run_benchmark_suite(
    config: BenchmarkConfig, repo_root: Path, git_sha: str | None
) -> list[dict[str, Any]]:
    suite = BenchmarkSuiteImpl()
    state = suite.setup(config, repo_root)
    runner = BenchmarkRunner(config.backend.framework, config.backend.device)

    rows: list[dict[str, Any]] = []

    def on_rep(_result: Any, elapsed_ms: float) -> None:
        rows.append(_base_row(config, len(rows), elapsed_ms, git_sha))

    last_result = runner.run(lambda: suite.run(state), config.benchmark, on_rep=on_rep)

    validation = suite.validate(last_result, state, config)
    reference_backend = (
        None
        if config.action.name == "operator_init"
        else ("numpy" if uses_ndft_reference(tuple(config.mri_setup.image_size)) else config.reference.backend)
    )
    for row in rows:
        row.update(
            {
                "cpu_rss_mb": None,
                "gpu_allocated_mb": None,
                "gpu_driver_mb": None,
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

    rows: list[dict[str, Any]] = []

    def on_rep(_result: Any, elapsed_ms: float) -> None:
        mem = probe.stop()
        row = _base_row(config, len(rows), elapsed_ms, git_sha)
        row.update(
            {
                **mem,
                "reference_backend": None,
                "relative_l2_error": None,
                "relative_linf_error": None,
                "adjointness_error": None,
                "validation_passed": None,
            }
        )
        rows.append(row)

    def run_fn() -> Any:
        probe.start()
        return suite.run(state)

    runner.run(run_fn, config.benchmark, on_rep=on_rep)
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
    run_id = writer.write(rows, OmegaConf.to_container(cfg, resolve=True), hardware_info)
    print(f"wrote {len(rows)} rows for run_id={run_id}")


if __name__ == "__main__":
    main()
