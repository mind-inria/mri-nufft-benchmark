"""Structured config dataclasses, registered with Hydra's config store.

Registering these as the schema means unknown fields or wrong types are
rejected before any benchmark code runs. Cross-field constraints that a
single YAML file cannot express live in ``__post_init__``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING


@dataclass
class BackendConfig:
    name: str = MISSING
    device: str = MISSING
    framework: str = MISSING
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass
class TrajectoryConfig:
    trajectory_id: str = MISSING
    trajectory_type: str = MISSING
    ndim: int = MISSING
    n_samples: int = MISSING
    n_shots: int = MISSING
    density: str = MISSING
    asset_file: str = MISSING
    asset_version: str = MISSING


@dataclass
class MriSetupConfig:
    ndim: int = MISSING
    image_size: list[int] = MISSING
    ncoils: int = MISSING
    smaps: bool = MISSING
    smaps_asset: str | None = None

    def __post_init__(self) -> None:
        if self.smaps and not self.smaps_asset:
            raise ValueError("smaps_asset is required when smaps=True")
        if len(self.image_size) != self.ndim:
            raise ValueError(f"image_size length must match ndim={self.ndim}")


@dataclass
class ActionConfig:
    name: str = MISSING


@dataclass
class MeasurementConfig:
    warmup_min_iters: int = MISSING
    warmup_min_seconds: float = MISSING
    min_runtime_budget_s: float = MISSING
    max_repetitions: int = MISSING


@dataclass
class ReferenceConfig:
    """Backend used to compute accuracy reference outputs.

    Only consulted for the benchmark suite; ignored for the memory suite.
    """

    backend: str = MISSING
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkConfig:
    backend: BackendConfig = MISSING
    trajectory: TrajectoryConfig = MISSING
    mri_setup: MriSetupConfig = MISSING
    action: ActionConfig = MISSING
    benchmark: MeasurementConfig = MISSING
    reference: ReferenceConfig = MISSING
    suite: str = "benchmark"  # "benchmark" or "memory"

    def __post_init__(self) -> None:
        if self.trajectory.ndim != self.mri_setup.ndim:
            raise ValueError(
                f"trajectory.ndim={self.trajectory.ndim} does not match "
                f"mri_setup.ndim={self.mri_setup.ndim}"
            )
        if self.suite not in ("benchmark", "memory"):
            raise ValueError(f"suite must be 'benchmark' or 'memory', got {self.suite!r}")


def register_configs() -> None:
    """Register the top-level schema so group compositions are validated against it."""
    cs = ConfigStore.instance()
    cs.store(name="base_config", node=BenchmarkConfig)
