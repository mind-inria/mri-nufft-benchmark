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
class InputLocationConfig:
    """Where the forward/adjoint test vector lives before the timed call.

    ``host``: a plain numpy array (default) - any host->device transfer a GPU
    backend needs happens inside the timed call, same as a cold call site.
    ``device``: pre-converted to the backend's native GPU array (cupy, or a
    CUDA torch tensor for torchkbnufft) during setup, so the timed call
    measures a backend fed already-GPU-resident data (e.g. mid-reconstruction
    loop), not the transfer.
    """

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
    input_location: InputLocationConfig = field(
        default_factory=lambda: InputLocationConfig(name="host")
    )
    suite: str = "benchmark"  # "benchmark" or "memory"

    def __post_init__(self) -> None:
        if self.trajectory.ndim != self.mri_setup.ndim:
            raise ValueError(
                f"trajectory.ndim={self.trajectory.ndim} does not match "
                f"mri_setup.ndim={self.mri_setup.ndim}"
            )
        if self.suite not in ("benchmark", "memory"):
            raise ValueError(f"suite must be 'benchmark' or 'memory', got {self.suite!r}")
        if self.input_location.name not in ("host", "device"):
            raise ValueError(
                f"input_location.name must be 'host' or 'device', got "
                f"{self.input_location.name!r}"
            )
        if self.input_location.name == "device":
            if self.action.name not in ("forward", "adjoint"):
                raise ValueError(
                    "input_location=device only applies to forward/adjoint "
                    f"actions, got action={self.action.name!r}"
                )
            if self.backend.device != "cuda":
                raise ValueError(
                    "input_location=device requires a cuda backend, got "
                    f"backend.device={self.backend.device!r}"
                )


def register_configs() -> None:
    """Register the top-level schema so group compositions are validated against it."""
    cs = ConfigStore.instance()
    cs.store(name="base_config", node=BenchmarkConfig)
