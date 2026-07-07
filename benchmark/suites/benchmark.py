"""BenchmarkSuite: runtime + accuracy.

The reference output (NDFT or high-precision FINUFFT, per README's reference
backend selection table) depends only on the scenario (trajectory, mri_setup,
action) - not on the backend under test - and is cached in-process so
comparing N backends against the same scenario doesn't recompute an
expensive reference N times.

Precision: test vectors for the backend under test are complex64; the
reference is computed internally in complex128 from the same seed, then cast
down to complex64 before diffing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from benchmark.backends.registry import build_operator, to_numpy, to_operator_array
from benchmark.config import BenchmarkConfig
from benchmark.suites.base import ValidationResult

ACCURACY_GATE_L2 = 1e-3

_REFERENCE_CACHE: dict[tuple, np.ndarray] = {}


class _ParamsOnly:
    """Duck-types as a BackendConfig for `build_operator`'s reference-backend calls."""

    def __init__(self, name: str, parameters: dict[str, Any]) -> None:
        self.name = name
        self.device = "cpu"
        self.parameters = parameters


def _image_shape(
    image_size: tuple[int, ...], ncoils: int, uses_sense: bool
) -> tuple[int, ...]:
    if ncoils == 1 or uses_sense:
        return image_size
    return (ncoils, *image_size)


def _kspace_shape(n_samples: int, ncoils: int) -> tuple[int, ...]:
    return (n_samples,) if ncoils == 1 else (ncoils, n_samples)


def _random_complex(shape: tuple[int, ...], seed: int, dtype: Any) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype(dtype)


def _load_trajectory(repo_root: Path, trajectory_config) -> np.ndarray:
    return np.load(repo_root / trajectory_config.asset_file)


def _load_smaps(repo_root: Path, mri_setup_config) -> np.ndarray | None:
    if not mri_setup_config.smaps:
        return None
    return np.load(repo_root / mri_setup_config.smaps_asset)


def uses_ndft_reference(image_size: tuple[int, ...]) -> bool:
    """True if the accuracy reference for this image size is the exact NDFT
    rather than high-eps FINUFFT (see README's reference backend selection
    table). Reused by run.py to fill the `reference_backend` raw column.
    """
    return int(np.prod(image_size)) <= 128 ** len(image_size)


def _apply_action(operator, action: str, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    if action == "forward":
        return to_numpy(operator.op(to_operator_array(x, operator)))
    if action == "adjoint":
        return to_numpy(operator.adj_op(to_operator_array(y, operator)))
    if action == "normal_operator":
        return to_numpy(
            operator.gram_op(to_operator_array(x, operator), toeplitz=False)
        )
    raise ValueError(f"no reference output defined for action={action!r}")


def _reference_output(
    config: BenchmarkConfig, trajectory: np.ndarray, smaps: np.ndarray | None
) -> np.ndarray:
    image_size = tuple(config.mri_setup.image_size)
    key = (
        config.trajectory.trajectory_id,
        image_size,
        config.mri_setup.ncoils,
        config.mri_setup.smaps,
        config.action.name,
    )
    if key in _REFERENCE_CACHE:
        return _REFERENCE_CACHE[key]

    backend_name = (
        "numpy" if uses_ndft_reference(image_size) else config.reference.backend
    )
    params = {} if backend_name == "numpy" else config.reference.parameters
    # FINUFFT's internal working precision follows the trajectory samples'
    # dtype, not the image data's - upcast to float64 so a high-eps FINUFFT
    # reference (e.g. eps=1e-12) is actually achievable rather than silently
    # clamped to float32 machine epsilon (~1.19e-7). smaps must match (an
    # in-place `coil_img *= smaps` keeps smaps' dtype, silently downcasting
    # the product back to complex64 otherwise).
    reference_smaps = smaps.astype(np.complex128) if smaps is not None else None
    operator = build_operator(
        _ParamsOnly(backend_name, params),
        trajectory=trajectory.astype(np.float64),
        image_size=image_size,
        ncoils=config.mri_setup.ncoils,
        smaps=reference_smaps,
    )

    uses_sense = smaps is not None
    x = _random_complex(
        _image_shape(image_size, config.mri_setup.ncoils, uses_sense),
        seed=0,
        dtype=np.complex128,
    )
    y = _random_complex(
        _kspace_shape(config.trajectory.n_samples, config.mri_setup.ncoils),
        seed=0,
        dtype=np.complex128,
    )
    output = _apply_action(operator, config.action.name, x, y)
    _REFERENCE_CACHE[key] = output
    return output


def _adjointness_error(operator, x: np.ndarray, y: np.ndarray) -> float:
    ax = to_numpy(operator.op(to_operator_array(x, operator)))
    aty = to_numpy(operator.adj_op(to_operator_array(y, operator)))
    lhs = np.vdot(ax, y)
    rhs = np.vdot(x, aty)
    return float(abs(lhs - rhs) / (np.linalg.norm(ax) * np.linalg.norm(y)))


class BenchmarkSuiteImpl:
    def setup(self, config: BenchmarkConfig, repo_root: Path) -> dict[str, Any]:
        trajectory = _load_trajectory(repo_root, config.trajectory)
        smaps = _load_smaps(repo_root, config.mri_setup)
        image_size = tuple(config.mri_setup.image_size)
        uses_sense = smaps is not None

        state: dict[str, Any] = {
            "action": config.action.name,
            "config": config,
            "trajectory": trajectory,
            "image_size": image_size,
            "ncoils": config.mri_setup.ncoils,
            "smaps": smaps,
            "x": _random_complex(
                _image_shape(image_size, config.mri_setup.ncoils, uses_sense),
                seed=0,
                dtype=np.complex64,
            ),
            "y": _random_complex(
                _kspace_shape(config.trajectory.n_samples, config.mri_setup.ncoils),
                seed=0,
                dtype=np.complex64,
            ),
        }

        if config.action.name != "operator_init":
            state["operator"] = build_operator(
                config.backend,
                trajectory=trajectory,
                image_size=image_size,
                ncoils=config.mri_setup.ncoils,
                smaps=smaps,
            )

        return state

    def run(self, state: dict[str, Any]) -> Any:
        action = state["action"]

        if action == "operator_init":
            return build_operator(
                state["config"].backend,
                trajectory=state["trajectory"],
                image_size=state["image_size"],
                ncoils=state["ncoils"],
                smaps=state["smaps"],
            )

        operator = state["operator"]
        if action == "forward":
            return to_numpy(operator.op(to_operator_array(state["x"], operator)))
        if action == "adjoint":
            return to_numpy(operator.adj_op(to_operator_array(state["y"], operator)))
        if action == "normal_operator":
            return to_numpy(
                operator.gram_op(
                    to_operator_array(state["x"], operator), toeplitz=False
                )
            )
        raise ValueError(f"unknown action {action!r}")

    def measure(self, result: Any, state: dict[str, Any]) -> dict[str, Any]:
        action = state["action"]
        if action == "operator_init":
            return {
                "relative_l2_error": None,
                "relative_linf_error": None,
                "adjointness_error": None,
            }

        config = state["config"]
        reference = _reference_output(
            config, state["trajectory"], state["smaps"]
        ).astype(np.complex64)
        output = np.asarray(result).astype(np.complex64)

        diff = output - reference
        relative_l2 = float(np.linalg.norm(diff) / np.linalg.norm(reference))
        relative_linf = float(np.max(np.abs(diff)) / np.max(np.abs(reference)))

        adjointness = None
        if action in ("forward", "adjoint"):
            adjointness = _adjointness_error(state["operator"], state["x"], state["y"])

        return {
            "relative_l2_error": relative_l2,
            "relative_linf_error": relative_linf,
            "adjointness_error": adjointness,
        }

    def validate(
        self, result: Any, state: dict[str, Any], config: BenchmarkConfig
    ) -> ValidationResult:
        if state["action"] == "operator_init":
            return ValidationResult(None, None, None, None)

        metrics = self.measure(result, state)
        l2 = metrics["relative_l2_error"]
        return ValidationResult(
            passed=l2 is not None and l2 < ACCURACY_GATE_L2,
            relative_l2_error=l2,
            relative_linf_error=metrics["relative_linf_error"],
            adjointness_error=metrics["adjointness_error"],
        )
