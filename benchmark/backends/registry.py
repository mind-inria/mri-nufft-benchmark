"""Operator construction on top of mri-nufft.

Every backend is a plain ``mrinufft.get_operator(...)`` operator exposing the
standard ``op``/``adj_op``/``gram_op`` interface - there is no adapter layer.
Our backend names match mri-nufft's registry keys directly (including
``torchkbnufft-cpu``/``torchkbnufft-gpu``, registered as two separate keys
upstream rather than one device-parameterized backend), except that
numpy/NDFT doesn't accept a ``density`` kwarg. The only other
backend-specific handling needed is: (1) converting to/from torch tensors
for torchkbnufft, whose op/adj_op/gram_op are not numpy-autocasting like the
other backends; and (2) pre-staging test vectors onto the GPU (cupy, or a
CUDA torch tensor) for ``input_location=device`` benchmarks.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from benchmark.config import BackendConfig

# mri-nufft's own backend registry key(s) each of our backend names depends on.
_MRINUFFT_BACKEND_KEYS: dict[str, set[str]] = {
    "finufft": {"finufft"},
    "cufinufft": {"cufinufft"},
    "numpy": {"numpy"},
    "torchkbnufft-cpu": {"torchkbnufft-cpu"},
    "torchkbnufft-gpu": {"torchkbnufft-gpu"},
    "gpunufft": {"gpunufft"},
    "ducc0": {"ducc0"},
    "pynfft": {"pynfft"},
}


def _available_mrinufft_backends() -> set[str]:
    from mrinufft.operators import list_backends

    return set(list_backends(available_only=True))


def list_available_backends() -> list[str]:
    """Our backend names that are actually usable in this environment."""
    available = _available_mrinufft_backends()
    return sorted(
        name for name, keys in _MRINUFFT_BACKEND_KEYS.items() if keys & available
    )


def build_operator(
    config: BackendConfig,
    *,
    trajectory: NDArray,
    image_size: tuple[int, ...],
    ncoils: int,
    smaps: NDArray | None,
) -> Any:
    if config.name not in list_available_backends():
        raise ValueError(
            f"backend {config.name!r} is not registered or not available in "
            f"this environment. Available: {list_available_backends()}"
        )

    import mrinufft

    operator_cls = mrinufft.get_operator(config.name)
    kwargs: dict[str, Any] = dict(n_coils=ncoils, smaps=smaps, **config.parameters)
    if config.name != "numpy":
        # numpy/NDFT hardcodes density=False internally and doesn't accept
        # a `density` kwarg (unlike the other mri-nufft backends).
        kwargs["density"] = False
    if config.name == "pynfft":
        # pyNFFT3's plan.x setter requires float64, C-contiguous - unlike
        # every other backend here, which accepts our default float32 samples.
        trajectory = np.ascontiguousarray(trajectory, dtype=np.float64)
    return operator_cls(trajectory, image_size, **kwargs)


def to_operator_array(array: NDArray, operator: Any) -> Any:
    """Convert a numpy array to whatever array type ``operator`` expects.

    Only torchkbnufft requires this: its op/adj_op/gram_op are not
    numpy-autocasting like the other mri-nufft backends.
    """
    if _is_torchkbnufft(operator):
        import torch

        return torch.as_tensor(array, device=operator.device)
    return array


def to_resident_array(array: NDArray, operator: Any) -> Any:
    """Move a host test vector onto the device ``operator`` runs on.

    Used once during setup for ``input_location=device`` benchmarks, so the
    host->device transfer happens outside the timed call - the point of that
    scenario is measuring the backend when fed already-GPU-resident data
    (e.g. mid-reconstruction-loop), not the transfer itself.
    """
    if _is_torchkbnufft(operator):
        import torch

        return torch.as_tensor(array, device=operator.device)
    import cupy as cp

    return cp.asarray(array)


def _is_torchkbnufft(operator: Any) -> bool:
    return operator.backend.startswith("torchkbnufft")


def to_numpy(array: Any) -> NDArray:
    """Convert an operator's output back to a plain numpy array."""
    if hasattr(array, "detach"):
        return array.detach().cpu().numpy()
    if type(array).__module__.startswith("cupy"):
        return array.get()
    return np.asarray(array)
