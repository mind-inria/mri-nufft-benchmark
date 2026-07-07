"""Operator construction on top of mri-nufft.

Every backend is a plain ``mrinufft.get_operator(...)`` operator exposing the
standard ``op``/``adj_op``/``gram_op`` interface - there is no adapter layer.
The only backend-specific handling needed is: (1) resolving our backend name
to mri-nufft's registry key, since torchkbnufft is registered as two
device-specific keys rather than one, and numpy/NDFT doesn't accept a
``density`` kwarg; and (2) converting to/from torch tensors for
torchkbnufft, whose op/adj_op/gram_op are not numpy-autocasting like the
other backends.
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
    "torchkbnufft": {"torchkbnufft-cpu", "torchkbnufft-gpu"},
    "gpunufft": {"gpunufft"},
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


def _mrinufft_backend_name(backend_name: str, device: str) -> str:
    if backend_name == "torchkbnufft":
        return "torchkbnufft-gpu" if device == "cuda" else "torchkbnufft-cpu"
    return backend_name


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

    operator_cls = mrinufft.get_operator(
        _mrinufft_backend_name(config.name, config.device)
    )
    kwargs: dict[str, Any] = dict(n_coils=ncoils, smaps=smaps, **config.parameters)
    if config.name != "numpy":
        # numpy/NDFT hardcodes density=False internally and doesn't accept
        # a `density` kwarg (unlike the other mri-nufft backends).
        kwargs["density"] = False
    return operator_cls(trajectory, image_size, **kwargs)


def to_operator_array(array: NDArray, operator: Any) -> Any:
    """Convert a numpy array to whatever array type ``operator`` expects.

    Only torchkbnufft requires this: its op/adj_op/gram_op are not
    numpy-autocasting like the other mri-nufft backends.
    """
    if operator.backend.startswith("torchkbnufft"):
        import torch

        return torch.as_tensor(array, device=operator.device)
    return array


def to_numpy(array: Any) -> NDArray:
    """Convert an operator's output back to a plain numpy array."""
    if hasattr(array, "detach"):
        return array.detach().cpu().numpy()
    return np.asarray(array)
