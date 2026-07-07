"""CPU/GPU synchronization primitives, dispatched on the backend's declared
framework/device (from BackendConfig) rather than runtime introspection.
Backends themselves never call synchronization.
"""

from __future__ import annotations


def synchronize(framework: str, device: str) -> None:
    if device != "cuda":
        return
    if framework == "torch":
        import torch

        torch.cuda.synchronize()
    elif framework == "cupy":
        import cupy

        cupy.cuda.Stream.null.synchronize()
    elif framework == "numpy":
        return
    else:
        raise ValueError(f"unknown framework: {framework!r}")
