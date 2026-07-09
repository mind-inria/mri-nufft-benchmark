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


def release_gpu_memory(framework: str, device: str) -> None:
    """Free the framework's GPU allocator caches back to the driver.

    Hydra's ``-m`` multirun runs every job in the sweep inside one process
    (BasicLauncher), so torch's caching allocator and cupy's memory pool
    otherwise keep every prior job's freed blocks reserved instead of
    releasing them - one sweep across many backends/scenarios then OOMs
    partway through even though no single job needs that much memory. Must
    be called after each job's state/operator has gone out of scope, so
    there are no live references left for the allocator to preserve.
    """
    if device != "cuda":
        return
    import gc

    gc.collect()
    if framework == "torch":
        import torch

        torch.cuda.empty_cache()
    elif framework == "cupy":
        import cupy

        cupy.get_default_memory_pool().free_all_blocks()
        cupy.get_default_pinned_memory_pool().free_all_blocks()
    elif framework == "numpy":
        return
    else:
        raise ValueError(f"unknown framework: {framework!r}")
