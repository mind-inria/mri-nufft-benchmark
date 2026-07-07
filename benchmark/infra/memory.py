"""MemoryProbe: peak CPU/GPU memory measurement for the memory suite.

Only used by the memory suite — memory profiling tools add runtime overhead
and must never contaminate the benchmark suite's timing numbers.
"""

from __future__ import annotations

from typing import Any


class MemoryProbe:
    def __init__(self, framework: str, device: str) -> None:
        self.framework = framework
        self.device = device

    def start(self) -> None:
        import gc

        gc.collect()
        if self.framework == "torch" and self.device == "cuda":
            import torch

            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        elif self.framework == "cupy" and self.device == "cuda":
            import cupy

            cupy.cuda.Stream.null.synchronize()
            cupy.get_default_memory_pool().free_all_blocks()

    def stop(self) -> dict[str, Any]:
        import psutil

        result: dict[str, Any] = {
            "cpu_rss_mb": psutil.Process().memory_info().rss / 1e6,
            "gpu_allocated_mb": None,
            "gpu_driver_mb": None,
        }

        if self.framework == "torch" and self.device == "cuda":
            import torch

            torch.cuda.synchronize()
            stats = torch.cuda.memory_stats()
            result["gpu_allocated_mb"] = stats["allocated_bytes.all.peak"] / 1e6
        elif self.framework == "cupy" and self.device == "cuda":
            import cupy

            cupy.cuda.Stream.null.synchronize()
            result["gpu_allocated_mb"] = (
                cupy.get_default_memory_pool().used_bytes() / 1e6
            )

        if self.device == "cuda":
            result["gpu_driver_mb"] = self._driver_peak_mb()

        return result

    @staticmethod
    def _driver_peak_mb() -> float | None:
        try:
            import pynvml

            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            pynvml.nvmlShutdown()
            return info.used / 1e6
        except Exception:
            return None
