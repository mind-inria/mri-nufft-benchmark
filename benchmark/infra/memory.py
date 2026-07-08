"""MemoryProbe: peak CPU/GPU memory measurement for the memory suite.

Only used by the memory suite - memory profiling tools add runtime overhead
and must never contaminate the benchmark suite's timing numbers. Peak memory
itself is measured by infra.monitor.ResourceMonitor (OS/driver-level, not
framework allocator counters - see its docstring for why); this class only
handles the framework-specific sync-before/after so the sampled window
brackets a fully-settled state on both ends.
"""

from __future__ import annotations

from typing import Any

from benchmark.infra.monitor import ResourceMonitor


class MemoryProbe:
    def __init__(self, framework: str, device: str) -> None:
        self.framework = framework
        self.device = device
        self._monitor = ResourceMonitor(gpu_device_index=0 if device == "cuda" else None)

    def _synchronize(self) -> None:
        if self.framework == "torch" and self.device == "cuda":
            import torch

            torch.cuda.synchronize()
        elif self.framework == "cupy" and self.device == "cuda":
            import cupy

            cupy.cuda.Stream.null.synchronize()

    def start(self) -> None:
        import gc

        gc.collect()
        self._synchronize()
        self._monitor.start()

    def stop(self) -> dict[str, Any]:
        self._synchronize()
        return self._monitor.stop()
