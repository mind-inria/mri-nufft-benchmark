"""ResourceMonitor: background child-process sampler for peak CPU/GPU memory.

Framework allocator counters aren't comparable across backends: torch's
``memory_stats()`` tracks a true peak, cupy's memory pool only exposes
*current* usage, and neither says anything about a numpy/finufft backend.
Instead this polls OS/driver-level memory - psutil RSS for the process,
pynvml device memory for the GPU - from a separate process at a tight
interval, so the resulting peak is measured the same way regardless of which
array library a backend uses.

Runs as a ``multiprocessing.Process`` rather than a thread so sampling isn't
starved by holding the GIL against a backend that doesn't release it (e.g.
numpy/finufft during an FFT). Adapted from the resource-monitor approach in
https://github.com/paquiteau/hydra-callbacks.
"""

from __future__ import annotations

import multiprocessing
import os
from typing import Any


def _sample_loop(
    pid: int,
    interval: float,
    gpu_device_index: int | None,
    rss_peak: Any,
    gpu_used_peak: Any,
    stop_event: Any,
    ready_event: Any,
) -> None:
    import psutil

    process = psutil.Process(pid)
    handle = None
    if gpu_device_index is not None:
        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_device_index)

    def sample() -> bool:
        try:
            rss = process.memory_info().rss
        except psutil.NoSuchProcess:
            return False
        if rss > rss_peak.value:
            rss_peak.value = rss
        if handle is not None:
            used = pynvml.nvmlDeviceGetMemoryInfo(handle).used
            if used > gpu_used_peak.value:
                gpu_used_peak.value = used
        return True

    try:
        # Sample once before signalling ready, so start() only returns to
        # the caller once a real measurement is in hand - otherwise a
        # bracket shorter than the child's own startup latency (fork +
        # nvmlInit) could see zero samples.
        sample()
        ready_event.set()
        while not stop_event.wait(interval):
            if not sample():
                break
        # One last sample right as we're told to stop, to catch a peak that
        # occurred between the last periodic sample and the stop signal.
        sample()
    finally:
        if handle is not None:
            pynvml.nvmlShutdown()


class ResourceMonitor:
    """Poll this process's peak RSS and (optionally) peak GPU device memory.

    One instance is meant to be reused across repeated start()/stop()
    brackets (e.g. one per benchmark repetition) - each start() spawns a
    fresh sampler process and each stop() tears it down, so a bracket
    shorter than ``interval`` may see no samples beyond the start/stop
    baseline.
    """

    def __init__(self, gpu_device_index: int | None = None, interval: float = 0.001) -> None:
        self._gpu_device_index = gpu_device_index
        self._interval = interval
        self._rss_peak = multiprocessing.Value("L", 0)
        self._gpu_used_peak = multiprocessing.Value("L", 0)
        self._gpu_used_baseline = 0
        self._stop_event = multiprocessing.Event()
        self._ready_event = multiprocessing.Event()
        self._process: multiprocessing.Process | None = None

    def start(self) -> None:
        import psutil

        self._rss_peak.value = psutil.Process().memory_info().rss
        self._gpu_used_baseline = self._current_gpu_used()
        self._gpu_used_peak.value = self._gpu_used_baseline
        self._stop_event.clear()
        self._ready_event.clear()
        self._process = multiprocessing.Process(
            target=_sample_loop,
            args=(
                os.getpid(),
                self._interval,
                self._gpu_device_index,
                self._rss_peak,
                self._gpu_used_peak,
                self._stop_event,
                self._ready_event,
            ),
            daemon=True,
        )
        self._process.start()
        self._ready_event.wait(timeout=5.0)

    def stop(self) -> dict[str, float | None]:
        self._stop_event.set()
        if self._process is not None:
            self._process.join(timeout=2.0)
            self._process = None

        gpu_driver_mb = None
        gpu_allocated_mb = None
        if self._gpu_device_index is not None:
            gpu_driver_mb = self._gpu_used_peak.value / 1e6
            gpu_allocated_mb = max(0, self._gpu_used_peak.value - self._gpu_used_baseline) / 1e6

        return {
            "cpu_rss_mb": self._rss_peak.value / 1e6,
            "gpu_driver_mb": gpu_driver_mb,
            "gpu_allocated_mb": gpu_allocated_mb,
        }

    def _current_gpu_used(self) -> int:
        if self._gpu_device_index is None:
            return 0
        import pynvml

        pynvml.nvmlInit()
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(self._gpu_device_index)
            return pynvml.nvmlDeviceGetMemoryInfo(handle).used
        finally:
            pynvml.nvmlShutdown()
