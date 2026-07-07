"""Wall-clock measurement of a single call, synchronized before and after."""

from __future__ import annotations

from time import perf_counter
from typing import Callable, TypeVar

T = TypeVar("T")


class BenchmarkTimer:
    def __init__(self, synchronize: Callable[[], None]) -> None:
        self._synchronize = synchronize

    def time_call(self, fn: Callable[[], T]) -> tuple[T, float]:
        """Return (result, elapsed_ms)."""
        self._synchronize()
        t0 = perf_counter()
        result = fn()
        self._synchronize()
        t1 = perf_counter()
        return result, (t1 - t0) * 1000.0
