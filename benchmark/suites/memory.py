"""MemorySuite: peak CPU/GPU memory, measured in a separate execution from timing.

Subclasses BenchmarkSuiteImpl to reuse its setup/run (identical operator
construction and action dispatch) - only measure/validate differ, since
suite="memory" rows carry no accuracy metrics (see README's Raw measurements
schema). The actual memory probing happens around the timed loop in run.py,
via infra.memory.MemoryProbe - not here.
"""

from __future__ import annotations

from typing import Any

from benchmark.config import BenchmarkConfig
from benchmark.suites.base import ValidationResult
from benchmark.suites.benchmark import BenchmarkSuiteImpl


class MemorySuiteImpl(BenchmarkSuiteImpl):
    def measure(self, result: Any, state: dict[str, Any]) -> dict[str, Any]:
        return {
            "relative_l2_error": None,
            "relative_linf_error": None,
            "adjointness_error": None,
        }

    def validate(
        self, result: Any, state: dict[str, Any], config: BenchmarkConfig
    ) -> ValidationResult:
        return ValidationResult(None, None, None, None)
