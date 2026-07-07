"""Suite protocol: setup -> run -> measure -> validate.

`run()` executes exactly the action under test, no validation logic inside
the timed call. `measure()`/`validate()` operate on the already-captured
output and never re-run the action.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from benchmark.config import BenchmarkConfig


@dataclass
class ValidationResult:
    passed: bool | None
    relative_l2_error: float | None
    relative_linf_error: float | None
    adjointness_error: float | None


class BenchmarkSuite(Protocol):
    def setup(self, config: BenchmarkConfig, repo_root: Path) -> dict[str, Any]: ...

    def run(self, state: dict[str, Any]) -> Any: ...

    def measure(self, result: Any, state: dict[str, Any]) -> dict[str, Any]: ...

    def validate(
        self, result: Any, state: dict[str, Any], config: BenchmarkConfig
    ) -> ValidationResult: ...
