"""BenchmarkRunner: warmup -> timed repetitions -> per-rep hook.

Shared between the benchmark suite (collects runtimes) and the memory suite
(collects memory-probe snapshots instead) - see suites/benchmark.py and
suites/memory.py. Calls infra.synchronize.synchronize(framework, device)
before/after every timed call; backends are never responsible for
synchronizing themselves. Serializes execution across concurrent Hydra
multirun jobs via infra.lock.global_run_lock, so GPU timing on one job is
never contaminated by contention from a sibling job.
"""

from __future__ import annotations

from typing import Any, Callable

from benchmark.config import MeasurementConfig
from benchmark.infra.lock import global_run_lock
from benchmark.infra.synchronize import synchronize
from benchmark.infra.timer import BenchmarkTimer


class BenchmarkRunner:
    def __init__(self, framework: str, device: str) -> None:
        self._timer = BenchmarkTimer(lambda: synchronize(framework, device))

    def run(
        self,
        run_fn: Callable[[], Any],
        measurement: MeasurementConfig,
        on_rep: Callable[[Any, float], None] | None = None,
    ) -> Any:
        """Run warmup then timed repetitions.

        ``on_rep(result, elapsed_ms)`` is called once per measured repetition
        (never during warmup). Returns the last repetition's result.
        """
        with global_run_lock():
            # One calibration iteration doubles as warmup rep #1 and gives us
            # a per-iteration duration estimate to size the rest of warmup.
            _, first_ms = self._timer.time_call(run_fn)
            extra_iters = 0
            if measurement.warmup_min_seconds > 0 and first_ms > 0:
                extra_iters = max(
                    0, round(measurement.warmup_min_seconds * 1000.0 / first_ms) - 1
                )
            remaining_warmup = max(measurement.warmup_min_iters - 1, extra_iters)
            for _ in range(remaining_warmup):
                self._timer.time_call(run_fn)

            result = None
            total_elapsed_s = 0.0
            n_reps = 0
            # A budget of 0 means "no time budget" (memory suite): run exactly
            # max_repetitions reps regardless of elapsed time.
            while (
                measurement.min_runtime_budget_s <= 0
                or total_elapsed_s < measurement.min_runtime_budget_s
            ) and n_reps < measurement.max_repetitions:
                result, elapsed_ms = self._timer.time_call(run_fn)
                if on_rep is not None:
                    on_rep(result, elapsed_ms)
                total_elapsed_s += elapsed_ms / 1000.0
                n_reps += 1

            return result
