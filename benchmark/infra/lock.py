"""Global cross-process execution lock.

Hydra `-m` multirun can launch several jobs in parallel, but GPU timing must
never be contaminated by a sibling job's contention (see design decision:
sequential execution only). Every BenchmarkRunner invocation acquires this
lock around its full setup->warmup->timed-reps span, serializing benchmark
runs regardless of how many Hydra jobs are launched concurrently.
"""

from __future__ import annotations

import contextlib
import fcntl
import tempfile
from pathlib import Path

_LOCK_PATH = Path(tempfile.gettempdir()) / "mri-nufft-benchmark.lock"


@contextlib.contextmanager
def global_run_lock():
    _LOCK_PATH.touch(exist_ok=True)
    with open(_LOCK_PATH, "w") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)
