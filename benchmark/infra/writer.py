"""ResultWriter: writes one raw parquet file + one metadata yaml per run.

Filenames use a uuid-based run_id rather than a sequential counter, so
concurrent Hydra multirun jobs never need cross-process coordination just to
pick a non-colliding filename. Raw files are append-only: this writer only
ever creates a new file, never opens an existing one for modification.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

import polars as pl
import yaml


class ResultWriter:
    def __init__(self, results_dir: Path) -> None:
        self.results_dir = Path(results_dir)
        (self.results_dir / "raw").mkdir(parents=True, exist_ok=True)
        (self.results_dir / "metadata").mkdir(parents=True, exist_ok=True)

    def write(
        self,
        rows: list[dict[str, Any]],
        resolved_config: dict[str, Any],
        hardware_info: dict[str, Any],
    ) -> str:
        run_id = uuid.uuid4().hex[:12]
        for row in rows:
            row["run_id"] = run_id
            row["hardware_id"] = hardware_info["hardware_id"]

        pl.DataFrame(rows).write_parquet(
            self.results_dir / "raw" / f"run_{run_id}.parquet"
        )

        metadata = {**hardware_info, "run_id": run_id, "resolved_config": resolved_config}
        with open(self.results_dir / "metadata" / f"run_{run_id}.yaml", "w") as f:
            yaml.safe_dump(metadata, f, sort_keys=False)

        return run_id
