"""Delete raw run files superseded by a later rerun of the same
(backend, action, scenario) combination.

Raw parquet files are append-only (see benchmark/infra/writer.py) -
rerunning a scenario adds a new run_*.parquet instead of replacing the old
one. analysis.processing.dedup already excludes those stale runs from every
report build, but they still sit on disk under benchmark_results/raw/ and
benchmark_results/metadata/ - this physically removes them.

Defaults to a dry run that only lists what would be deleted; pass --apply
to actually delete the files.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from analysis.loaders.raw import load_raw  # noqa: E402
from analysis.processing.dedup import latest_run_ids  # noqa: E402
from analysis.processing.summary import DEDUP_GROUP_COLS  # noqa: E402

RESULTS_DIR = REPO_ROOT / "benchmark_results"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="delete the stale run files (default: dry run, only lists them)",
    )
    args = parser.parse_args()

    raw = load_raw(RESULTS_DIR)
    if raw.is_empty():
        print("no raw runs found")
        return

    keep_ids = latest_run_ids(raw, DEDUP_GROUP_COLS)
    all_ids = set(raw["run_id"].unique().to_list())
    stale_ids = sorted(all_ids - keep_ids)

    if not stale_ids:
        print(f"no duplicated runs - {len(all_ids)} run(s), all unique")
        return

    print(f"{len(stale_ids)} stale run(s) out of {len(all_ids)} total (keeping {len(keep_ids)}):")
    for run_id in stale_ids:
        print(f"  run_{run_id}")

    if not args.apply:
        print("\ndry run - pass --apply to delete these files")
        return

    removed = 0
    for run_id in stale_ids:
        for path in (
            RESULTS_DIR / "raw" / f"run_{run_id}.parquet",
            RESULTS_DIR / "metadata" / f"run_{run_id}.yaml",
        ):
            if path.exists():
                path.unlink()
                removed += 1
    print(f"\ndeleted {removed} file(s) for {len(stale_ids)} stale run(s)")


if __name__ == "__main__":
    main()
