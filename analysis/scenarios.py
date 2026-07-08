"""Named trajectory scenarios shared by processing and reporting.

Kept in one place so adding/renaming a scenario in configs/trajectory only
needs a matching edit here, instead of desyncing scaling.py and report.py.
"""

from __future__ import annotations

# Trajectory families with more than one scenario in the default sweep
# (see scripts/run_all.sh) - each is a candidate axis for scaling plots.
SPIRAL_2D_FAMILY = {
    "spiral_2d_128_standard",
    "spiral_2d_256_standard",
    "spiral_2d_512_standard",
}
SOS_3D_FAMILY = {"sos_3d_64_standard", "sos_3d_192_standard", "sos_3d_256_standard"}

# Trajectories with a multi-coil (ncoils > 1) variant, so these are the only
# ones that can drive an ncoils-scaling axis (see scaling.py).
MULTICOIL_TRAJECTORY_IDS = {
    "spiral_2d_128_standard",
    "spiral_2d_256_standard",
    "spiral_2d_512_standard",
    "sos_3d_64_standard",
    "sos_3d_192_standard",
    "sos_3d_256_standard",
}

# (trajectory_id, ncoils) -> short label, matching the scenario list in
# README ("Benchmark Scenarios"). Ordering here is the display order for
# per-scenario figures.
SCENARIO_LABELS: dict[tuple[str, int], str] = {
    ("spiral_2d_128_standard", 1): "2D-S",
    ("spiral_2d_128_standard", 8): "2D-S (8 coils)",
    ("spiral_2d_128_standard", 32): "2D-S (32 coils)",
    ("spiral_2d_256_standard", 1): "2D-M",
    ("spiral_2d_256_standard", 8): "2D-M (8 coils)",
    ("spiral_2d_256_standard", 32): "2D-M (32 coils)",
    ("spiral_2d_512_standard", 1): "2D-L",
    ("spiral_2d_512_standard", 8): "2D-L (8 coils)",
    ("spiral_2d_512_standard", 32): "2D-L (32 coils)",
    ("sos_3d_64_standard", 1): "3D-S",
    ("sos_3d_64_standard", 8): "3D-S (8 coils)",
    ("sos_3d_64_standard", 32): "3D-S (32 coils)",
    ("sos_3d_192_standard", 1): "3D-M",
    ("sos_3d_192_standard", 8): "3D-M (8 coils)",
    ("sos_3d_192_standard", 32): "3D-M (32 coils)",
    ("sos_3d_256_standard", 1): "3D-L",
    ("sos_3d_256_standard", 8): "3D-L (8 coils)",
    ("sos_3d_256_standard", 32): "3D-L (32 coils)",
}
SCENARIO_ORDER = list(SCENARIO_LABELS.values())


def scenario_label(trajectory_id: str, ncoils: int) -> str:
    """Short display label for a (trajectory_id, ncoils) scenario.

    Falls back to a generated label for combinations not in the default
    scenario list (see README), instead of raising, so new/ad-hoc scenario
    rows still render.
    """
    return SCENARIO_LABELS.get((trajectory_id, ncoils), f"{trajectory_id} ({ncoils}c)")


def trajectory_label(trajectory_id: str) -> str:
    """Coil-count-independent label for a trajectory (e.g. "2D-M").

    Uses the ncoils=1 entry of SCENARIO_LABELS as the base label for that
    trajectory - every default-sweep trajectory has an ncoils=1 scenario -
    for figures that group backends by trajectory alone, independent of the
    ncoils variant (e.g. a per-coil-normalized runtime figure).
    """
    return SCENARIO_LABELS.get((trajectory_id, 1), trajectory_id)


# Backend -> device, mirroring each configs/backend/*.yaml's `device` field.
# Not in the raw/summary schema itself (run.py's _base_row doesn't record
# config.backend.device), so kept here rather than re-derived per report
# build - matches this module's existing job of mirroring config metadata
# the processed data doesn't carry.
BACKEND_DEVICES: dict[str, str] = {
    "finufft": "cpu",
    "ducc0": "cpu",
    "pynfft": "cpu",
    "torchkbnufft-cpu": "cpu",
    "cufinufft": "cuda",
    "gpunufft": "cuda",
    "torchkbnufft-gpu": "cuda",
}


def order_backends_by_device(backends: list[str]) -> list[str]:
    """CPU backends first (alphabetical), then GPU/cuda backends (alphabetical).

    Backends not in BACKEND_DEVICES (e.g. a new one not yet classified here)
    sort after both groups, alphabetically, instead of raising - so the
    figure still renders, just without a meaningful group placement for that
    one backend.
    """

    def key(b: str) -> tuple[int, str]:
        device = BACKEND_DEVICES.get(b)
        rank = {"cpu": 0, "cuda": 1}.get(device, 2)
        return (rank, b)

    return sorted(backends, key=key)
