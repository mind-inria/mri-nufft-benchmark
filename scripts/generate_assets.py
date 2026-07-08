"""One-time asset generation: canonical trajectory .npy files and synthetic
sensitivity maps, written once and versioned. Trajectory generation must
never occur at benchmark runtime (see README) - this script is the only
place that calls into mrinufft.trajectories.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import yaml

import mrinufft.trajectories as mt

REPO_ROOT = Path(__file__).resolve().parent.parent
TRAJ_DIR = REPO_ROOT / "assets" / "trajectories"
SMAPS_DIR = REPO_ROOT / "assets" / "smaps"

ASSET_VERSION = "v1.0.0"
GENERATION_SEED = (
    42  # birdcage coil placement is deterministic; recorded for schema parity
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_trajectory(
    trajectory_id: str, array: np.ndarray, params: dict[str, Any]
) -> dict:
    path = TRAJ_DIR / f"{trajectory_id}.npy"
    np.save(path, array.astype(np.float32))
    print(f"wrote {path} shape={array.shape}")
    return {
        "trajectory_id": trajectory_id,
        "asset_version": ASSET_VERSION,
        "generation_library": "mrinufft.trajectories",
        "generation_seed": GENERATION_SEED,
        "generation_params": params,
        "checksum_sha256": _sha256(path),
    }


def generate_trajectories() -> list[dict]:
    TRAJ_DIR.mkdir(parents=True, exist_ok=True)
    manifest: list[dict] = []

    params = {"Nc": 128, "Ns": 256, "in_out": True, "nb_revolutions": 8}
    manifest.append(
        _write_trajectory(
            "spiral_2d_128_standard", mt.initialize_2D_spiral(**params), params
        )
    )

    # Matches the trajectory metadata example in README.md exactly.
    params = {"Nc": 256, "Ns": 256, "in_out": True, "nb_revolutions": 10}
    manifest.append(
        _write_trajectory(
            "spiral_2d_256_standard", mt.initialize_2D_spiral(**params), params
        )
    )

    params = {"Nc": 512, "Ns": 512, "in_out": True, "nb_revolutions": 12}
    manifest.append(
        _write_trajectory(
            "spiral_2d_512_standard", mt.initialize_2D_spiral(**params), params
        )
    )

    # Kept deliberately small (n_samples=1024, not a realistic Nyquist density
    # for a 64x64x44 image). What matters here is a consistent, fast,
    # comparable trajectory across backends, not reconstruction-grade
    # sampling density.
    base_params = {"Nc": 4, "Ns": 64, "in_out": True, "nb_revolutions": 8}
    base = mt.initialize_2D_spiral(**base_params)
    params = {**base_params, "nb_stacks": 4}
    manifest.append(
        _write_trajectory("sos_3d_64_standard", mt.stack(base, nb_stacks=4), params)
    )

    base_params = {"Nc": 64, "Ns": 256, "in_out": True, "nb_revolutions": 10}
    base = mt.initialize_2D_spiral(**base_params)
    params = {**base_params, "nb_stacks": 128}
    manifest.append(
        _write_trajectory("sos_3d_192_standard", mt.stack(base, nb_stacks=128), params)
    )

    # nb_stacks=176 matches mri_setup 3d_l's nz, same convention as 3D-M above.
    base_params = {"Nc": 128, "Ns": 256, "in_out": True, "nb_revolutions": 12}
    base = mt.initialize_2D_spiral(**base_params)
    params = {**base_params, "nb_stacks": 176}
    manifest.append(
        _write_trajectory("sos_3d_256_standard", mt.stack(base, nb_stacks=176), params)
    )

    # Available diagnostic asset, not part of the default scenario list.
    params = {"Nc": 512, "Ns": 512, "in_out": False}
    manifest.append(
        _write_trajectory(
            "radial_2d_512_standard", mt.initialize_2D_radial(**params), params
        )
    )

    return manifest


def _birdcage_maps(
    shape: tuple[int, ...], r: float = 1.5, nzz: int = 8, dtype: type = np.complex64
) -> np.ndarray:
    """Simulate birdcage coil sensitivities.

    Ported from a prior benchmark attempt (old-mri-nufft-benchmark/utils.py);
    original reference: sigpy.mri.sim.birdcage_maps.

    Parameters
    ----------
    shape: (n_coils, ny, nx) or (n_coils, nz, ny, nx)
    """
    if len(shape) == 4:
        nc, nz, ny, nx = shape
    elif len(shape) == 3:
        nc, ny, nx = shape
        nz = 1
    else:
        raise ValueError("shape must be (n_coils, ny, nx) or (n_coils, nz, ny, nx)")
    c, z, y, x = np.mgrid[:nc, :nz, :ny, :nx]

    coilx = r * np.cos(c * (2 * np.pi / nzz), dtype=np.float32)
    coily = r * np.sin(c * (2 * np.pi / nzz), dtype=np.float32)
    coilz = np.floor(np.float32(c / nzz)) - 0.5 * (np.ceil(nc / nzz) - 1)
    coil_phs = np.float32(-(c + np.floor(c / nzz)) * (2 * np.pi / nzz))

    x_co = (x - nx / 2.0) / (nx / 2.0) - coilx
    y_co = (y - ny / 2.0) / (ny / 2.0) - coily
    z_co = (z - nz / 2.0) / (nz / 2.0) - coilz
    rr = (x_co**2 + y_co**2 + z_co**2) ** 0.5
    phi = np.arctan2(x_co, -y_co) + coil_phs
    out = (1 / rr) * np.exp(1j * phi)

    rss = np.sum(np.abs(out) ** 2, axis=0) ** 0.5
    out /= rss
    out = np.squeeze(out)
    return out.astype(dtype)


def generate_smaps() -> list[dict]:
    SMAPS_DIR.mkdir(parents=True, exist_ok=True)
    manifest: list[dict] = []

    for smaps_id, ncoils, image_size in [
        ("smaps_2d_256_v1", 8, (256, 256)),
        ("smaps_2d_256_32coils_v1", 32, (256, 256)),
        ("smaps_2d_128_8coils_v1", 8, (128, 128)),
        ("smaps_2d_128_32coils_v1", 32, (128, 128)),
        # Matches mri_setup 3d_m_32coils' image_size - the 3D coil-scaling
        # counterpart to the 2D 8-coil/32-coil pair above.
        ("smaps_3d_192_32coils_v1", 32, (192, 192, 128)),
        ("smaps_3d_192_8coils_v1", 8, (192, 192, 128)),
        ("smaps_3d_64_8coils_v1", 8, (64, 64, 44)),
        ("smaps_3d_64_32coils_v1", 32, (64, 64, 44)),
        ("smaps_2d_512_8coils_v1", 8, (512, 512)),
        ("smaps_2d_512_32coils_v1", 32, (512, 512)),
        ("smaps_3d_256_8coils_v1", 8, (256, 256, 176)),
        ("smaps_3d_256_32coils_v1", 32, (256, 256, 176)),
    ]:
        smaps = _birdcage_maps((ncoils, *image_size), nzz=ncoils)
        path = SMAPS_DIR / f"{smaps_id}.npy"
        np.save(path, smaps)
        print(f"wrote {path} shape={smaps.shape}")
        manifest.append(
            {
                "smaps_id": smaps_id,
                "asset_version": ASSET_VERSION,
                "generation_method": "birdcage",
                "ncoils": ncoils,
                "image_size": list(image_size),
                "checksum_sha256": _sha256(path),
            }
        )
    return manifest


def main() -> None:
    trajectory_manifest = generate_trajectories()
    with open(TRAJ_DIR / "manifest.yaml", "w") as f:
        yaml.safe_dump(trajectory_manifest, f, sort_keys=False)

    smaps_manifest = generate_smaps()
    with open(SMAPS_DIR / "manifest.yaml", "w") as f:
        yaml.safe_dump(smaps_manifest, f, sort_keys=False)


if __name__ == "__main__":
    main()
