"""Collect hardware/software metadata, stored alongside each run's raw results."""

from __future__ import annotations

import importlib.metadata
import platform
import socket
import subprocess
from datetime import date
from pathlib import Path
from typing import Any


def _git(*args: str, cwd: Path) -> str | None:
    try:
        return (
            subprocess.run(
                ["git", *args], cwd=cwd, capture_output=True, text=True, check=True
            )
            .stdout.strip()
            or None
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _gpu_info() -> dict[str, Any]:
    try:
        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        name = pynvml.nvmlDeviceGetName(handle)
        driver_version = pynvml.nvmlSystemGetDriverVersion()
        pynvml.nvmlShutdown()
        cuda_version = None
        try:
            import torch

            cuda_version = torch.version.cuda
        except ImportError:
            pass
        return {"gpu": name, "cuda_version": cuda_version, "driver_version": driver_version}
    except Exception:
        return {"gpu": None, "cuda_version": None, "driver_version": None}


def collect_hardware_info(repo_root: Path) -> dict[str, Any]:
    gpu = _gpu_info()
    hardware_id = f"{date.today().isoformat()}-{socket.gethostname()}"
    return {
        "hardware_id": hardware_id,
        "hardware": {
            "cpu": platform.processor() or platform.machine(),
            "ram_gb": _total_ram_gb(),
            **gpu,
        },
        "software": {
            "os": platform.platform(),
            "python_version": platform.python_version(),
            "mri_nufft_version": importlib.metadata.version("mri-nufft"),
            "git_sha": _git("rev-parse", "HEAD", cwd=repo_root),
            "git_tag": _git("describe", "--tags", "--exact-match", cwd=repo_root),
        },
        "environment": {"lock_file": "uv.lock"},
    }


def _total_ram_gb() -> float:
    import psutil

    return round(psutil.virtual_memory().total / 1e9, 1)
