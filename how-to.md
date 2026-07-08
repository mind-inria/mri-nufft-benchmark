# How to reproduce the benchmark

## 1. Environment

```bash
uv venv
uv sync
```

This installs `mri-nufft`, `finufft`, `cufinufft` `torchkbnufft`, Hydra, Polars/PyArrow, Plotly,
psutil/pynvml, and the dev tools (`pytest`, `ruff`).

`cufinufft`'s GPU kernels require a real CUDA toolkit visible to CMake at

## 2. Generate trajectory and smaps assets

One-time step. Trajectory generation must never happen at benchmark runtime,
so canonical `.npy` files are pre-generated and versioned:

```bash
uv run python scripts/generate_assets.py
```

This writes:
- `assets/trajectories/*.npy` (+ `manifest.yaml`) — the 5 default scenarios'
  trajectories plus the diagnostic `radial_2d_512_standard`.
- `assets/smaps/smaps_2d_256_v1.npy` (+ `manifest.yaml`) — synthetic birdcage
  coil sensitivity maps for the 8-coil scenario.

Re-run this if you ever change trajectory/smaps generation parameters in the
script; it's idempotent (overwrites the same files).

## 3. Run a single benchmark

```bash
uv run python run.py \
  backend=finufft \
  trajectory=spiral_2d_256_standard \
  mri_setup=2d_m_8coils \
  action=forward
```

- `backend`: `finufft` | `cufinufft` | `gpunufft` | `ducc0` | `pynfft` | `torchkbnufft-cpu` | `torchkbnufft-gpu`
- `trajectory`: any file under `configs/trajectory/` (must match `mri_setup`'s
  `ndim`)
- `mri_setup`: `2d_s_1coil` | `2d_m_1coil` | `2d_m_8coils` | `2d_m_32coils` |
  `3d_s_1coil` | `3d_m_1coil` | `3d_m_32coils`
- `action`: `operator_init` | `forward` | `adjoint` | `normal_operator`
- `input_location`: `host` (default) | `device` — `device` pre-stages the
  `forward`/`adjoint` test vector as a GPU-resident array (cupy, or a CUDA
  torch tensor) before timing; only valid for `forward`/`adjoint` on a
  `device=cuda` backend, e.g.:

  ```bash
  uv run python run.py backend=cufinufft trajectory=spiral_2d_256_standard \
    mri_setup=2d_m_8coils action=forward input_location=device
  ```

Results are written to `benchmark_results/raw/run_<id>.parquet` (one row per
replicate) and `benchmark_results/metadata/run_<id>.yaml` (hardware/software
info + the fully resolved config). This directory is gitignored — it's
runtime output, not source.

To shorten a run during development, override the measurement protocol
directly:

```bash
uv run python run.py backend=finufft trajectory=spiral_2d_128_standard \
  mri_setup=2d_s_1coil action=forward \
  benchmark.min_runtime_budget_s=1 benchmark.max_repetitions=5 \
  benchmark.warmup_min_iters=1 benchmark.warmup_min_seconds=0
```

## 4. Run the memory suite

Always a separate execution from timing (profiling overhead must not
contaminate runtime numbers):

```bash
uv run python run.py suite=memory benchmark=memory \
  backend=finufft trajectory=spiral_2d_256_standard \
  mri_setup=2d_m_8coils action=forward
```

## 5. Run the full scenario matrix

One Hydra `-m` (multirun) command per scenario row, sweeping backend × action.
Jobs are serialized internally (a global lock around the timed loop) even
though Hydra may launch them concurrently, so GPU timing is never
contaminated by a sibling job:

```bash
# 2D-S, 1 coil
uv run python run.py -m backend=finufft,cufinufft,torchkbnufft-cpu,torchkbnufft-gpu \
  trajectory=spiral_2d_128_standard mri_setup=2d_s_1coil \
  action=operator_init,forward,adjoint,normal_operator

# 2D-M, 1 coil
uv run python run.py -m backend=finufft,cufinufft,torchkbnufft-cpu,torchkbnufft-gpu \
  trajectory=spiral_2d_256_standard mri_setup=2d_m_1coil \
  action=operator_init,forward,adjoint,normal_operator

# 2D-M, 8 coils
uv run python run.py -m backend=finufft,cufinufft,torchkbnufft-cpu,torchkbnufft-gpu \
  trajectory=spiral_2d_256_standard mri_setup=2d_m_8coils \
  action=operator_init,forward,adjoint,normal_operator

# 2D-M, 32 coils
uv run python run.py -m backend=finufft,cufinufft,torchkbnufft-cpu,torchkbnufft-gpu \
  trajectory=spiral_2d_256_standard mri_setup=2d_m_32coils \
  action=operator_init,forward,adjoint,normal_operator

# 3D-S, 1 coil
uv run python run.py -m backend=finufft,cufinufft,torchkbnufft-cpu,torchkbnufft-gpu \
  trajectory=sos_3d_64_standard mri_setup=3d_s_1coil \
  action=operator_init,forward,adjoint,normal_operator

# 3D-M, 1 coil
uv run python run.py -m backend=finufft,cufinufft,torchkbnufft-cpu,torchkbnufft-gpu \
  trajectory=sos_3d_192_standard mri_setup=3d_m_1coil \
  action=operator_init,forward,adjoint,normal_operator

# 3D-M, 32 coils
uv run python run.py -m backend=finufft,cufinufft,torchkbnufft-cpu,torchkbnufft-gpu \
  trajectory=sos_3d_192_standard mri_setup=3d_m_32coils \
  action=operator_init,forward,adjoint,normal_operator
```

Repeat all 7 commands with `suite=memory benchmark=memory` for the memory
suite.

## 6. Build the report

Regenerates `benchmark_results/processed/{summary,pareto,scaling}.parquet`
from every raw file, then a single self-contained `report.html`:

```bash
uv run python scripts/build_report.py
open benchmark_results/report.html   # or just open it in a browser
```

The report has 7 tabs grouped into 3 sections. Per-Scenario Breakdown holds
Runtime (faceted by action x base trajectory family — 2D-S/2D-M/3D-S/3D-M —
bars grouped by ncoils, metric is runtime **per coil**), GPU Memory, and CPU
Memory (raw, one row per action, every backend x scenario combination),
restricted to `input_location=host`; runtime bars carry a [p5, p95] error
bar. GPU-Resident Input (GPU Resident Runtime, GPU Resident Memory) compares
`input_location=host` vs `input_location=device` for `forward`/`adjoint` on
cuda-capable backends (one row per action, one column per scenario). Scaling
(Scaling Runtime, Scaling Memory) plots runtime/memory vs image size or
coil count as line charts, one line per backend. Bar panels order backends
CPU-first then GPU, with a shaded band marking the boundary. It's fully
self-contained (Plotly is embedded inline) — no network access needed to
view it.

## 7. Publishing results

`benchmark_results/` is gitignored — don't commit it directly. To publish a
result set, copy the relevant `raw/` + `metadata/` files into a tagged GitHub
Release or a dedicated `results/` branch, as a deliberate separate act.

## Checks

```bash
uv run ruff check .
uv run pytest
```
