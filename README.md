# MRI-NUFFT Benchmark

Repository: https://github.com/mind-inria/mri-nufft  
Benchmark repository: https://github.com/mind-inria/mri-nufft-benchmark

## Quick install

```bash
curl -LsSf https://raw.githubusercontent.com/mind-inria/mri-nufft-benchmark/main/scripts/setup.sh | sh
```

Self-contained: creates a `mri-nufft-benchmark-workspace/` folder
(override with `WORKSPACE_DIR=...`) and pre-generates the trajectory/smaps assets.
See [`how-to.md`](how-to.md) for the full step-by-step guide and manual setup.

## What's being compared

| Backend            | Device | Framework |
| ------------------- | ------ | --------- |
| `finufft`            | CPU    | numpy     |
| `ducc0`               | CPU    | numpy     |
| `pynfft`              | CPU    | numpy     |
| `torchkbnufft-cpu`    | CPU    | torch     |
| `cufinufft`           | GPU    | cupy      |
| `gpunufft`            | GPU    | cupy      |
| `torchkbnufft-gpu`    | GPU    | torch     |

`finufft` (CPU, tightest achievable `eps`) is the accuracy reference and the
denominator for relative speedup figures.

Each backend is exercised on four **actions**:

| Action             | What it measures                          |
| ------------------ | ------------------------------------------ |
| `operator_init`     | Backend construction / plan computation   |
| `forward`           | NUFFT forward pass: image → k-space       |
| `adjoint`           | NUFFT adjoint pass: k-space → image       |
| `normal_operator`   | Adjoint composed with forward (A*A)       |

For `forward`/`adjoint` on a GPU backend, `input_location` additionally
controls whether the input array starts on the host (default, includes
transfer cost) or is pre-staged on the GPU (`input_location=device`,
isolating compute from transfer — the mid-reconstruction-loop scenario).

Two independent suites cover the numbers above:

- **Benchmark suite** — runtime (median, p5/p95, MAD) and accuracy
  (relative L2/L∞ error against the reference, adjointness error),
  measured together since accuracy comes from the same captured output.
- **Memory suite** — peak CPU RSS and peak/persistent GPU memory, run as a
separate execution so profiling overhead never pollutes timing numbers.

### Scenario matrix

| mri_setup      | trajectory               |
| -------------- | ------------------------- |
| 2D-S, 1/8/32 coils   | `spiral_2d_128_standard` |
| 2D-M, 1/8/32 coils   | `spiral_2d_256_standard` |
| 2D-L, 1/8/32 coils   | `spiral_2d_512_standard` |
| 3D-S, 1/8/32 coils   | `sos_3d_64_standard`     |
| 3D-M, 1/8/32 coils   | `sos_3d_192_standard`    |
| 3D-L, 1/8/32 coils   | `sos_3d_256_standard`    |

| Label | Image size       |
| ----- | ----------------- |
| 2D-S  | 128 × 128         |
| 2D-M  | 256 × 256         |
| 2D-L  | 512 × 512         |
| 3D-S  | 64 × 64 × 44      |
| 3D-M  | 192 × 192 × 128   |
| 3D-L  | 256 × 256 × 176   |

At 32 coils, and across all of 3D-L, a reduced 4-backend tier
(`finufft`, `cufinufft`, `gpunufft`, `ducc0`) runs instead of all 7 — see
[Known Limitations](#known-limitations) for why `torchkbnufft-cpu`/
`torchkbnufft-gpu` and `pynfft` are dropped there. A GPU-resident-input
sweep (`forward`/`adjoint` only, `input_location=device`) runs on top of
every row for the 3 CUDA backends.

Trajectories are pre-generated, versioned `.npy` assets (never generated at
benchmark runtime), so every backend sees the exact same k-space
coordinates. A `radial_2d_512_standard` trajectory is also available as a
diagnostic asset but isn't part of the default scenario list.

---

## Running the benchmark

### Single run

```bash
uv run python run.py \
  backend=finufft \
  trajectory=spiral_2d_256_standard \
  mri_setup=2d_m_8coils \
  action=forward
```

- `backend`: `finufft` | `cufinufft` | `gpunufft` | `ducc0` | `pynfft` | `torchkbnufft-cpu` | `torchkbnufft-gpu`
- `trajectory`: any file under `configs/trajectory/` (must match `mri_setup`'s `ndim`)
- `mri_setup`: `{2d,3d}_{s,m,l}_{1,8,32}coils`, e.g. `2d_m_8coils`, `3d_l_32coils`
- `action`: `operator_init` | `forward` | `adjoint` | `normal_operator`
- `input_location`: `host` (default) | `device` (GPU backends only, `forward`/`adjoint` only)

Sweep multiple values with Hydra's `-m` (multirun) flag, e.g. every backend
× action for one scenario row:

```bash
uv run python run.py -m \
  backend=finufft,cufinufft,gpunufft,ducc0,pynfft,torchkbnufft-cpu,torchkbnufft-gpu \
  trajectory=spiral_2d_256_standard mri_setup=2d_m_8coils \
  action=operator_init,forward,adjoint,normal_operator
```

Results land in `benchmark_results/raw/` (one Parquet file per run) and
`benchmark_results/metadata/` (hardware/software info + the fully resolved
config) — gitignored, since it's runtime output.

To shorten a run during development, override the measurement protocol
directly, e.g. `benchmark.min_runtime_budget_s=1 benchmark.max_repetitions=5`.

### Memory suite

Always a separate execution:

```bash
uv run python run.py suite=memory benchmark=memory \
  backend=finufft trajectory=spiral_2d_256_standard \
  mri_setup=2d_m_8coils action=forward
```

### Full scenario matrix

```bash
./scripts/run_all.sh   # CPU + GPU backends, every scenario row, both suites
./scripts/run_gpu.sh   # GPU backends only — for a GPU-only machine/CI job
```

Both scripts generate assets, sweep every scenario row × backend × action
(benchmark suite, GPU-resident-input sweep, and memory suite), then build
`benchmark_results/report.html`. A single job crashing doesn't abort the
run — failures are collected and reported at the end, and the report is
built from whatever succeeded.

## Viewing results

```bash
uv run python scripts/build_report.py
open benchmark_results/report.html
```

Regenerates the processed datasets from every raw run, then a single self-contained `report.html` (Plotly embedded
inline — no network access needed to view it). The report has 7 tabs in 3
sections:

- **Per-Scenario Breakdown** — Runtime (per-coil, faceted by action ×
  trajectory family, `input_location=host` only), GPU Memory, CPU Memory.
- **GPU-Resident Input** — Runtime and memory for `input_location=host` vs
  `device`, `forward`/`adjoint` on the 3 CUDA backends.
- **Scaling** — runtime/memory vs. image size and vs. coil count, one line
  per backend, log-scaled axes.

---

## Interpreting results

### Accuracy metrics

Computed against `finufft` at its tightest achievable `eps`, using the same
fixed-seed test vectors as every other backend:

- **Relative L2 error**: `‖Ax − A_ref x‖₂ / ‖A_ref x‖₂`
- **Relative L∞ error** (peak error): `‖Ax − A_ref x‖∞ / ‖A_ref x‖∞`
- **Adjointness error**: `|⟨Ax, y⟩ − ⟨x, A*y⟩| / (‖Ax‖ · ‖y‖)`

A backend passes the **validation gate** if its relative L2 error is below
`1e-3` on the standard-density scenarios. Failing runs are stored with
`validation_passed=false`, excluded from the Pareto plot, but kept in the
raw dataset for diagnostics.

### Runtime statistics

Each measurement reports **median** (primary metric), **p5**/**p95**
(best-case and tail latency), and **MAD** (robust spread) over up to 200
repetitions within a 30-second budget — never just the mean, which is
skewed by occasional slow outliers (thermal throttling, memory pressure).

**Throughput** is `(n_samples × ncoils) / median_runtime_s`
(samples·coils/second).

### Memory numbers

Peak memory is deterministic (no meaningful distribution), so the memory
suite uses a short fixed protocol (1 warmup + 3 repetitions, max across
reps) instead of the timing budget. Reported per backend: peak CPU RSS
(`psutil`), framework-level peak GPU allocation (`torch.cuda`/`cupy`), and a
driver-level GPU peak (`pynvml`) recorded for every backend as a
cross-backend baseline regardless of framework.

---

## Publishing results

`benchmark_results/` is gitignored — it's runtime output, not source.
Publishing a result set is a separate, deliberate act: copy the relevant
`raw/` + `metadata/` files into a tagged GitHub Release or a dedicated
`results/` branch, never commit them directly to `main`.

---

## Known Limitations

- **`torchkbnufft-cpu`/`torchkbnufft-gpu` on 32-coil and 3D-L scenarios**:
  deliberately skipped there (see `scripts/run_all.sh`) — their per-coil,
  unbatched FFT plan construction makes `operator_init` dominate the whole
  matrix's wall time at that scale. Run manually via `run.py` if you need
  that data point.
- **`finufft`'s requested `eps=1e-6` is a single-precision floor, not
  always achievable**: single precision (`complex64`) can't get below
  roughly `N_max * eps_mach` for the largest grid dimension — a hard
  numerical limit. `finufft` is configured to clamp to the best achievable
  accuracy and proceed rather than error, so its `relative_l2_error` may sit
  above `1e-6` on the larger scenarios (2D-L, 3D-L, and to a lesser extent
  2D-M/3D-M) — check the accuracy columns rather than assuming the
  requested `eps` was met.
- **Not covered by this version**: Toeplitz acceleration, composite actions
  (`cg_iteration`, full reconstruction), automated CI regression
  gating, multi-GPU/distributed backends, non-uniform density compensation
  in accuracy comparisons, half-precision (FP16), batched reconstruction.
