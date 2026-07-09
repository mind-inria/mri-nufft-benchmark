# MRI-NUFFT Benchmark

Repository: https://github.com/mind-inria/mri-nufft  
Benchmark repository: https://github.com/mind-inria/mri-nufft-benchmark

---

## Overview

This benchmark provides a reproducible and fair comparison of MRI NUFFT
backends for non-Cartesian MRI reconstruction. All backends are evaluated
on the same trajectory files, the same image data, and the same hardware,
using a common measurement loop to ensure results are directly comparable.

The benchmark has two suites: a **benchmark suite** (runtime + accuracy,
measured together since accuracy is derived from the already-captured
output, not re-run) and a **memory suite** (run separately, since memory
profiling tools add runtime overhead). Both share a common scenario matrix,
trajectory asset library, and output schema.

---

## Objectives

### Performance

- Which NUFFT backend is the fastest for a given accuracy level?
- How does runtime scale with image size, trajectory size, and number of coils?
- What is the initialization overhead of each backend?

### Memory

- What is the peak CPU memory consumption?
- What is the peak GPU memory consumption?
- How does memory usage scale with multi-coil acquisitions?

### Accuracy

- What accuracy does each backend achieve for a given parameter configuration?
- What is the runtime-versus-accuracy trade-off?

---

## Benchmark Scenarios

The dimensions below (image size, coils, trajectory) are not run as a full
cross-product — most combinations are invalid (a 3D trajectory cannot pair
with a 2D `mri_setup`) and a full product would be needlessly expensive to
run. The **actual scenario list** is the following fixed, explicit set —
one `mri_setup` × `trajectory` pair per row, every backend × action
combination is run against each. Rows are chosen so that each scaling
dimension has at least two comparable points sharing everything except the
variable being scaled:

| mri_setup      | trajectory               | varies vs. row above                             |
| -------------- | ------------------------ | -------------------------------------------------- |
| 2D-S, 1 coil   | `spiral_2d_128_standard` | — (baseline)                                       |
| 2D-S, 8 coils  | `spiral_2d_128_standard` | ncoils (same size, same trajectory)                |
| 2D-S, 32 coils | `spiral_2d_128_standard` | ncoils (same size, same trajectory)                |
| 2D-M, 1 coil   | `spiral_2d_256_standard` | image size (same ncoils, same trajectory family)   |
| 2D-M, 8 coils  | `spiral_2d_256_standard` | ncoils (same size, same trajectory)                |
| 2D-M, 32 coils | `spiral_2d_256_standard` | ncoils (same size, same trajectory)                |
| 2D-L, 1 coil   | `spiral_2d_512_standard` | image size (same ncoils, same trajectory family)   |
| 2D-L, 8 coils  | `spiral_2d_512_standard` | ncoils (same size, same trajectory)                |
| 2D-L, 32 coils | `spiral_2d_512_standard` | ncoils (same size, same trajectory)                |
| 3D-S, 1 coil   | `sos_3d_64_standard`     | 3D coverage (image-size scaling vs. 3D-M below)    |
| 3D-S, 8 coils  | `sos_3d_64_standard`     | ncoils (same size, same trajectory)                |
| 3D-S, 32 coils | `sos_3d_64_standard`     | ncoils (same size, same trajectory)                |
| 3D-M, 1 coil   | `sos_3d_192_standard`    | 3D coverage (image-size scaling vs. 3D-S above)    |
| 3D-M, 8 coils  | `sos_3d_192_standard`    | ncoils (same size, same trajectory)                |
| 3D-M, 32 coils | `sos_3d_192_standard`    | ncoils (same size, same trajectory)                |
| 3D-L, 1 coil   | `sos_3d_256_standard`    | image size (same ncoils, same trajectory family)   |
| 3D-L, 8 coils  | `sos_3d_256_standard`    | ncoils (same size, same trajectory)                |
| 3D-L, 32 coils | `sos_3d_256_standard`    | ncoils (same size, same trajectory)                |

Radial trajectories remain defined as an available asset (used for
diagnostics or extending the matrix later) but are not part of the default
list, to avoid points that can't be compared across any single scaling axis.

The 32-coil rows, plus all three 3D-L rows, run a reduced 4-backend "fast"
tier (`finufft`, `cufinufft`, `gpunufft`, `ducc0`) instead of the full 7 (see
`scripts/run_all.sh`'s `FAST_BACKENDS`/`SLOW_BACKENDS`): `torchkbnufft-cpu`
builds one FFT plan per coil with no batching, so its `operator_init` cost
scales ~linearly with ncoils and dominates wall time at 32 coils;
`torchkbnufft-gpu` and `pynfft` are dropped from those rows too to keep the
matrix tractable given how expensive the 256³-scale 3D-L trajectory already
is at every coil count. `pynfft` also fails fast on every `ncoils>1`
scenario regardless of tier (see Known Limitations).

That's 10 scenarios (the "all"-tier 1-coil/8-coil rows) × 7 backends × 4
actions × 2 suites = 560 runs, plus 8 scenarios (32-coil rows and all
3D-L rows) × 4 backends × 4 actions × 2 suites = 256 runs, plus 18
scenarios × 3 cuda backends (`cufinufft`, `gpunufft`, `torchkbnufft-gpu`) ×
2 actions (`forward`/`adjoint` with `input_location=device`) × 2 suites =
216 more runs (the GPU-resident sweep always uses these 3 cuda backends,
independent of the CPU backend tier). At the default 30s budget for the
benchmark suite and a handful of seconds for the memory suite (see below),
the full matrix completes in a day or two on a single GPU (the 256³-scale
3D trajectory dominates). Extend this list deliberately (new row = new
evidence you need, and matching an existing trajectory family if it should
feed a scaling curve), not by re-introducing a full cross-product.

### Image dimensions

| Label | Size            |
| ----- | --------------- |
| 2D-S  | 128 × 128       |
| 2D-M  | 256 × 256       |
| 2D-L  | 512 × 512       |
| 3D-S  | 64 × 64 × 44    |
| 3D-M  | 192 × 192 × 128 |
| 3D-L  | 256 × 256 × 176 |

### Coil configurations

| ncoils | smaps |
| ------ | ----- |
| 1      | false |
| 8      | true  |
| 32     | true  |

`smaps` is a boolean field indicating whether complex-valued sensitivity maps
are applied per coil. When `smaps=true`, a synthetic set of coil sensitivity
maps (generated once, stored as a versioned asset) is applied. The sensitivity
map asset filename and version must be recorded in the run metadata.

### K-space trajectories

| type             | dimensionality | label convention          |
| ---------------- | -------------- | ------------------------- |
| Radial           | 2D             | `radial_2d_{N}_{density}` |
| Spiral           | 2D             | `spiral_2d_{N}_{density}` |
| Stack of spirals | 3D             | `sos_3d_{N}_{density}`    |

`density` is one of `low`, `standard`, or `high`, corresponding to
under-, Nyquist-, and over-sampled regimes respectively. Start with
`standard` only; add `low`/`high` once the standard case is validated.

For each trajectory asset, the following metadata must be stored explicitly
(not inferred from the filename):

```yaml
trajectory_id: spiral_2d_256_standard
trajectory_type: spiral
ndim: 2
n_samples: 65536
n_shots: 256
density: standard
generation_library: mrinufft.trajectories
generation_version: 0.5.0
generation_seed: 42
generation_params:
  nb_revolutions: 10
  in_out: true
```

Trajectory generation must never occur at benchmark runtime. Canonical
trajectory files are pre-generated once and stored as versioned `.npy` assets
in the benchmark repository. All backends operate on the exact same trajectory
coordinates.

### Actions

| action            | description                               | Priority |
| ----------------- | ----------------------------------------- | -------- |
| `operator_init`   | Backend construction and plan computation | High     |
| `forward`         | NUFFT forward pass: image → k-space       | High     |
| `adjoint`         | NUFFT adjoint pass: k-space → image       | High     |
| `normal_operator` | Adjoint composed with forward (A*A)       | Medium   |

Composite actions (`cg_iteration`, `reconstruction`) and Toeplitz-specific
actions are deferred — see Known Limitations.

### Input location

For `forward`/`adjoint` on a cuda-capable backend (`cufinufft`, `gpunufft`,
`torchkbnufft-gpu`), the `input_location` config controls
where the test vector under test lives before the timed call:

| `input_location` | description                                                                    |
| ----------------- | ------------------------------------------------------------------------------- |
| `host` (default)  | Plain numpy array — any host→device transfer happens inside the timed call, same as a cold call site. |
| `device`          | Pre-staged as a GPU-resident array (cupy, or a CUDA torch tensor) during setup, isolating backend compute from transfer cost — the scenario relevant mid-reconstruction-loop, where data is already on the GPU. |

`input_location=device` is only valid for `forward`/`adjoint` on a
`device=cuda` backend; any other combination fails config validation.

---

## Benchmark Suites

### 1. Benchmark Suite (runtime + accuracy)

`setup` builds the backend operator and generates the test vectors from the
fixed seed. `run` calls exactly the action under test — no validation logic
inside the timed call. `measure` computes runtime statistics (handled by the
runner) plus accuracy metrics from the already-captured output. `validate`
checks the accuracy gate once, from the same captured output — it does not
re-run anything.

The reference output (high-precision FINUFFT) depends only on the scenario
(trajectory, mri_setup, action) — not on the backend under test. It is
computed once per scenario and cached (in-process, keyed by scenario) so
that comparing N backends on the same scenario does not recompute an
expensive reference N times.

**`operator_init` is a special case.** For every other action, `setup()`
builds the operator once and `run()` reuses it. For `operator_init`, the
thing being timed is the construction itself, so `setup()` must not
pre-build the operator under test — it only loads config, trajectory, and
test vectors, and `run()` performs `build_operator(config)`.

#### Reference backend

FINUFFT with the smallest eps, for every image size.

#### Metrics

- **Relative L2 error**: `‖Ax − A_ref x‖₂ / ‖A_ref x‖₂`
- **Relative L∞ error** (peak error): `‖Ax − A_ref x‖∞ / ‖A_ref x‖∞`
- **Adjointness error**: `|⟨Ax, y⟩ − ⟨x, A*y⟩| / (‖Ax‖ · ‖y‖)`

Test vectors `x` and `y` are generated in `setup()` from a fixed random seed
(seed=0) recorded in the run config — deterministic from the seed alone, so
there is no need to store or version them as asset files.

#### Validation gate

A backend passes the accuracy validation gate if its relative L2 error is below
`1e-3` for the standard density scenario. Runs that fail the gate are stored
with `validation_passed=false` and are excluded from the Pareto plot but
retained in the raw dataset for diagnostic purposes.

#### Throughput definition

```
throughput = (n_samples × ncoils) / runtime_s        [samples·coils / second]
```

where `runtime_s` is the median runtime in seconds over all repetitions.

---

### 2. Memory Suite

Measured in a **separate execution** from timing — memory profiling tools
introduce runtime overhead and must not contaminate runtime numbers.

Unlike runtime, peak memory allocation is deterministic and stabilizes after
the first call — it has no meaningful distribution to average out, so the
30s/200-rep timing protocol does not apply here. The memory suite uses its
own minimal config (`configs/benchmark/memory.yaml`): 1 warmup iteration,
then a fixed **3 repetitions**, no time budget. Reported memory is the max
across those 3 reps (guards against a one-off allocator fluke, nothing more).

#### Measurement tools

| Metric                | Tool                                                                | Notes                                       |
| --------------------- | ------------------------------------------------------------------- | ------------------------------------------- |
| Peak CPU memory       | `psutil.Process.memory_info().rss`                                  | Total RSS                                   |
| Peak GPU memory       | `torch.cuda.memory_stats()["allocated_bytes.all.peak"]`             | For PyTorch-based backends                  |
| Peak GPU memory       | `cupy.get_default_memory_pool().used_bytes()` after `synchronize()` | For CuPy-based backends                     |
| Driver-level GPU peak | `pynvml.nvmlDeviceGetMemoryInfo()`                                  | Always recorded as a cross-backend baseline |

The `pynvml` driver-level measurement is recorded for every backend regardless
of framework, so results can be compared across backends that use different
memory management APIs. Framework-specific numbers are additionally recorded
when available.

#### Memory categories

- **Persistent GPU memory**: memory still allocated after operator
  initialization, before any NUFFT call (plans, lookup tables, precomputed
  kernels).
- **Peak GPU memory**: maximum allocation at any point during the action.

---

## Measurement Loop

All runtime measurements follow this protocol.

### Warmup

Before any timing begins, each backend must complete a warmup phase to
eliminate JIT compilation, kernel caching, and GPU pipeline startup costs from
the reported numbers. Warmup is defined as:

```
max(3 iterations, enough iterations to fill 1 second of wall-clock time)
```

Warmup is configurable via the `benchmark.warmup_min_iters` and
`benchmark.warmup_min_seconds` config fields. Warmup time is never included in
reported runtimes.

### Stopping condition

Repeated measurements continue until either condition is met:

- `min_runtime_budget_s` seconds of total measurement time have elapsed
  (default: **30 seconds**), or
- `max_repetitions` repetitions have been completed (default: **200**).

These defaults must be set explicitly in the config schema and stored in run
metadata so results from different machines can be interpreted correctly.

### Synchronization

Before and after each timed `run()` call, CPU and GPU must be synchronized to
prevent asynchronous GPU kernels from leaking into adjacent timing windows.

| Framework      | Synchronization call                   |
| -------------- | -------------------------------------- |
| PyTorch (CUDA) | `torch.cuda.synchronize()`             |
| CuPy           | `cupy.cuda.Stream.null.synchronize()`  |
| CPU-only       | No-op (synchronize is a no-op wrapper) |

The benchmark infrastructure must detect the backend's framework and call the
appropriate primitive. Backends must not be responsible for calling
synchronization themselves.

### Reporting statistics

Each benchmark reports:

- **Median runtime** (primary metric)
- **5th percentile** (best-case, indicative of peak hardware throughput)
- **95th percentile** (tail latency, important for interactive reconstruction)
- **MAD** (median absolute deviation, a robust spread measure)

Reporting only the mean is not acceptable — it is sensitive to occasional
slow outliers (thermal throttling, memory pressure) and misrepresents
typical performance.

### Benchmark loop (reference implementation)

```python
def run_benchmark(config, setup, run, measure, validate):
    state = setup(config)

    # Warmup phase — not timed
    warmup_iters = max(
        config.benchmark.warmup_min_iters,
        _iters_for_duration(state, config.benchmark.warmup_min_seconds),
    )
    for _ in range(warmup_iters):
        synchronize(state)
        run(state)
        synchronize(state)

    runtimes = []
    total_elapsed = 0.0
    n_reps = 0

    while (
        total_elapsed < config.benchmark.min_runtime_budget_s
        and n_reps < config.benchmark.max_repetitions
    ):
        synchronize(state)
        t0 = perf_counter()
        result = run(state)
        synchronize(state)
        t1 = perf_counter()

        elapsed = t1 - t0
        runtimes.append(elapsed * 1000)     # convert to ms
        total_elapsed += elapsed
        n_reps += 1

    metrics = measure(result, state)
    validation = validate(result, state, config)

    write_results(
        config=config,
        metrics=metrics,
        validation=validation,
        runtimes=runtimes,
    )
```

The memory suite follows the same loop shape but wraps `run()` in memory
probe start/stop calls instead of collecting `runtimes`.

### Benchmark structure (per-backend implementation)

```python
def setup(config):
    """
    Instantiate the NUFFT operator for the configured backend,
    image size, trajectory, and coil layout.
    Returns a state dict containing the operator, test vectors,
    and any pre-allocated buffers.
    """
    return state


def run(state):
    """
    Execute exactly the action under test (e.g. forward pass).
    Must not include any setup, allocation, or validation logic.
    Returns the raw output for downstream validation.
    """
    return result


def measure(result, state):
    """
    Extract scalar metrics from the result (e.g. accuracy errors).
    Must not re-run the operation.
    """
    return metrics


def validate(result, state, config):
    """
    Check whether the result passes the accuracy validation gate.
    Returns a ValidationResult with fields:
      passed: bool
      relative_l2_error: float
      relative_linf_error: float
      adjointness_error: float
    A failed validation does NOT abort storage — the result is
    written with validation.passed=false.
    """
    return validation
```

---

## Configuration

Configuration is managed with Hydra. Each config group maps to one dimension
of the benchmark. The complete config for a single run is the composition of
one file from each group. Hydra resolves all interpolations and writes the
final config to `.hydra/config.yaml` in the job working directory;
`infra/writer.py` reads this resolved file and embeds it verbatim in the run
metadata — this is what makes results reproducible even if a default value
changes in a later version.

### Config group files

**`configs/backend/finufft.yaml`** — example backend config:

```yaml
name: finufft
device: cpu
framework: numpy
parameters:
  eps: 1.0e-6
  upsampfac: 2.0
  nthreads: 4
```

`version` is deliberately not a config field: it's read from
`importlib.metadata.version("finufft")` at run time and written directly
into the run metadata YAML, so it always reflects the environment that
actually ran — not a value that could drift from what's installed.

**`configs/trajectory/spiral_2d_256_standard.yaml`**:

```yaml
trajectory_id: spiral_2d_256_standard
trajectory_type: spiral
ndim: 2
n_samples: 65536
n_shots: 256
density: standard
asset_file: assets/trajectories/spiral_2d_256_standard.npy
asset_version: v1.0.0
```

**`configs/mri_setup/2d_m_8coils.yaml`**:

```yaml
ndim: 2
image_size: [256, 256]
ncoils: 8
smaps: true
smaps_asset: assets/smaps/smaps_2d_256_v1.npy # required when smaps=true
```

**`configs/action/forward.yaml`**:

```yaml
name: forward
```

**`configs/input_location/device.yaml`** — see Input location:

```yaml
name: device
```

**`configs/benchmark/default.yaml`** — used by the benchmark suite (runtime + accuracy):

```yaml
warmup_min_iters: 3
warmup_min_seconds: 1.0
min_runtime_budget_s: 30.0
max_repetitions: 200
```

**`configs/benchmark/memory.yaml`** — used by the memory suite; no time
budget, since memory is deterministic and does not need statistical
averaging:

```yaml
warmup_min_iters: 1
warmup_min_seconds: 0.0
min_runtime_budget_s: 0.0
max_repetitions: 3
```

**`configs/run.yaml`** — top-level defaults list:

```yaml
defaults:
  - backend: finufft
  - trajectory: spiral_2d_256_standard
  - mri_setup: 2d_m_8coils
  - action: forward
  - benchmark: default
  - input_location: host
  - _self_

reference: # used by the accuracy metrics only
  backend: finufft
  parameters:
    eps: 1.0e-12
    upsampfac: 2.0
```

A single `mri_setup`/`trajectory` pair is one row of the scenario list
above; `-m` (Hydra `--multirun`) is used to sweep `backend` and `action`
within a row, not to cross trajectories against mri_setups (that would
regenerate the invalid/mismatched pairs the scenario list exists to avoid):

```bash
# Single run during development
python run.py backend=finufft trajectory=spiral_2d_256_standard \
              mri_setup=2d_m_8coils action=forward

# One scenario row, all backends x all actions, local parallel execution
python run.py -m backend=finufft,cufinufft,torchkbnufft-cpu,torchkbnufft-gpu \
              trajectory=spiral_2d_256_standard \
              mri_setup=2d_m_8coils \
              action=operator_init,forward,adjoint,normal_operator

# Full scenario list = one such command per row in the table above
```

### Structured config dataclasses

`benchmark/config.py` defines dataclasses that Hydra instantiates from the
composed YAML. They are registered with Hydra's config store and act as the
schema: unknown fields or wrong types are rejected before any benchmark code
runs. Cross-field constraints live in `__post_init__`:

```python
@dataclass
class MriSetupConfig:
    ndim: int
    image_size: list[int]
    ncoils: int
    smaps: bool
    smaps_asset: str | None = None

    def __post_init__(self):
        if self.smaps and not self.smaps_asset:
            raise ValueError("smaps_asset is required when smaps=True")
        if len(self.image_size) != self.ndim:
            raise ValueError(f"image_size length must match ndim={self.ndim}")
```

`BenchmarkConfig.__post_init__` additionally cross-validates _between_
groups — the check that matters most is that `mri_setup` and `trajectory`
agree on dimensionality, since nothing else stops composing a 3D trajectory
with a 2D `mri_setup`:

```python
@dataclass
class BenchmarkConfig:
    backend: BackendConfig
    trajectory: TrajectoryConfig
    mri_setup: MriSetupConfig
    action: ActionConfig
    benchmark: MeasurementConfig

    def __post_init__(self):
        if self.trajectory.ndim != self.mri_setup.ndim:
            raise ValueError(
                f"trajectory.ndim={self.trajectory.ndim} does not match "
                f"mri_setup.ndim={self.mri_setup.ndim}"
            )
```

Adding a new backend or trajectory means adding a YAML file to the relevant
config group and a `BackendAdapter` subclass in `benchmark/backends/`. No
changes to `run.py`, the dataclasses, or `configs/run.yaml` are needed.

---

## Output Schema

### Raw measurements

One row per replicate. Never overwritten after creation.

```text
run_id              string    — references hardware/software metadata
suite               string    — "benchmark" or "memory"; determines which
                                 columns below are meaningful (see notes)
backend             string
backend_version     string

ndim                int
nx                  int
ny                  int
nz                  int         — null for 2D
ncoils              int
smaps               bool

trajectory_id       string
trajectory_type     string
n_samples           int
n_shots             int
density             string

action              string

replicate           int         — 0-indexed repetition counter

runtime_ms          float       — wall-clock time of run(); for suite="memory"
                                  rows this includes profiler overhead and
                                  must NOT be used for performance analysis
                                  (filter summary/pareto/scaling on
                                  suite="benchmark")

cpu_rss_mb          float       — RSS from psutil, null for suite="benchmark"
gpu_allocated_mb    float       — framework peak, null for suite="benchmark"
gpu_driver_mb       float       — pynvml driver-level peak, null for suite="benchmark"

reference_backend   string      — null for suite="memory"
relative_l2_error   float       — null for suite="memory"
relative_linf_error float       — null for suite="memory"
adjointness_error   float       — null for suite="memory"
validation_passed   bool        — null for suite="memory"

hardware_id         string      — references hardware metadata
git_sha             string
timestamp           string      — ISO 8601
```

### Summary table

Derived from raw measurements. Regenerated on demand.

```text
suite               — "benchmark" or "memory"; summary is built separately
                       per suite, never mixed in one groupby
backend
backend_version

action

ndim
nx  ny  nz
ncoils
smaps
trajectory_id
n_samples

runtime_median_ms
runtime_mad_ms
runtime_p5_ms
runtime_p95_ms

throughput_median            — n_samples × ncoils / runtime_s

peak_cpu_rss_mb
peak_gpu_allocated_mb
peak_gpu_driver_mb
persistent_gpu_mb

relative_l2_error
relative_linf_error
adjointness_error
validation_passed

n_replicates
```

---

## Hardware and Software Metadata

Stored separately from measurement tables, referenced by `hardware_id`.

```yaml
hardware_id: 2026-01-05-a100-01

hardware:
  cpu: AMD EPYC 7742 64-Core
  ram_gb: 512
  gpu: NVIDIA A100 SXM4 80GB
  cuda_version: "12.2"
  driver_version: "535.104"

software:
  os: Ubuntu 22.04.3 LTS
  python_version: "3.11.5"
  mri_nufft_version: "0.5.0"
  git_sha: "a3f8c21"
  git_tag: "v0.5.0" # optional, null if not on a tag

environment:
  conda_env_file: environment.lock.yml # exact pinned versions
```

All benchmark results must include a `hardware_id` that references a metadata
entry, stored alongside the raw parquet files. `benchmark_results/` itself is
gitignored (it's runtime output, and parquet/npy are binary blobs unsuited to
a normal git history). "Published" results are a separate, deliberate act:
copy the relevant `raw/` + `metadata/` files into a tagged GitHub Release or
a dedicated `results/` branch — never commit them directly to `main`.

---

## Result Storage Structure

```text
benchmark_results/

├── raw/
│   ├── run_0001.parquet        — one file per benchmark execution
│   ├── run_0002.parquet
│   └── ...

├── metadata/
│   ├── run_0001.yaml           — hardware + software metadata
│   ├── run_0002.yaml
│   └── ...

├── assets/
│   ├── trajectories/
│   │   ├── spiral_2d_256_standard.npy
│   │   ├── radial_2d_512_standard.npy
│   │   └── manifest.yaml       — version, generation params, checksum
│   └── smaps/
│       ├── smaps_2d_256_v1.npy
│       └── manifest.yaml

├── processed/
│   ├── summary.parquet         — aggregated from all raw files
│   ├── pareto.parquet          — runtime vs accuracy, Pareto flags
│   └── scaling.parquet         — runtime/memory vs image size / n_samples

└── report.html                 — single self-contained report, see below
```

Raw measurement files are **append-only**. Processed files are regenerated from
raw data and may be overwritten. The report is regenerated from processed files.

---

## Code Structure

The repository is split into two top-level packages: `benchmark/` (the
measurement machinery) and `analysis/` (the post-processing and reporting
pipeline). They share nothing at runtime — `benchmark/` writes Parquet files,
`analysis/` reads them.

```text
mri-nufft-benchmark/
│
├── benchmark/
│   ├── __init__.py
│   ├── config.py                — structured config dataclasses
│   ├── runner.py                  — BenchmarkRunner: warmup → timed reps → write.
│   │                                 Calls infra.synchronize.synchronize(framework)
│   │                                 before/after each timed call — no per-device
│   │                                 subclasses needed for a single dispatch call.
│   ├── backends/
│   │   ├── base.py               — BackendAdapter interface
│   │   ├── finufft.py
│   │   ├── cufinufft.py
│   │   ├── torchkbnufft.py
│   │   └── registry.py           — REGISTRY: dict[str, type[BackendAdapter]]
│   ├── infra/
│   │   ├── timer.py               — BenchmarkTimer
│   │   ├── memory.py              — MemoryProbe (psutil, torch.cuda, cupy, pynvml)
│   │   ├── synchronize.py
│   │   ├── writer.py              — ResultWriter (raw + metadata)
│   │   └── hardware.py            — collect_hardware_info()
│   └── suites/
│       ├── base.py                — BenchmarkSuite protocol
│       ├── benchmark.py           — BenchmarkSuite: runtime + accuracy
│       └── memory.py              — MemorySuite
│
├── analysis/
│   ├── __init__.py
│   ├── loaders.py                 — load_raw(results_dir), load_metadata(results_dir)
│   ├── processing/
│   │   ├── summary.py             — build_summary
│   │   ├── pareto.py              — build_pareto
│   │   ├── scaling.py             — build_scaling
│   │   └── pipeline.py            — run_pipeline(raw_dir, processed_dir)
│   ├── figures/
│   │   ├── common.py              — shared chart helpers (colors, capping, error bars)
│   │   ├── breakdown.py           — per-scenario runtime/memory bar figures
│   │   ├── scaling.py             — runtime/memory-vs-size line figure
│   │   └── tables.py              — HTML detail/speedup tables
│   └── report.py                  — assembles analysis.figures.* into the
│                                     single report.html (tabbed page: Pareto,
│                                     scaling, action breakdown, memory)
│
├── configs/
│   ├── run.yaml
│   ├── backend/
│   ├── trajectory/
│   ├── mri_setup/
│   ├── action/
│   └── benchmark/
│
├── assets/
│   ├── trajectories/
│   └── smaps/
│
├── scripts/
│   ├── generate_assets.py         — one-time asset generation
│   └── build_report.py            — regenerate processed data + report.html
│
├── run.py                         — @hydra.main entry point
│
├── benchmark_results/              — gitignored; populated at runtime
│
├── pyproject.toml
└── environment.lock.yml
```

### Key design principles

**The benchmark and analysis packages share no runtime code.** The boundary
between them is the Parquet files in `benchmark_results/`. The analysis
pipeline can run on a laptop using data collected on a remote GPU cluster.

**Runners don't know about backends; backends don't know about timing;
suites don't know about files.** This keeps each piece testable with a small
fixture, without spinning up a real benchmark.

**File I/O is confined to three places:** `infra/writer.py` (writes raw
results), `analysis/loaders.py` (reads raw results), and `analysis/report.py`
(writes HTML). Everything else operates on in-memory objects.

**Adding a new backend** means a new YAML file in `configs/backend/` and a
new `BackendAdapter` subclass registered in `benchmark/backends/registry.py`.
No changes to `run.py`, the dataclasses, or any other config file are needed.

**Hydra owns the config boundary; `benchmark/` knows nothing about YAML.**
The `benchmark/` package receives a fully validated `BenchmarkConfig`
dataclass instance — it never reads YAML files or calls the Hydra API
directly, so it can be exercised in unit tests without Hydra initialization.

---

## Fair Comparison Principles

### Baseline backend

FINUFFT (CPU, `eps=1e-6`) is the canonical baseline for normalized
comparisons. Relative speedup figures use this backend as the denominator.

### Cross-backend requirements

- All backends operate on the exact same `.npy` trajectory files.
- All backends use the same test vectors (generated from the same fixed
  seed, recorded in the run config).
- No trajectory generation occurs at benchmark time.
- All timings use the same warmup protocol and stopping condition.
- Memory measurements are always in a separate execution from timing
  measurements.

---

## Derived Datasets

### Pareto dataset

Contains one row per (backend, configuration) pair with accuracy and runtime.
The `is_pareto_optimal` flag marks points on the Pareto frontier of the
runtime-vs-accuracy trade-off.

```text
backend
action
runtime_median_ms
throughput_median
relative_l2_error
is_pareto_optimal
```

A point is Pareto-optimal if no other point is both faster and more accurate.

### Scaling dataset

Derived from the summary table, filtered to action `forward` and the
`spiral_2d_*_standard`/`sos_3d_*_standard` trajectory families (the two
families with more than one scenario row, per the scenario list above).
Contains one row per (backend, trajectory_id, variable, value) triplet —
`trajectory_id` disambiguates points that would otherwise share the same
(backend, variable, value) key across families (e.g. `ncoils=1` exists for
both the 2D and 3D multi-coil trajectories):

- `variable=nx`: 2D-S vs 2D-M vs 2D-L (all 1 coil), and 3D-S vs 3D-M vs 3D-L
  (all 1 coil) — image-size scaling.
- `variable=ncoils`: 2D-S, 2D-M, 2D-L, 3D-S, 3D-M, and 3D-L each at 1/8/32
  coils — coil scaling.

---

## Expected Figures

`report.html` has seven tabs, grouped into three sections in the tab bar
(a visual grouping only — each tab is still an independent panel):

**Per-Scenario Breakdown** (Runtime, GPU Memory, CPU Memory):

- *Runtime* is a grid faceted by action (rows: `forward`/`adjoint`/
  `normal_operator`/`operator_init`, in that order) and by base trajectory
  family (columns: 2D-S, 2D-M, 2D-L, 3D-S, 3D-M, 3D-L). Within each panel,
  bars run one per backend (y-axis), grouped and colored by ncoils (1/8/32
  coils, where that trajectory has multi-coil variants) — the metric is
  **runtime per coil** (`runtime_median_ms / ncoils`), not raw runtime, so
  the 1/8/32-coil variants of the same trajectory land in the same panel as
  directly comparable bars instead of needing a separate column each.
  Restricted to `input_location=host`.
- *GPU Memory* and *CPU Memory* are a grid of horizontal grouped bar plots,
  restricted to `input_location=host` so each (backend, action) pair appears
  once. Rows: action, same order as above. Within each panel, bars run one
  per backend (y-axis), colored by scenario (2D-S, 2D-S/8-coil, 2D-S/32-coil,
  2D-M, 2D-M/8-coil, 2D-M/32-coil, 2D-L, 2D-L/8-coil, 2D-L/32-coil, 3D-S,
  3D-S/8-coil, 3D-S/32-coil, 3D-M, 3D-M/8-coil, 3D-M/32-coil, 3D-L,
  3D-L/8-coil, 3D-L/32-coil) — raw (not per-coil) peak memory, read from the
  memory suite.

Runtime bars carry an asymmetric error bar spanning the [p5, p95] replicate
range around the median (divided by ncoils along with the bar itself, for the
per-coil Runtime panel) — memory bars don't, since peak memory is a single
max over replicates (see summary.py), not a distribution.

A lone backend that's much larger than the runner-up in its panel (e.g. a
pure-Python reference implementation, or a 3D scenario dwarfing every 2D
one) has its bar capped at ~1.2x the runner-up's value and labelled with its
true number (and its error bar dropped, since it'd otherwise anchor at the
wrong position), instead of compressing every other bar to invisibility on a
linear axis.

**GPU-Resident Input** (GPU Resident Runtime, GPU Resident Memory) compares
`input_location=host` against `input_location=device` for `forward`/`adjoint`
on the cuda-capable backends only (the ones that actually ran with
`input_location=device`). Grid: one row per action, one column per scenario
(raw, not per-coil); bars run one per backend, grouped and colored by
`input_location` (host vs. device); runtime bars carry the same [p5, p95]
error bar as above.

**Scaling** (Scaling Runtime, Scaling Memory) plots the scaling dataset (see
above) as line charts: one row per `variable` (`nx`, `ncoils`), one column
per trajectory family (2D spiral, 3D stack-of-spirals), one line per backend
(raw, not per-coil), x-axis log-scaled with ticks at the actual scenario
values (e.g. `ncoils=1/8/32`). Runtime's y-axis is also log-scaled, since
backends span orders of magnitude; memory's isn't, since a 0 MB
persistent-allocation point (a real, valid measurement for some GPU backends)
can't be placed on a log axis.

Every bar chart uses a narrow `bargap`/`bargroupgap` so bars read as thick,
filled shapes rather than thin lines. Runtime axes are labelled with their
unit (`ms`, or `ms / coil` for the per-coil Runtime panel).

Figures are sized generously (up to ~1600px wide, ~480px per action row)
with larger fonts throughout, since these are read zoomed-in on a wide
screen, not thumbnailed.

In every bar panel, backends are ordered CPU first (alphabetical), then
GPU/cuda (alphabetical) — see `analysis.scenarios.BACKEND_DEVICES` — with a
shaded background band over the GPU rows marking the group boundary.

---

## Analysis Stack

| Component     | Tool           | Purpose                                        |
| ------------- | -------------- | ---------------------------------------------- |
| Storage       | Parquet        | Columnar, compressed, fast aggregation         |
| Processing    | Polars         | Fast DataFrame operations for derived datasets |
| Visualization | Plotly         | Interactive HTML report                        |
| CI            | GitHub Actions | Runs the benchmark matrix on demand            |

---

## Known Limitations

The following are explicitly out of scope for this version and should not be
assumed to be covered. Add scenario definitions and schema fields for these
deliberately, once the core pipeline above is running and validated:

- **Toeplitz acceleration benchmarking** (init cost, apply cost, break-even
  analysis).
- **Composite actions** (`cg_iteration`, `reconstruction`).
- **Automated CI regression tracking/gating** on runtime or memory drift.
- **Multi-GPU / distributed backends**.
- **Non-uniform density compensation** in accuracy comparisons.
- **Half-precision (FP16)**.
- **Batched reconstruction** (multiple volumes in a single call).
- **`pynfft` on multi-coil (`ncoils>1`) scenarios**: mri-nufft's pyNFFT3
  binding raises on the `forward` action for `ncoils>1` (its per-coil array
  slice isn't the C-contiguous complex128 array pyNFFT3's plan requires) -
  an upstream mri-nufft/pyNFFT3 integration issue, not fixable from this
  repo's backend registry. `pynfft` is otherwise run on every scenario;
  this one (backend, action) cell is simply absent from the results.
- **`torchkbnufft-cpu` on the 32-coil scenarios**: deliberately skipped by
  `scripts/run_all.sh` (see `SLOW_BACKENDS`), not a failure - its
  per-coil, unbatched FFT plan construction makes `operator_init` at 32
  coils expensive enough to dominate the whole matrix's wall time. Run it
  manually via `run.py` (see how-to.md) if you need that data point.
- **`finufft`'s requested `eps=1e-6` is a single-precision floor, not always
  achievable**: FINUFFT's `eps` is a *relative* ℓ2-norm accuracy target, but
  single precision (`complex64`) can't get below roughly `N_max * eps_mach`
  (`eps_mach ≈ 6e-8`) for the largest grid dimension `N_max` - a hard
  numerical limit, not a bug. `configs/backend/finufft.yaml` sets
  `allow_eps_too_small: 1` so FINUFFT clamps to the best achievable accuracy
  and proceeds instead of hard-erroring (FINUFFT's own escape hatch for
  this); it logs a `check_sigma warning` to stderr in that case. This means
  finufft's `relative_l2_error` may sit above `1e-6` on the larger scenarios
  (2D-L, 3D-L, and to a lesser extent 2D-M/3D-M) - check the accuracy
  columns rather than assuming the requested `eps` was met.
