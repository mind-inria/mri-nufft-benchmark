#!/usr/bin/env bash
# Reproduce the full benchmark: assets -> all scenarios x backends x actions
# (benchmark + memory suites) -> report.html. See how-to.md for details.
#
# A single (backend, action) job crashing (e.g. a bad eps for one backend)
# is already isolated by Hydra's multirun launcher - it logs the traceback
# and keeps running the rest of that sweep. What -e would otherwise do is
# abort the *whole script* on that one sweep's nonzero exit, skipping every
# later scenario. So failures here are tracked and reported at the end
# instead, and the report is still built from whatever rows were collected.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

BACKENDS=finufft,cufinufft,gpunufft,ducc0,pynfft,torchkbnufft-cpu,torchkbnufft-gpu
GPU_BACKENDS=cufinufft,gpunufft,torchkbnufft-gpu
# torchkbnufft-cpu builds one FFT plan per coil with no batching, so its
# operator_init cost scales ~linearly with ncoils - at 32 coils on the
# larger scenarios that's expensive enough to dominate the whole matrix's
# wall time for one backend's data. Dropped from the 32-coil rows only;
# it still runs on every other scenario. (pynfft is also excluded from any
# ncoils>1 scenario, but that's an upstream failure, not a speed problem -
# see README's Known Limitations - so it doesn't need listing here.)
SLOW_BACKENDS=torchkbnufft-cpu,pynfft,torchkbnufft-gpu
FAST_BACKENDS=finufft,cufinufft,gpunufft,ducc0
ACTIONS=operator_init,forward,adjoint,normal_operator

# scenario rows: "trajectory mri_setup backend_tier" - backend_tier selects
# $BACKENDS ("all") or $FAST_BACKENDS ("fast") for that row.
SCENARIOS=(
  "spiral_2d_128_standard 2d_s_1coil all"
  "spiral_2d_128_standard 2d_s_8coils all"
  "spiral_2d_128_standard 2d_s_32coils fast"
  "spiral_2d_256_standard 2d_m_1coil all"
  "spiral_2d_256_standard 2d_m_8coils all"
  "spiral_2d_256_standard 2d_m_32coils fast"
  "spiral_2d_512_standard 2d_l_1coil all"
  "spiral_2d_512_standard 2d_l_8coils all"
  "spiral_2d_512_standard 2d_l_32coils fast"
  "sos_3d_64_standard 3d_s_1coil all"
  "sos_3d_64_standard 3d_s_8coils all"
  "sos_3d_64_standard 3d_s_32coils fast"
  "sos_3d_192_standard 3d_m_1coil all"
  "sos_3d_192_standard 3d_m_8coils all"
  "sos_3d_192_standard 3d_m_32coils fast"
  "sos_3d_256_standard 3d_l_1coil fast"
  "sos_3d_256_standard 3d_l_8coils fast"
  "sos_3d_256_standard 3d_l_32coils fast"
)

FAILURES=()

run_sweep() {
  local desc="$1"
  shift
  echo "== ${desc} =="
  if ! uv run python run.py -m "$@"; then
    echo "!! FAILED: ${desc} (see traceback above) - continuing with remaining scenarios" >&2
    FAILURES+=("${desc}")
  fi
}

echo "== Generating trajectory/smaps assets =="
uv run python scripts/generate_assets.py || { echo "asset generation failed, aborting" >&2; exit 1; }

for scenario in "${SCENARIOS[@]}"; do
  read -r trajectory mri_setup backend_tier <<< "$scenario"
  backends="${BACKENDS}"
  [ "${backend_tier}" = "fast" ] && backends="${FAST_BACKENDS}"

  run_sweep "Benchmark suite: ${trajectory} / ${mri_setup}" \
    backend="${backends}" \
    trajectory="${trajectory}" \
    mri_setup="${mri_setup}" \
    action="${ACTIONS}"

  run_sweep "Benchmark suite (GPU-resident input): ${trajectory} / ${mri_setup}" \
    backend="${GPU_BACKENDS}" \
    trajectory="${trajectory}" \
    mri_setup="${mri_setup}" \
    action=forward,adjoint \
    input_location=device

  run_sweep "Memory suite: ${trajectory} / ${mri_setup}" \
    suite=memory benchmark=memory \
    backend="${backends}" \
    trajectory="${trajectory}" \
    mri_setup="${mri_setup}" \
    action="${ACTIONS}"

  run_sweep "Memory suite (GPU-resident input): ${trajectory} / ${mri_setup}" \
    suite=memory benchmark=memory \
    backend="${GPU_BACKENDS}" \
    trajectory="${trajectory}" \
    mri_setup="${mri_setup}" \
    action=forward,adjoint \
    input_location=device
done

echo "== Building report =="
uv run python scripts/build_report.py || { echo "report build failed" >&2; exit 1; }

if [ "${#FAILURES[@]}" -gt 0 ]; then
  echo "Done with failures. Open benchmark_results/report.html"
  printf ' - %s\n' "${FAILURES[@]}"
  exit 1
fi

echo "Done. Open benchmark_results/report.html"
