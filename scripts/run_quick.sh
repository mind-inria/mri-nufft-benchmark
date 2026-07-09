#!/usr/bin/env bash
# Fast backend comparison: 2 representative scenarios (2D + 3D, both
# multi-coil) x all 7 backends x forward/adjoint, short measurement budget,
# writing to its own results dir so it never mixes with (or, via dedup,
# overwrites) full-matrix data. See how-to.md's "Quick backend comparison"
# section.
#
# Same job-isolation rationale as run_all.sh: a single (backend, action) job
# crashing is already isolated by Hydra's multirun launcher, so this doesn't
# abort on the first failure - it tracks failures and reports them at the end.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

BACKENDS=finufft,cufinufft,gpunufft,ducc0,pynfft,torchkbnufft-cpu,torchkbnufft-gpu
ACTIONS=forward,adjoint
RESULTS_DIR=benchmark_results_quick

SCENARIOS=(
  "spiral_2d_256_standard 2d_m_8coils"
  "sos_3d_64_standard 3d_s_8coils"
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
  read -r trajectory mri_setup <<< "$scenario"

  run_sweep "Quick benchmark suite: ${trajectory} / ${mri_setup}" \
    backend="${BACKENDS}" \
    trajectory="${trajectory}" \
    mri_setup="${mri_setup}" \
    action="${ACTIONS}" \
    benchmark=quick \
    results_dir="${RESULTS_DIR}"
done

echo "== Building quick report =="
uv run python scripts/build_quick_report.py || { echo "quick report build failed" >&2; exit 1; }

if [ "${#FAILURES[@]}" -gt 0 ]; then
  echo "Done with failures. Open ${RESULTS_DIR}/quick_report.html"
  printf ' - %s\n' "${FAILURES[@]}"
  exit 1
fi

echo "Done. Open ${RESULTS_DIR}/quick_report.html"
