#!/usr/bin/env bash
# Reproduce the full benchmark: assets -> all scenarios x backends x actions
# (benchmark + memory suites) -> report.html. See how-to.md for details.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

BACKENDS=finufft,cufinufft,torchkbnufft,gpunufft
ACTIONS=operator_init,forward,adjoint,normal_operator

# scenario rows: "trajectory mri_setup"
SCENARIOS=(
  "spiral_2d_128_standard 2d_s_1coil"
  "spiral_2d_256_standard 2d_m_1coil"
  "spiral_2d_256_standard 2d_m_8coils"
  "sos_3d_64_standard 3d_s_1coil"
  "sos_3d_192_standard 3d_m_1coil"
)

echo "== Generating trajectory/smaps assets =="
uv run python scripts/generate_assets.py

for scenario in "${SCENARIOS[@]}"; do
  read -r trajectory mri_setup <<< "$scenario"

  echo "== Benchmark suite: ${trajectory} / ${mri_setup} =="
  uv run python run.py -m \
    backend="${BACKENDS}" \
    trajectory="${trajectory}" \
    mri_setup="${mri_setup}" \
    action="${ACTIONS}"

  echo "== Memory suite: ${trajectory} / ${mri_setup} =="
  uv run python run.py -m \
    suite=memory benchmark=memory \
    backend="${BACKENDS}" \
    trajectory="${trajectory}" \
    mri_setup="${mri_setup}" \
    action="${ACTIONS}"
done

echo "== Building report =="
uv run python scripts/build_report.py

echo "Done. Open benchmark_results/report.html"
