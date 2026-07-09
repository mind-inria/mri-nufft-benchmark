#!/bin/sh
# Bootstrap a self-contained mri-nufft-benchmark workspace: creates a top
# folder, clones this repo plus the two sibling repos its pyproject.toml
# expects at ../mri-nufft and ../finufft (mri-nufft, cufinufft and finufft
# are installed as editable path dependencies - see pyproject.toml's
# [tool.uv.sources]) inside it, syncs the environment, and pre-generates the
# trajectory/smaps assets.
#
# uv is only installed if not already on PATH, and then only into the
# workspace folder itself (UV_UNMANAGED_INSTALL: no PATH/shell-profile
# changes, no self-update) - nothing is written outside the workspace.
#
# Usage:
#   curl -LsSf https://raw.githubusercontent.com/mind-inria/mri-nufft-benchmark/main/scripts/setup.sh | sh
# Workspace folder defaults to ./mri-nufft-benchmark-workspace; override with:
#   curl -LsSf .../setup.sh | WORKSPACE_DIR=/path/to/workspace sh
#
# POSIX sh (not bash) - this runs piped through `sh`, not sourced, so it
# can't rely on bashisms (arrays, [[, local).
set -eu

MRI_NUFFT_URL="https://github.com/mind-inria/mri-nufft"
FINUFFT_URL="https://github.com/flatironinstitute/finufft"
BENCH_URL="https://github.com/mind-inria/mri-nufft-benchmark"

log() { echo "== $1 =="; }

if ! command -v git >/dev/null 2>&1; then
  echo "git is required but not found on PATH." >&2
  exit 1
fi

# Running from inside an existing checkout (./scripts/setup.sh) vs. a fresh
# directory (curl | sh). In the former case the workspace is just the
# repo's parent - no new top folder needed, the checkout already is one.
if [ -f pyproject.toml ] && grep -q '^name = "mri-nufft-benchmark"' pyproject.toml 2>/dev/null; then
  BENCH_DIR=$(pwd)
  WORKSPACE_DIR=$(dirname "$BENCH_DIR")
else
  WORKSPACE_DIR="${WORKSPACE_DIR:-$(pwd)/mri-nufft-benchmark-workspace}"
  mkdir -p "$WORKSPACE_DIR"
  BENCH_DIR="$WORKSPACE_DIR/mri-nufft-benchmark"
  if [ ! -d "$BENCH_DIR" ]; then
    log "Cloning mri-nufft-benchmark into $WORKSPACE_DIR"
    git clone "$BENCH_URL" "$BENCH_DIR"
  fi
fi

clone_sibling() {
  name="$1"
  url="$2"
  target="$WORKSPACE_DIR/$name"
  if [ ! -d "$target" ]; then
    log "Cloning $name (editable path dependency) into $WORKSPACE_DIR"
    git clone "$url" "$target"
  fi
}

clone_sibling mri-nufft "$MRI_NUFFT_URL"
clone_sibling finufft "$FINUFFT_URL"

# Prefer an already-installed uv; otherwise install one scoped to this
# workspace only, so a fresh setup never touches the system PATH or shell
# profiles.
if command -v uv >/dev/null 2>&1; then
  UV=$(command -v uv)
else
  UV_DIR="$WORKSPACE_DIR/.uv"
  UV="$UV_DIR/uv"
  if [ ! -x "$UV" ]; then
    log "Installing uv locally into $UV_DIR (not touching system PATH)"
    curl -LsSf https://astral.sh/uv/install.sh | env UV_UNMANAGED_INSTALL="$UV_DIR" sh
  fi
fi

if ! command -v nvcc >/dev/null 2>&1 && ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "Warning: no CUDA toolkit/driver detected - cufinufft's GPU kernels" >&2
  echo "need a real CUDA toolkit visible to CMake to build; GPU backends" >&2
  echo "(cufinufft, gpunufft, torchkbnufft-gpu) will be unavailable." >&2
fi

cd "$BENCH_DIR"

log "Syncing environment (uv sync)"
"$UV" sync

log "Generating trajectory/smaps assets"
"$UV" run python scripts/generate_assets.py

log "Done"
echo "Workspace: $WORKSPACE_DIR"
echo "cd $BENCH_DIR && see how-to.md to run benchmarks."
case "$UV" in
  */.uv/uv) echo "uv was installed locally - run it as $UV, or add $(dirname "$UV") to PATH." ;;
esac
