#!/bin/bash
# Build the ExaBoost CUDA wheel into a fresh venv for the gates suite.
# Run from the repo root on the self-hosted GPU runner.
# Uses ccache for CXX/CUDA when available (install it: sudo apt install ccache
# -- cuts warm rebuilds from ~6 min to ~1-2 min).
set -euo pipefail

VENV="${GATES_VENV:-.gates-venv}"

python3 -m venv "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
pip install --quiet --upgrade pip
pip install --quiet numpy scipy scikit-learn

CMAKE_ARGS="-DCMAKE_CUDA_ARCHITECTURES=120-real;120-virtual -DBUILD_WITH_SHARED_NCCL=ON"
if command -v ccache >/dev/null 2>&1; then
  CMAKE_ARGS="$CMAKE_ARGS -DCMAKE_CXX_COMPILER_LAUNCHER=ccache -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache"
  echo "ci_build: ccache enabled"
else
  echo "ci_build: ccache NOT found -- full rebuild every run (sudo apt install ccache to fix)"
fi
export CMAKE_ARGS

sh build-python.sh install --cuda

python -c "import lightgbm; print('lightgbm', lightgbm.__version__, '->', lightgbm.__file__)"
