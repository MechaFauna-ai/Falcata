#!/bin/bash
# Build the Falcata CUDA wheel into a fresh venv for the gates suite.
# Run from the repo root on the self-hosted GPU runner.
# Uses ccache for CXX/CUDA when available (install it: sudo apt install ccache
# -- cuts warm rebuilds from ~6 min to ~1-2 min).
set -euo pipefail

VENV="${GATES_VENV:-.gates-venv}"

# Pin the validated CUDA toolchain (12.9): the box has several toolkits and
# the runner's service environment has none of them on PATH. /usr/local/cuda
# may point at a newer, unvalidated toolkit -- do not use it.
CUDA_HOME="${FALCATA_CUDA_HOME:-/usr/local/cuda-12.9}"
export PATH="$CUDA_HOME/bin:$PATH"
export CUDACXX="$CUDA_HOME/bin/nvcc"

python3 -m venv "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
pip install --quiet --upgrade pip
# xgboost/catboost: the FALB import-parity gates (import_xgboost.py,
# import_catboost.py) train tiny reference models with them.
# pytest + friends: the nightly runs tests/python_package_test on CUDA, which
# hosted CI cannot do (no GPU runners). pytest-forked is not optional there --
# a CUDA context error is sticky, so without one process per test a single
# failure cascades into dozens and eventually aborts the run.
pip install --quiet numpy scipy scikit-learn pyarrow xgboost catboost \
  cloudpickle matplotlib pandas psutil pytest pytest-forked pytest-timeout

CMAKE_ARGS="-DCMAKE_CUDA_ARCHITECTURES=120-real;120-virtual -DUSE_NCCL=ON -DBUILD_WITH_SHARED_NCCL=ON"
if command -v ccache >/dev/null 2>&1; then
  CMAKE_ARGS="$CMAKE_ARGS -DCMAKE_CXX_COMPILER_LAUNCHER=ccache -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache"
  echo "ci_build: ccache enabled"
else
  echo "ci_build: ccache NOT found -- full rebuild every run (sudo apt install ccache to fix)"
fi
export CMAKE_ARGS

# Build into the gate's own directory. The build wipes its output dir first, so
# using the default dist/ meant a nightly run deleted whatever release artifact
# was staged there -- it took out a signed 1.0.2 wheel once.
FALCATA_DIST_DIR="${FALCATA_DIST_DIR:-$PWD/.gates-dist}"
export FALCATA_DIST_DIR

sh build-python.sh install --cuda

python -c "import falcata; print('falcata', falcata.__version__, '->', falcata.__file__)"
