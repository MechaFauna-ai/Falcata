#!/usr/bin/env bash
# Set up the python environments for the GPU benchmark suite:
#   env-falcata      Falcata built from THIS checkout (CUDA)
#   env-competitors  upstream LightGBM (CUDA, built from PyPI sdist),
#                    XGBoost and CatBoost (PyPI wheels, ship CUDA support)
#   env-lightgbm-ocl upstream LightGBM built against its legacy OpenCL backend
#
# Three environments because Falcata and upstream LightGBM both install under
# the import name "lightgbm", and upstream's CUDA and OpenCL backends are
# separate builds of the same package.
#
# Environment variables:
#   FALCATA_BENCH_ROOT   workspace dir (default: benchmarks/workspace)
#   FALCATA_CUDA_ARCHS   CMAKE_CUDA_ARCHITECTURES (default: native)
#   UPSTREAM_LGBM_VERSION upstream LightGBM version (default: latest on PyPI)
#   SKIP_OCL=1           skip the OpenCL environment
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(dirname "$HERE")"
ROOT="${FALCATA_BENCH_ROOT:-$HERE/workspace}"
ARCHS="${FALCATA_CUDA_ARCHS:-native}"

mkdir -p "$ROOT"/{data,results,report}

echo "=== env-falcata: building Falcata from $REPO (archs: $ARCHS)"
python3 -m venv "$ROOT/env-falcata"
# shellcheck disable=SC1091
source "$ROOT/env-falcata/bin/activate"
pip install -q --upgrade pip
pip install -q -r "$HERE/requirements.txt"
# BUILD_WITH_SHARED_NCCL avoids nvlink failures against the static NCCL on
# arches it was not device-linked for (e.g. Blackwell)
(cd "$REPO" && CMAKE_ARGS="-DCMAKE_CUDA_ARCHITECTURES=$ARCHS -DBUILD_WITH_SHARED_NCCL=ON" sh build-python.sh install --cuda)
git -C "$REPO" rev-parse HEAD > "$ROOT/results/falcata_sha.txt"
python -c "import falcata as f; print('falcata OK', f.__version__)"
deactivate

echo "=== env-competitors: upstream LightGBM (CUDA) + XGBoost + CatBoost"
python3 -m venv "$ROOT/env-competitors"
# shellcheck disable=SC1091
source "$ROOT/env-competitors/bin/activate"
pip install -q --upgrade pip
pip install -q -r "$HERE/requirements.txt"
pip install -q xgboost catboost
CMAKE_ARGS="-DCMAKE_CUDA_ARCHITECTURES=$ARCHS" \
  pip install --no-binary lightgbm --config-settings=cmake.define.USE_CUDA=ON \
  "lightgbm${UPSTREAM_LGBM_VERSION:+==$UPSTREAM_LGBM_VERSION}"
python -c "import lightgbm as l, xgboost as x, catboost as c; print('upstream lgbm', l.__version__, '| xgb', x.__version__, '| catboost', c.__version__)"
deactivate

if [ -z "${SKIP_OCL:-}" ]; then
  # Upstream's legacy OpenCL backend, which the suite runs as `lightgbm-ocl`.
  # It is the reference wherever upstream's CUDA backend crashes, so without it
  # those cells have no upstream number at all. Needs an OpenCL ICD and Boost
  # headers (Debian/Ubuntu: ocl-icd-opencl-dev libboost-dev
  # libboost-filesystem-dev libboost-system-dev); the build is non-fatal
  # because neither is guaranteed present.
  echo "=== env-lightgbm-ocl: upstream LightGBM (OpenCL backend)"
  python3 -m venv "$ROOT/env-lightgbm-ocl"
  # shellcheck disable=SC1091
  source "$ROOT/env-lightgbm-ocl/bin/activate"
  pip install -q --upgrade pip
  pip install -q -r "$HERE/requirements.txt"
  if pip install --no-binary lightgbm --config-settings=cmake.define.USE_GPU=ON \
      "lightgbm${UPSTREAM_LGBM_VERSION:+==$UPSTREAM_LGBM_VERSION}"; then
    python -c "import lightgbm as l; print('upstream lgbm OpenCL', l.__version__)"
  else
    echo "WARNING: OpenCL build failed -- lightgbm-ocl cells will be skipped." >&2
    echo "         Install the OpenCL/Boost dev packages and re-run, or SKIP_OCL=1." >&2
  fi
  deactivate
fi

echo "setup complete; workspace: $ROOT"
