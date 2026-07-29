#!/usr/bin/env bash
# Set up the two python environments for the GPU benchmark suite:
#   env-falcata     Falcata built from THIS checkout (CUDA)
#   env-competitors  upstream LightGBM (CUDA, built from PyPI sdist),
#                    XGBoost and CatBoost (PyPI wheels, ship CUDA support)
#
# Environment variables:
#   FALCATA_BENCH_ROOT   workspace dir (default: benchmarks/workspace)
#   FALCATA_CUDA_ARCHS   CMAKE_CUDA_ARCHITECTURES (default: native)
#   UPSTREAM_LGBM_VERSION upstream LightGBM version (default: latest on PyPI)
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
git -C "$REPO" rev-parse HEAD > "$ROOT/results/exaboost_sha.txt"
python -c "import lightgbm as l; print('falcata OK', l.__version__)"
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

echo "setup complete; workspace: $ROOT"
