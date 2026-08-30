#!/bin/bash
# Build and run the two suites that need a native binary rather than the Python
# package: the GoogleTest C++ suite (tests/cpp_tests) and the distributed
# training tests (tests/distributed), which drive the falcata CLI over sockets.
#
# GH CI runs both against a CPU build. This runs them against a CUDA build --
# the configuration that ships -- which is also the only place the CUDA
# device-link path of the test binary is exercised.
#
# The build tree is kept (FALCATA_CPP_TEST_BUILD) so nightly runs are
# incremental; ccache is used when present. The binaries land in the repo root,
# which is where this project's CMake puts executables.
set -euo pipefail

REPO="${FALCATA_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
BUILD="${FALCATA_CPP_TEST_BUILD:-$REPO/.cpp-test-build}"
PYTHON="${FALCATA_GATES_PYTHON:-$REPO/.gates-venv/bin/python}"

# Same pinned toolchain as ci_build.sh: the box has several CUDA toolkits and
# /usr/local/cuda may point at an unvalidated one.
CUDA_HOME="${FALCATA_CUDA_HOME:-/usr/local/cuda-12.9}"
export PATH="$CUDA_HOME/bin:$PATH"
export CUDACXX="$CUDA_HOME/bin/nvcc"

cmake_args=(
  -S "$REPO" -B "$BUILD"
  -DBUILD_CPP_TEST=ON
  -DUSE_CUDA=ON
  -DCMAKE_BUILD_TYPE=Release
  -DCMAKE_CUDA_ARCHITECTURES="120-real;120-virtual"
)
if command -v ccache >/dev/null 2>&1; then
  cmake_args+=(-DCMAKE_CXX_COMPILER_LAUNCHER=ccache -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache)
fi

cmake "${cmake_args[@]}"
# Only these two targets: this suite runs the test binary and the CLI, and
# building everything would add a lib_falcata.so relink no gate reads.
cmake --build "$BUILD" --target testfalcata falcata -j "$(nproc)"

echo "== C++ tests (CUDA build)"
"$REPO/testfalcata"

echo "== distributed tests (CUDA build, CPU training)"
"$PYTHON" -m pytest "$REPO/tests/distributed/_test_distributed.py" \
  -q -p no:cacheprovider --timeout=600 --execfile "$REPO/falcata"
