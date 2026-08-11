#!/usr/bin/env bash
# Build the release wheel: prebuilt CUDA binary for the architectures users
# actually train on. No PTX -- a device outside this list gets an actionable
# error at first use (CheckCUDADeviceSupportsThisBuild) pointing at the sdist,
# which auto-detects the local GPU. Run inside a manylinux container for a
# publishable platform tag; see the release checklist.
#
#   FALCATA_WHEEL_ARCHS  override the architecture list
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
ARCHS="${FALCATA_WHEEL_ARCHS:-61-real;70-real;75-real;80-real;86-real;89-real;90-real;100-real;120-real}"

cd "$HERE"
CMAKE_ARGS="-DCMAKE_CUDA_ARCHITECTURES=${ARCHS} -DBUILD_WITH_SHARED_NCCL=ON -DFALCATA_CUDA_LINEINFO=OFF" \
  sh build-python.sh bdist_wheel --cuda
ls -la dist/*.whl | tail -1
