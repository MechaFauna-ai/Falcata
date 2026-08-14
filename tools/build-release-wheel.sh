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
# 60-real, not 61-real: an sm_60 cubin runs on every Pascal device (CUDA
# binary compatibility is upward within a major), so one slot covers P100 AND
# GTX 10xx; the codebase uses no sm_61-only features (no dp4a).
ARCHS="${FALCATA_WHEEL_ARCHS:-60-real;70-real;75-real;80-real;86-real;89-real;90-real;100-real;120-real}"

cd "$HERE"
CMAKE_ARGS="-DCMAKE_CUDA_ARCHITECTURES=${ARCHS} -DUSE_NCCL=ON -DFALCATA_CUDA_LINEINFO=OFF" \
  sh build-python.sh bdist_wheel --cuda

# manylinux tag + graft libgomp; libcuda comes from the user's driver and
# libnvrtc from the nvidia-cuda-nvrtc-cu12 wheel this package depends on
RAW=$(find dist -name "falcata-*-py3-none-linux_x86_64.whl" | sort | tail -1)
auditwheel repair --exclude libcuda.so.1 --exclude "libnvrtc.so.12" "$RAW" -w dist/
REPAIRED=$(find dist -name "falcata-*manylinux*.whl" | sort | tail -1)

# let the loader find nvrtc inside the nvidia wheel's install location
python -m wheel unpack "$REPAIRED" -d /tmp/fal-wheel
LIB=$(ls /tmp/fal-wheel/*/falcata/lib/lib_falcata.so)
OLD_RPATH=$(patchelf --print-rpath "$LIB")
# nvrtc for the runtime JIT, nccl for the multi-GPU dlopen (falcata[multigpu])
patchelf --set-rpath "${OLD_RPATH}:\$ORIGIN/../../nvidia/cuda_nvrtc/lib:\$ORIGIN/../../nvidia/nccl/lib" "$LIB"
python -m wheel pack /tmp/fal-wheel/* -d dist/
rm -rf /tmp/fal-wheel
du -h dist/*manylinux*.whl
