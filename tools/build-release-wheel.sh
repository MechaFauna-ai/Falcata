#!/usr/bin/env bash
# Build the release wheel (CUDA 13): prebuilt CUDA binary for the architectures
# users actually train on (sm_75 upward). No PTX -- a device outside this list
# gets an actionable error at first use (CheckCUDADeviceSupportsThisBuild)
# pointing at the sdist, which auto-detects the local GPU. Run inside a
# manylinux container for a publishable platform tag; see the release checklist.
#
#   FALCATA_WHEEL_ARCHS  override the architecture list
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
# CUDA 13 cannot emit pre-Turing code (sm_50..sm_72), so the list starts at 75.
# Cubins are upward-compatible within a major: 80/86/89 cover all Ampere and
# Ada, 100 covers sm_100 (B200/GB200) and sm_103 (GB300), 120 covers RTX 50xx and sm_121.
# Pascal/Volta (P100, GTX 10xx, V100) users build from source with CUDA 12.x.
# PTX would be useless until CheckCUDADeviceSupportsThisBuild accepts an
# unknown major that a -virtual entry can JIT; follow-up, not 1.1.
ARCHS="${FALCATA_WHEEL_ARCHS:-75-real;80-real;86-real;89-real;90-real;100-real;120-real}"

cd "$HERE"
CMAKE_ARGS="-DCMAKE_CUDA_ARCHITECTURES=${ARCHS} -DUSE_NCCL=ON -DFALCATA_CUDA_LINEINFO=OFF" \
  sh build-python.sh bdist_wheel --cuda

# manylinux tag + graft libgomp; libcuda comes from the user's driver
RAW=$(find dist -name "falcata-*-py3-none-linux_x86_64.whl" | sort | tail -1)
auditwheel repair --exclude libcuda.so.1 "$RAW" -w dist/
REPAIRED=$(find dist -name "falcata-*manylinux*.whl" | sort | tail -1)

# let the loader find nccl inside the nvidia wheel's install location.
# The manylinux images ship pip but not the wheel module, and without it the
# rpath below is never applied -- the wheel then loads no nccl.
python -c 'import wheel' 2>/dev/null || python -m pip install --quiet --user wheel
python -m wheel unpack "$REPAIRED" -d /tmp/fal-wheel
LIB=$(ls /tmp/fal-wheel/*/falcata/lib/lib_falcata.so)
OLD_RPATH=$(patchelf --print-rpath "$LIB")
# nccl for the multi-GPU dlopen (falcata[multigpu])
patchelf --set-rpath "${OLD_RPATH}:\$ORIGIN/../../nvidia/nccl/lib" "$LIB"
python -m wheel pack /tmp/fal-wheel/* -d dist/
rm -rf /tmp/fal-wheel
du -h dist/*manylinux*.whl
