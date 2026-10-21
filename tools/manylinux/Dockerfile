# Build environment for the falcata release wheel: manylinux_2_28 (glibc 2.28,
# runs on Ubuntu 20.04+/Debian 10+/RHEL 8+) plus the CUDA 13.3 toolkit, the
# same major the release targets (sm_75..sm_120, driver R580+; see
# tools/build-release-wheel.sh). Compiling needs no GPU. auditwheel ships with
# the base image.
FROM quay.io/pypa/manylinux_2_28_x86_64

RUN dnf config-manager --add-repo \
      https://developer.download.nvidia.com/compute/cuda/repos/rhel8/x86_64/cuda-rhel8.repo && \
    dnf install -y cuda-nvcc-13-3 cuda-cudart-devel-13-3 cuda-driver-devel-13-3 libnvjitlink-devel-13-3 && \
    dnf clean all

ENV PATH=/usr/local/cuda-13.3/bin:$PATH \
    CUDACXX=/usr/local/cuda-13.3/bin/nvcc
