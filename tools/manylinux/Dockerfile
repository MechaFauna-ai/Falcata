# Build environment for the falcata release wheel: manylinux_2_28 (glibc 2.28,
# runs on Ubuntu 20.04+/Debian 10+/RHEL 8+) plus the CUDA 12.9 toolkit.
# Compiling needs no GPU. auditwheel ships with the base image.
FROM quay.io/pypa/manylinux_2_28_x86_64

RUN dnf config-manager --add-repo \
      https://developer.download.nvidia.com/compute/cuda/repos/rhel8/x86_64/cuda-rhel8.repo && \
    dnf install -y cuda-nvcc-12-9 cuda-cudart-devel-12-9 cuda-nvrtc-devel-12-9 cuda-driver-devel-12-9 libnvjitlink-devel-12-9 && \
    dnf clean all

ENV PATH=/usr/local/cuda-12.9/bin:$PATH \
    CUDACXX=/usr/local/cuda-12.9/bin/nvcc
