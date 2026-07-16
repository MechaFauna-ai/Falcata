// Empirical DRAM bandwidth microbench for the RTX 5090.
// Measures sustained streaming read+write (copy) and pure read (reduce) GB/s.
#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>

__global__ void copy_kernel(const float4* __restrict__ in, float4* __restrict__ out, size_t n) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = (size_t)gridDim.x * blockDim.x;
  for (; i < n; i += stride) out[i] = in[i];
}

__global__ void read_kernel(const float4* __restrict__ in, size_t n, float4* __restrict__ sink) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = (size_t)gridDim.x * blockDim.x;
  float4 acc = make_float4(0,0,0,0);
  for (; i < n; i += stride) { float4 v = in[i]; acc.x += v.x; acc.y += v.y; acc.z += v.z; acc.w += v.w; }
  if (acc.x == -1.0f) sink[0] = acc;  // never taken; prevents DCE
}

int main() {
  size_t bytes = (size_t)2 * 1024 * 1024 * 1024;  // 2 GiB per buffer
  size_t n = bytes / sizeof(float4);
  float4 *in, *out;
  cudaMalloc(&in, bytes); cudaMalloc(&out, bytes);
  cudaMemset(in, 1, bytes); cudaMemset(out, 0, bytes);
  int block = 256; int grid = 0;
  cudaDeviceProp p; cudaGetDeviceProperties(&p, 0);
  grid = p.multiProcessorCount * 32;
  cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);
  const int iters = 50;
  // warmup
  for (int i=0;i<5;i++) copy_kernel<<<grid,block>>>(in,out,n);
  cudaDeviceSynchronize();
  cudaEventRecord(s);
  for (int i=0;i<iters;i++) copy_kernel<<<grid,block>>>(in,out,n);
  cudaEventRecord(e); cudaEventSynchronize(e);
  float ms=0; cudaEventElapsedTime(&ms,s,e);
  double gbps_copy = (double)bytes*2*iters / (ms*1e-3) / 1e9;  // read+write
  // read-only
  for (int i=0;i<5;i++) read_kernel<<<grid,block>>>(in,n,out);
  cudaDeviceSynchronize();
  cudaEventRecord(s);
  for (int i=0;i<iters;i++) read_kernel<<<grid,block>>>(in,n,out);
  cudaEventRecord(e); cudaEventSynchronize(e);
  cudaEventElapsedTime(&ms,s,e);
  double gbps_read = (double)bytes*iters / (ms*1e-3) / 1e9;
  printf("SM count=%d\n", p.multiProcessorCount);
  printf("COPY (r+w) sustained: %.1f GB/s\n", gbps_copy);
  printf("READ-only  sustained: %.1f GB/s\n", gbps_read);
  cudaFree(in); cudaFree(out);
  return 0;
}
