// Lever 1 A/B: does baking BINS/COLS/row_stride as compile-time constants speed
#include <functional>
// up the quantized discretized construct inner loop vs the runtime-offset AOT
// kernel? Both kernels replicate ConstructDiscretizedHistogramDenseInner exactly
// (int16 packed shared hist, atomicAdd_block, 32-bit flush). Measured on real
// bench shapes. Occupancy is already ~94% (thread-bound 3 blk/SM on sm_120);
// this tests whether leaner ISA in the inner loop helps at fixed occupancy.
#include <cstdio>
#include <cstdint>
#include <vector>
#include <random>
#include <cuda_runtime.h>

// ---- RUNTIME variant: mirrors the AOT kernel; reads offsets from arrays. ----
template<int SHIST>
__global__ void construct_runtime(
    const int32_t* __restrict__ gh, const uint8_t* __restrict__ data,
    const int* __restrict__ part_col_off, const uint32_t* __restrict__ col_hist_off,
    const uint32_t* __restrict__ part_hist_off, const int32_t* __restrict__ indices,
    int num_data, int num_data_in_leaf, int64_t* __restrict__ out_hist) {
  __shared__ int16_t sh[SHIST];
  int32_t* shp = (int32_t*)sh;
  const int dim_y = gridDim.y * blockDim.y;
  const int ndpt = (num_data_in_leaf + dim_y - 1) / dim_y;
  const int block_start = (blockIdx.y * blockDim.y) * ndpt;
  if (block_start >= num_data_in_leaf) return;
  const int pcs = part_col_off[blockIdx.x];
  const int pce = part_col_off[blockIdx.x + 1];
  const int ncol = pce - pcs;
  const int row_stride = ncol;
  const uint8_t* dptr = data + (size_t)pcs * num_data;
  const uint32_t phs = part_hist_off[blockIdx.x];
  const uint32_t nitems = part_hist_off[blockIdx.x + 1] - phs;
  const unsigned tidx = threadIdx.x + threadIdx.y * blockDim.x;
  const unsigned ntpb = blockDim.x * blockDim.y;
  for (unsigned i = tidx; i < nitems; i += ntpb) shp[i] = 0;
  __syncthreads();
  const int32_t* idx_block = indices + block_start;
  int bnd = max(0, min(num_data_in_leaf - block_start, ndpt * (int)blockDim.y));
  int nit = (bnd + blockDim.y - 1) / blockDim.y;
  int rem = bnd % blockDim.y;
  int nit_this = rem == 0 ? nit : nit - (int)(threadIdx.y >= (unsigned)rem);
  int inner = threadIdx.y;
  const int col = threadIdx.x + pcs;
  if (threadIdx.x < (unsigned)ncol) {
    int32_t* shptr = shp + col_hist_off[col];
    for (int i = 0; i < nit_this; ++i) {
      int di = idx_block[inner];
      int32_t g = gh[di];
      uint32_t bin = (uint32_t)dptr[(size_t)di * row_stride + threadIdx.x];
      atomicAdd_block(shptr + bin, g);
      inner += blockDim.y;
    }
  }
  __syncthreads();
  long long* fh = (long long*)out_hist + phs;
  for (unsigned i = tidx; i < nitems; i += ntpb) {
    int32_t p = shp[i];
    int64_t v = ((int64_t)((int16_t)(p >> 16)) << 32) | (int64_t)(p & 0xffff);
    atomicAdd((unsigned long long*)(fh + i), (unsigned long long)v);
  }
}

// ---- LEAN variant: BINS, COLS, row_stride baked as compile-time constants;
// col_hist_off is col*BINS (uniform), nitems = COLS*BINS. No offset-array loads
// on the hot path; the compiler knows exact loop bounds and can size/unroll. ----
template<int SHIST, int BINS, int COLS>
__global__ void construct_lean(
    const int32_t* __restrict__ gh, const uint8_t* __restrict__ data,
    const int* __restrict__ part_col_off,
    const uint32_t* __restrict__ part_hist_off, const int32_t* __restrict__ indices,
    int num_data, int num_data_in_leaf, int64_t* __restrict__ out_hist) {
  __shared__ int16_t sh[SHIST];
  int32_t* shp = (int32_t*)sh;
  const int dim_y = gridDim.y * blockDim.y;
  const int ndpt = (num_data_in_leaf + dim_y - 1) / dim_y;
  const int block_start = (blockIdx.y * blockDim.y) * ndpt;
  if (block_start >= num_data_in_leaf) return;
  const int pcs = part_col_off[blockIdx.x];          // partition base column (still needed for data ptr)
  const uint8_t* dptr = data + (size_t)pcs * num_data;
  const uint32_t phs = part_hist_off[blockIdx.x];
  const unsigned tidx = threadIdx.x + threadIdx.y * blockDim.x;
  const unsigned ntpb = blockDim.x * blockDim.y;
  constexpr int NITEMS = COLS * BINS;               // baked
  #pragma unroll 4
  for (unsigned i = tidx; i < NITEMS; i += ntpb) shp[i] = 0;
  __syncthreads();
  const int32_t* idx_block = indices + block_start;
  int bnd = max(0, min(num_data_in_leaf - block_start, ndpt * (int)blockDim.y));
  int nit = (bnd + blockDim.y - 1) / blockDim.y;
  int rem = bnd % blockDim.y;
  int nit_this = rem == 0 ? nit : nit - (int)(threadIdx.y >= (unsigned)rem);
  int inner = threadIdx.y;
  if (threadIdx.x < COLS) {
    int32_t* shptr = shp + threadIdx.x * BINS;       // baked col_hist_off = col*BINS
    for (int i = 0; i < nit_this; ++i) {
      int di = idx_block[inner];
      int32_t g = gh[di];
      uint32_t bin = (uint32_t)dptr[(size_t)di * COLS + threadIdx.x];  // baked row_stride=COLS
      atomicAdd_block(shptr + bin, g);
      inner += blockDim.y;
    }
  }
  __syncthreads();
  long long* fh = (long long*)out_hist + phs;
  #pragma unroll 4
  for (unsigned i = tidx; i < NITEMS; i += ntpb) {
    int32_t p = shp[i];
    int64_t v = ((int64_t)((int16_t)(p >> 16)) << 32) | (int64_t)(p & 0xffff);
    atomicAdd((unsigned long long*)(fh + i), (unsigned long long)v);
  }
}

struct Shape { const char* name; int num_data, num_feature, bins, parts, cols_per_part; };

int main() {
  cudaDeviceProp prop; cudaGetDeviceProperties(&prop,0);
  printf("GPU %s SMs=%d\n", prop.name, prop.multiProcessorCount);
  // real bench root-level shapes (bins=255, single leaf = all rows)
  Shape shapes[] = {
    {"higgs",   1000000, 28,  219, 1, 28},   // 1 part (scaled rows for memory)
    {"covtype", 464809,     53,  115, 1, 53},   // 1 part
    {"epsilon", 200000,     1992,255, 83,24},   // 84 parts, 24 col/part
  };
  std::mt19937 rng(42);
  printf("%-9s %-6s %-5s | %-11s %-11s %-7s\n","shape","parts","cols","runtime_ms","lean_ms","speedup");
  for (auto& S : shapes) { printf("--- %s ---\n",S.name); fflush(stdout);
    int nd = S.num_data, ndl = nd;
    // layout
    std::vector<int> pco(S.parts+1); std::vector<uint32_t> cho(S.num_feature), pho(S.parts+1);
    int col=0; uint32_t hoff=0; pco[0]=0; pho[0]=0;
    for(int p=0;p<S.parts;p++){ uint32_t poff=hoff; int ncol=min(S.cols_per_part,S.num_feature-col);
      for(int i=0;i<ncol;i++){cho[col]=hoff-poff; hoff+=S.bins; col++;} pco[p+1]=col; pho[p+1]=hoff; }
    uint32_t total_bins=hoff;
    std::vector<int32_t> gh(nd), idx(ndl);
    std::uniform_int_distribution<int> bd(0,S.bins-1);
    for(int i=0;i<nd;i++){int16_t g=(int16_t)(bd(rng)%7-3);gh[i]=((int32_t)g<<16)|1;}
    for(int i=0;i<ndl;i++) idx[i]=i;
    std::vector<uint8_t> pdata((size_t)S.num_feature*nd);
    for(size_t i=0;i<pdata.size();i++) pdata[i]=(uint8_t)bd(rng);
    int32_t *d_gh,*d_idx; uint8_t* d_data; int* d_pco; uint32_t *d_cho,*d_pho; int64_t* d_hist;
    cudaMalloc(&d_gh,nd*4); cudaMalloc(&d_idx,ndl*4); cudaMalloc(&d_data,pdata.size());
    cudaMalloc(&d_pco,(S.parts+1)*4); cudaMalloc(&d_cho,S.num_feature*4);
    cudaMalloc(&d_pho,(S.parts+1)*4); cudaMalloc(&d_hist,(size_t)total_bins*8);
    cudaMemcpy(d_gh,gh.data(),nd*4,cudaMemcpyHostToDevice);
    cudaMemcpy(d_idx,idx.data(),ndl*4,cudaMemcpyHostToDevice);
    cudaMemcpy(d_data,pdata.data(),pdata.size(),cudaMemcpyHostToDevice);
    cudaMemcpy(d_pco,pco.data(),(S.parts+1)*4,cudaMemcpyHostToDevice);
    cudaMemcpy(d_cho,cho.data(),S.num_feature*4,cudaMemcpyHostToDevice);
    cudaMemcpy(d_pho,pho.data(),(S.parts+1)*4,cudaMemcpyHostToDevice);
    int bx=S.cols_per_part, by=504/bx, gy=160;
    dim3 grid(S.parts,gy,1), block(bx,by,1);
    auto runR=[&](int it){for(int k=0;k<it;k++){cudaMemset(d_hist,0,(size_t)total_bins*8);
      construct_runtime<12288><<<grid,block>>>(d_gh,d_data,d_pco,d_cho,d_pho,d_idx,nd,ndl,d_hist);}};
    auto timeit=[&](std::function<void(int)> f)->float{ f(3); cudaDeviceSynchronize();
      cudaEvent_t s,e;cudaEventCreate(&s);cudaEventCreate(&e);cudaEventRecord(s);f(20);cudaEventRecord(e);
      cudaEventSynchronize(e);float ms;cudaEventElapsedTime(&ms,s,e);return ms/20; };
    float tR=timeit(runR);
    float tL=0;
    // dispatch lean by shape (baked cols)
    if(S.cols_per_part==28) tL=timeit([&](int it){for(int k=0;k<it;k++){cudaMemset(d_hist,0,(size_t)total_bins*8);
      construct_lean<12288,219,28><<<grid,block>>>(d_gh,d_data,d_pco,d_pho,d_idx,nd,ndl,d_hist);}});
    else if(S.cols_per_part==53) tL=timeit([&](int it){for(int k=0;k<it;k++){cudaMemset(d_hist,0,(size_t)total_bins*8);
      construct_lean<12288,115,53><<<grid,block>>>(d_gh,d_data,d_pco,d_pho,d_idx,nd,ndl,d_hist);}});
    else if(S.cols_per_part==24) tL=timeit([&](int it){for(int k=0;k<it;k++){cudaMemset(d_hist,0,(size_t)total_bins*8);
      construct_lean<12288,255,24><<<grid,block>>>(d_gh,d_data,d_pco,d_pho,d_idx,nd,ndl,d_hist);}});
    printf("%-9s %-6d %-5d | %-11.4f %-11.4f %.3fx\n",S.name,S.parts,S.cols_per_part,tR,tL,tR/tL);
    cudaFree(d_gh);cudaFree(d_idx);cudaFree(d_data);cudaFree(d_pco);cudaFree(d_cho);cudaFree(d_pho);cudaFree(d_hist);
  }
  return 0;
}
