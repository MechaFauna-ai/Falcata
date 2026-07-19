// Standalone A/B of the quantized discretized construct kernel: partition count
// vs kernel time, on the epsilon shape. Replicates the accumulation body of
// CUDAConstructDiscretizedHistogramDenseInner (int16 packed shared hist, atomicAdd,
// 32-bit flush). Compares baseline (SP: 24 cols/part, 84 parts) to bigshmem
// (48 cols/part, 42 parts) and 96-col/128KB dynamic-shmem (21 parts).
#include <cstdio>
#include <cstdint>
#include <vector>
#include <random>
#include <cuda_runtime.h>

// Static-shmem variant (baseline SP-like and 48KB). SHIST = int16 slots.
template<int SHIST>
__global__ void construct_static(
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
  const uint32_t phe = part_hist_off[blockIdx.x + 1];
  const uint32_t nitems = phe - phs;
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

// Dynamic-shmem variant (for >48KB, e.g. 96 cols).
__global__ void construct_dyn(
    const int32_t* __restrict__ gh, const uint8_t* __restrict__ data,
    const int* __restrict__ part_col_off, const uint32_t* __restrict__ col_hist_off,
    const uint32_t* __restrict__ part_hist_off, const int32_t* __restrict__ indices,
    int num_data, int num_data_in_leaf, int64_t* __restrict__ out_hist) {
  extern __shared__ int16_t sh[];
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
  const uint32_t phe = part_hist_off[blockIdx.x + 1];
  const uint32_t nitems = phe - phs;
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

struct Layout {
  int num_feature, bins, cols_per_part, parts, block_x, block_y;
  std::vector<int> part_col_off; std::vector<uint32_t> col_hist_off, part_hist_off;
  int total_bins;
};

Layout make_layout(int num_feature, int bins, int cols_per_part) {
  Layout L; L.num_feature=num_feature; L.bins=bins; L.cols_per_part=cols_per_part;
  L.parts = (num_feature + cols_per_part - 1) / cols_per_part;
  L.part_col_off.resize(L.parts+1); L.part_hist_off.resize(L.parts+1);
  L.col_hist_off.resize(num_feature);
  int col=0; uint32_t hoff=0;
  L.part_col_off[0]=0; L.part_hist_off[0]=0;
  for (int p=0;p<L.parts;p++){
    int c0=col; uint32_t poff=hoff;
    int ncol = min(cols_per_part, num_feature - c0);
    for (int i=0;i<ncol;i++){ L.col_hist_off[col] = hoff - poff; hoff += bins; col++; }
    L.part_col_off[p+1]=col; L.part_hist_off[p+1]=hoff;
  }
  L.total_bins = hoff;
  L.block_x = cols_per_part;
  L.block_y = 504 / cols_per_part;
  return L;
}

int main(int argc, char** argv) {
  int num_data = 400000, num_feature = 2000, bins = 255;
  int leaf_frac_pct = 100;  // rows in the leaf being built
  cudaDeviceProp prop; cudaGetDeviceProperties(&prop,0);
  int num_data_in_leaf = (int)((long)num_data * leaf_frac_pct / 100);
  std::mt19937 rng(42);
  std::vector<uint8_t> h_data((size_t)num_feature*num_data);
  std::uniform_int_distribution<int> bd(0,bins-1);
  // data is stored partitioned; but the per-partition slice layout differs by
  // cols_per_part. For timing we only need a byte buffer of the right size and a
  // valid bin range; build it per-layout below.
  std::vector<int32_t> h_gh(num_data); std::vector<int32_t> h_idx(num_data_in_leaf);
  for (int i=0;i<num_data;i++){ int16_t g=(int16_t)(bd(rng)%7-3); int16_t h=1; h_gh[i]=((int32_t)g<<16)|((int32_t)h&0xffff);}
  for (int i=0;i<num_data_in_leaf;i++) h_idx[i]=i;

  int32_t *d_gh,*d_idx; cudaMalloc(&d_gh,num_data*4); cudaMalloc(&d_idx,num_data_in_leaf*4);
  cudaMemcpy(d_gh,h_gh.data(),num_data*4,cudaMemcpyHostToDevice);
  cudaMemcpy(d_idx,h_idx.data(),num_data_in_leaf*4,cudaMemcpyHostToDevice);

  printf("epsilon shape: num_data=%d num_feature=%d bins=%d leaf_rows=%d, SM=%d\n",
         num_data,num_feature,bins,num_data_in_leaf,prop.multiProcessorCount);
  printf("%-8s %-6s %-6s %-8s %-9s %-10s %-8s\n","cols","parts","blkx","blky","shmemKB","kernel","time_ms");

  struct Cfg { const char* name; int cols; bool dyn; };
  // 128-col/128KB is omitted: it exceeds the sm_120 per-block dynamic-shared
  // ceiling (cudaFuncSetAttribute rejects it), and the 96KB point already
  // establishes the monotone slowdown. Query the ceiling so the probe adapts to
  // other archs without ever issuing a rejected attribute request.
  int max_dyn_shared = 0;
  cudaDeviceGetAttribute(&max_dyn_shared, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0);
  Cfg cfgs[] = {
    {"SP(base)", 24, false},   // 84 parts, 24KB
    {"BIG48", 48, false},      // 42 parts, 48KB static
    {"DYN96", 96, true},       // 21 parts, 96KB dynamic
  };
  for (auto& cf : cfgs) {
    Layout L = make_layout(num_feature, bins, cf.cols);
    // build partitioned data buffer for this layout
    std::vector<uint8_t> pdata((size_t)num_feature*num_data);
    // fill each partition slice row-major
    for (int p=0;p<L.parts;p++){
      int c0=L.part_col_off[p], c1=L.part_col_off[p+1]; int ncol=c1-c0;
      uint8_t* base = pdata.data() + (size_t)c0*num_data;
      for (int r=0;r<num_data;r++) for(int c=0;c<ncol;c++) base[(size_t)r*ncol+c]=(uint8_t)bd(rng);
    }
    uint8_t* d_data; cudaMalloc(&d_data, pdata.size()); cudaMemcpy(d_data,pdata.data(),pdata.size(),cudaMemcpyHostToDevice);
    int* d_pco; cudaMalloc(&d_pco,(L.parts+1)*4); cudaMemcpy(d_pco,L.part_col_off.data(),(L.parts+1)*4,cudaMemcpyHostToDevice);
    uint32_t* d_cho; cudaMalloc(&d_cho,num_feature*4); cudaMemcpy(d_cho,L.col_hist_off.data(),num_feature*4,cudaMemcpyHostToDevice);
    uint32_t* d_pho; cudaMalloc(&d_pho,(L.parts+1)*4); cudaMemcpy(d_pho,L.part_hist_off.data(),(L.parts+1)*4,cudaMemcpyHostToDevice);
    int64_t* d_hist; cudaMalloc(&d_hist,(size_t)L.total_bins*8);
    int block_y = L.block_y;
    int grid_y = 160;  // like epsilon root level
    dim3 grid(L.parts, grid_y, 1), block(cf.cols, block_y, 1);
    size_t dynb = (size_t)2 * (bins * cf.cols) * 2; // int16 slots * 2 bytes; *2 pad safety
    // exact needed: max partition bins * 2 (int16 per int32 slot) *2 bytes
    size_t need = (size_t)(bins * cf.cols) * 4; // bins*cols int32 = *4 bytes = int16*2
    cudaError_t attr = cudaSuccess;
    auto run = [&](int iters){
      for(int it=0;it<iters;it++){
        cudaMemset(d_hist,0,(size_t)L.total_bins*8);
        if (cf.dyn) construct_dyn<<<grid,block,need>>>(d_gh,d_data,d_pco,d_cho,d_pho,d_idx,num_data,num_data_in_leaf,d_hist);
        else if (cf.cols<=24) construct_static<12288><<<grid,block>>>(d_gh,d_data,d_pco,d_cho,d_pho,d_idx,num_data,num_data_in_leaf,d_hist);
        else construct_static<24576><<<grid,block>>>(d_gh,d_data,d_pco,d_cho,d_pho,d_idx,num_data,num_data_in_leaf,d_hist);
      }
    };
    double shkb = cf.dyn ? need/1024.0 : (cf.cols<=24?24.0:48.0);
    if (cf.dyn) {
      // Opt-in for >48KB dynamic shared, but never issue a request the device
      // will reject (keeps the reproducer sanitizer-clean): skip if it exceeds
      // the queried per-block dynamic-shared ceiling.
      if ((int)need > max_dyn_shared) {
        printf("%-8s %-6d %-6d %-8d %-9.1f %-10s   skipped (needs %zuKB > %dKB dynamic-shared ceiling)\n",
               cf.name, L.parts, cf.cols, block_y, shkb, "dyn", need/1024, max_dyn_shared/1024);
        cudaFree(d_data);cudaFree(d_pco);cudaFree(d_cho);cudaFree(d_pho);cudaFree(d_hist);
        continue;
      }
      attr = cudaFuncSetAttribute(construct_dyn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)need);
    }
    run(3); cudaError_t e = cudaDeviceSynchronize();
    cudaEvent_t s,ev; cudaEventCreate(&s); cudaEventCreate(&ev);
    cudaEventRecord(s); run(20); cudaEventRecord(ev); cudaEventSynchronize(ev);
    float ms=0; cudaEventElapsedTime(&ms,s,ev); ms/=20;
    printf("%-8s %-6d %-6d %-8d %-9.1f %-10s %.3f  %s\n",
           cf.name, L.parts, cf.cols, block_y, shkb, cf.dyn?"dyn":"static", ms,
           e!=cudaSuccess?cudaGetErrorString(e):"");
    cudaFree(d_data);cudaFree(d_pco);cudaFree(d_cho);cudaFree(d_pho);cudaFree(d_hist);
  }
  return 0;
}
