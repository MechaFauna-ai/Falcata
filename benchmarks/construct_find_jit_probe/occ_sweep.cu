#include <cstdio>
#include <cuda_runtime.h>
template<int N>
__global__ void k(int* out){ __shared__ short s[N]; int t=threadIdx.x+threadIdx.y*blockDim.x; for(int i=t;i<N;i+=blockDim.x*blockDim.y) s[i]=i; __syncthreads(); if(t==0) out[blockIdx.x]=s[0]; }
template<int N> void probe(const char* lbl){
  for(int bx:{24,28,53}){ int by=504/bx; int tpb=bx*by; int nb=0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb,k<N>,tpb,0);
    cudaFuncAttributes fa; cudaFuncGetAttributes(&fa,k<N>);
    printf("%s N=%d(%zuB) bx=%d tpb=%d smem=%zu -> %d blk/SM %d warps %.0f%%\n",lbl,N,(size_t)N*2,bx,tpb,fa.sharedSizeBytes,nb,nb*(tpb/32),100.0*nb*(tpb/32)/64);
  }
}
int main(){
  cudaDeviceProp p; cudaGetDeviceProperties(&p,0);
  printf("reservedSharedPerBlock=%zu maxThreadsPerSM=%d regsPerSM=%d\n",p.reservedSharedMemPerBlock,p.maxThreadsPerMultiProcessor,p.regsPerMultiprocessor);
  probe<12288>("cur24KB");  // current
  probe<10240>("20KB");
  probe<8192>("16KB");
  probe<6144>("12KB");
  return 0;
}
