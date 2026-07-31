// Tensor-core histogram construction feasibility benchmark.
//
// Question: can the GBDT histogram pass H[col,bin] += (grad,hess) over rows
// be faster as an int8 one-hot MMA (tensor cores, no atomics) than as the
// shared-memory-atomic kernel falcata ships today, on numerai-like data
// (<=8 bins per column, quantized int8 gradients)?
//
// Kernel A (baseline): representative of falcata's discretized inner loop.
//   Row-major packed 4-bit bins, thread-per-column, packed int32 grad|hess
//   shared atomics, block flush to global.
// Kernel B (tensor cores): WMMA int8 16x16x16. A warp owns TWO columns
//   (m dim = 2 cols x 8 bins). Per 16-row step the A fragment is the one-hot
//   of the two columns' bins (built via a 256-entry byte -> 16-byte LUT) and
//   the B fragment carries (grad, hess) in n columns 0/1. The accumulator
//   fragment holds the running 2x8x2 histogram in REGISTERS across the whole
//   row strip; one global-atomic flush per warp at the end.
//   Reads column-major unpacked bins (the landed colmajor_fill lever already
//   pays for a column-major copy in real training).
//
// Both kernels produce EXACT integer histograms -> verified against CPU.

#include <cuda.h>
#include <mma.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

using namespace nvcuda;

#define CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
  printf("CUDA error %s at line %d\n", cudaGetErrorString(e), __LINE__); exit(1); } } while (0)

static constexpr int kRows = 2 * 1024 * 1024;   // 2M rows
static constexpr int kCols = 356;               // numerai compact view (ff=0.1)
static constexpr int kBins = 8;                 // <=8-bin columns (span-7 + pad)
static constexpr int kColsPerByte = 2;          // 4-bit packing (kernel A input)
static constexpr int kRowBytes = kCols / kColsPerByte;

// ---------------------------------------------------------------- kernel A --
// Baseline: mirrors the shipped discretized construct structure: one thread
// per column, rows strided across blocks, shared int32 hist with packed
// grad|hess atomics (hess in low 16, grad biased in high 16 -- the packed-int
// trick the real kernel uses at <=8 bins), block-level flush via global atomics.
__global__ void HistAtomic(const uint8_t* __restrict__ packed,  // row-major
                           const int8_t* __restrict__ grad,
                           const int8_t* __restrict__ hess,
                           int rows, int* __restrict__ ghist /* cols*bins*2 */) {
  __shared__ int shist[kCols * kBins];  // packed grad|hess per (col,bin)
  for (int i = threadIdx.x; i < kCols * kBins; i += blockDim.x) shist[i] = 0;
  __syncthreads();
  const int col = threadIdx.x;
  const bool active = col < kCols;
  const int byte_idx = col >> 1, shift = (col & 1) << 2;
  const int row_begin = blockIdx.x * ((rows + gridDim.x - 1) / gridDim.x);
  const int row_end = min(rows, row_begin + ((rows + gridDim.x - 1) / gridDim.x));
  if (active) {
    for (int r = row_begin; r < row_end; ++r) {
      const int b = (packed[(size_t)r * kRowBytes + byte_idx] >> shift) & 0xF;
      const int g = grad[r], h = hess[r];
      atomicAdd(&shist[col * kBins + b], (g << 16) | (h & 0xFFFF));
    }
  }
  __syncthreads();
  for (int i = threadIdx.x; i < kCols * kBins; i += blockDim.x) {
    const int v = shist[i];
    if (v != 0) {
      atomicAdd(&ghist[i * 2 + 0], v >> 16);            // grad
      atomicAdd(&ghist[i * 2 + 1], (short)(v & 0xFFFF));  // hess (sign-safe: hess>=0, no borrow at these magnitudes)
    }
  }
}

// ---------------------------------------------------------------- kernel B --
// One warp <-> two columns x one row strip. LUT: byte (two 4-bit bins, but
// here one column's bin per byte laid col-major unpacked) -> handled per
// column; A fragment is built 16 rows x 16 slots.
//
// A (row-major m16 x k16): A[slot][step_row] = 1 iff bin(row, col(slot)) == bin(slot)
//   slot = c * kBins + b for c in {0,1}
// B (col-major k16 x n16): B[step_row][0] = grad, [1] = hess, rest 0
// D += A x B accumulates [slot][0] = sum grad, [slot][1] = sum hess.
__global__ void HistTC(const uint8_t* __restrict__ colmajor,  // [col][row] bins
                       const int8_t* __restrict__ grad,
                       const int8_t* __restrict__ hess,
                       int rows, int rows_per_strip, int* __restrict__ ghist) {
  const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  const int lane = threadIdx.x & 31;
  const int n_col_pairs = kCols / 2;
  const int col_pair = warp_id % n_col_pairs;
  const int strip = warp_id / n_col_pairs;
  const int c0 = col_pair * 2;
  const int row_begin = strip * rows_per_strip;
  const int row_end = min(rows, row_begin + rows_per_strip);
  if (row_begin >= rows) return;

  // shared staging for this warp's fragments (per-warp slices)
  extern __shared__ uint8_t smem[];
  const int warp_in_block = threadIdx.x >> 5;
  int8_t* aStage = reinterpret_cast<int8_t*>(smem) + warp_in_block * (256 + 256);
  int8_t* bStage = aStage + 256;

  wmma::fragment<wmma::accumulator, 16, 16, 16, int> dFrag;
  wmma::fill_fragment(dFrag, 0);

  for (int r0 = row_begin; r0 < row_end; r0 += 16) {
    // build A: 16 slots x 16 rows (row-major, ld=16). Two lanes per data row:
    // lane 0-15 -> column c0 one-hot rows, lane 16-31 -> column c0+1.
    {
      const int my_row = r0 + (lane & 15);
      const int my_col = c0 + (lane >> 4);
      const int b = (my_row < row_end) ? colmajor[(size_t)my_col * rows + my_row] : 0xF;
      // zero the 16-byte A column for this (row, colhalf), then set the hot slot
      // A is row-major [slot][row]: entries A[slot][lane&15] for slot in colhalf block
      #pragma unroll
      for (int s = 0; s < kBins; ++s) {
        aStage[((lane >> 4) * kBins + s) * 16 + (lane & 15)] = (b == s) ? 1 : 0;
      }
    }
    // build B: 16 rows x 16 n, col-major (ld=16): column 0 grad, 1 hess
    {
      if (lane < 16) {
        const int my_row = r0 + lane;
        const bool in = my_row < row_end;
        bStage[0 * 16 + lane] = in ? grad[my_row] : 0;
        bStage[1 * 16 + lane] = in ? hess[my_row] : 0;
      } else {
        // zero the remaining 14 n-columns (each lane 16..31 zeroes one+)
        for (int n = lane - 16 + 2; n < 16; n += 16) {
          #pragma unroll
          for (int r = 0; r < 16; ++r) bStage[n * 16 + r] = 0;
        }
      }
    }
    __syncwarp();
    wmma::fragment<wmma::matrix_a, 16, 16, 16, signed char, wmma::row_major> aFrag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, signed char, wmma::col_major> bFrag;
    wmma::load_matrix_sync(aFrag, aStage, 16);
    wmma::load_matrix_sync(bFrag, bStage, 16);
    wmma::mma_sync(dFrag, aFrag, bFrag, dFrag);
    __syncwarp();
  }

  // flush: D[slot][0]=grad sum, D[slot][1]=hess sum for slot=0..15
  __shared__ int dOut[8][16][16];  // per warp-in-block staging (8 warps max)
  wmma::store_matrix_sync(&dOut[warp_in_block][0][0], dFrag, 16, wmma::mem_row_major);
  __syncwarp();
  if (lane < 16) {
    const int slot_col = c0 + (lane >> 3);
    const int slot_bin = lane & 7;
    atomicAdd(&ghist[(slot_col * kBins + slot_bin) * 2 + 0], dOut[warp_in_block][lane][0]);
    atomicAdd(&ghist[(slot_col * kBins + slot_bin) * 2 + 1], dOut[warp_in_block][lane][1]);
  }
}


// ---------------------------------------------------------------- kernel B2 --
// Register-direct int8 MMA (PTX mma.m16n8k32): no shared staging at all.
// Warp owns 2 columns x row strip. Per 32-row k-tile each lane builds its
// A fragment (one-hot) with one uchar4 load + __vcmpeq4 per register, and its
// B fragment (grad/hess) with a single conditional uchar4 load. The running
// 2x8x2 histogram lives in the D registers for the whole strip.
__global__ void HistTC2(const uint8_t* __restrict__ colmajor,
                        const int8_t* __restrict__ grad,
                        const int8_t* __restrict__ hess,
                        int rows, int rows_per_strip, int* __restrict__ ghist) {
  const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  const int lane = threadIdx.x & 31;
  const int n_col_pairs = kCols / 2;
  const int col_pair = warp_id % n_col_pairs;
  const int strip = warp_id / n_col_pairs;
  const int c0 = col_pair * 2;
  int row_begin = strip * rows_per_strip;
  const int row_end_full = min(rows, row_begin + rows_per_strip);
  if (row_begin >= rows) return;
  // per-lane constants
  const int g = lane >> 2;             // group 0..7 = bin slot
  const int tg = lane & 3;             // thread-in-group
  const unsigned bin_pat = 0x01010101u * (unsigned)g;
  const uint8_t* col0p = colmajor + (size_t)c0 * rows;
  const uint8_t* col1p = colmajor + (size_t)(c0 + 1) * rows;

  int d0 = 0, d1 = 0, d2 = 0, d3 = 0;
  // full 32-row tiles only in the fast loop; ragged tail handled scalar below
  const int full_end = row_begin + ((row_end_full - row_begin) & ~31);
  for (int r0 = row_begin; r0 < full_end; r0 += 32) {
    const int rlo = r0 + tg * 4, rhi = rlo + 16;
    const unsigned b0lo = *reinterpret_cast<const unsigned*>(col0p + rlo);
    const unsigned b1lo = *reinterpret_cast<const unsigned*>(col1p + rlo);
    const unsigned b0hi = *reinterpret_cast<const unsigned*>(col0p + rhi);
    const unsigned b1hi = *reinterpret_cast<const unsigned*>(col1p + rhi);
    const unsigned a0 = __vcmpeq4(b0lo, bin_pat) & 0x01010101u;  // col0, rows lo
    const unsigned a1 = __vcmpeq4(b1lo, bin_pat) & 0x01010101u;  // col1, rows lo
    const unsigned a2 = __vcmpeq4(b0hi, bin_pat) & 0x01010101u;
    const unsigned a3 = __vcmpeq4(b1hi, bin_pat) & 0x01010101u;
    unsigned bb0 = 0, bb1 = 0;
    if (g == 0) {
      bb0 = *reinterpret_cast<const unsigned*>(grad + rlo);
      bb1 = *reinterpret_cast<const unsigned*>(grad + rhi);
    } else if (g == 1) {
      bb0 = *reinterpret_cast<const unsigned*>(hess + rlo);
      bb1 = *reinterpret_cast<const unsigned*>(hess + rhi);
    }
    asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+r"(d0), "+r"(d1), "+r"(d2), "+r"(d3)
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(bb0), "r"(bb1));
  }
  // ragged tail: plain scalar adds into the same D layout via global atomics later
  int tail_g[2] = {0, 0}, tail_h[2] = {0, 0};  // per (col, my bin g)
  for (int r = full_end + lane; r < row_end_full; r += 32) {
    // handled cooperatively below via atomics; rare path (only last strip)
  }
  // D layout: d0 = [row g][cols tg*2, tg*2+1]; d2/d3 = row g+8.
  // rows of D = slots: slot g -> (col0, bin g); slot g+8 -> (col1, bin g).
  // cols of D = n: 0 = grad, 1 = hess.
  const int n0 = tg * 2, n1 = tg * 2 + 1;
  if (n0 == 0) {  // tg==0 lanes hold n=0(grad),1(hess)
    atomicAdd(&ghist[(c0 * kBins + g) * 2 + 0], d0);
    atomicAdd(&ghist[(c0 * kBins + g) * 2 + 1], d1);
    atomicAdd(&ghist[((c0 + 1) * kBins + g) * 2 + 0], d2);
    atomicAdd(&ghist[((c0 + 1) * kBins + g) * 2 + 1], d3);
  }
  // scalar tail (few rows, once per column pair: only the strip containing the end)
  if (full_end < row_end_full && lane == 0) {
    for (int r = full_end; r < row_end_full; ++r) {
      const int b0 = col0p[r], b1 = col1p[r];
      atomicAdd(&ghist[(c0 * kBins + b0) * 2 + 0], (int)grad[r]);
      atomicAdd(&ghist[(c0 * kBins + b0) * 2 + 1], (int)hess[r]);
      atomicAdd(&ghist[((c0 + 1) * kBins + b1) * 2 + 0], (int)grad[r]);
      atomicAdd(&ghist[((c0 + 1) * kBins + b1) * 2 + 1], (int)hess[r]);
    }
  }
}


// ---------------------------------------------------------------- kernel B3 --
// tc2 + two fixes: (1) reads the SAME 4-bit packed row-major... no -- packed
// COLUMN-PAIR-major: byte = (col1<<4)|col0 for a row, columns paired, laid
// col-pair-major so one uchar4 load covers 4 rows x 2 cols; (2) one warp owns
// FOUR column pairs, so B fragments (grad/hess) and their loads amortize 4x.
__global__ void HistTC3(const uint8_t* __restrict__ pairmajor,  // [colpair][row] packed
                        const int8_t* __restrict__ grad,
                        const int8_t* __restrict__ hess,
                        int rows, int rows_per_strip, int* __restrict__ ghist) {
  const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  const int lane = threadIdx.x & 31;
  const int n_quads = (kCols / 2 + 3) / 4;    // 4 col-pairs per warp (last quad may be ragged)
  const int quad = warp_id % n_quads;
  const int strip = warp_id / n_quads;
  int row_begin = strip * rows_per_strip;
  const int row_end_full = min(rows, row_begin + rows_per_strip);
  if (row_begin >= rows) return;
  const int g = lane >> 2;
  const int tg = lane & 3;
  const unsigned bin_lo = 0x01010101u * (unsigned)g;          // col0 nibble target
  const unsigned bin_hi = 0x10101010u * (unsigned)g;          // col1 nibble target
  const int n_live = min(4, kCols / 2 - quad * 4);  // live col-pairs in this quad
  const uint8_t* pp[4];
  #pragma unroll
  for (int q = 0; q < 4; ++q)
    pp[q] = pairmajor + (size_t)(quad * 4 + min(q, n_live - 1)) * rows;

  int d[4][4] = {};
  const int full_end = row_begin + ((row_end_full - row_begin) & ~31);
  for (int r0 = row_begin; r0 < full_end; r0 += 32) {
    const int rlo = r0 + tg * 4, rhi = rlo + 16;
    unsigned bb0 = 0, bb1 = 0;
    if (g == 0) {
      bb0 = *reinterpret_cast<const unsigned*>(grad + rlo);
      bb1 = *reinterpret_cast<const unsigned*>(grad + rhi);
    } else if (g == 1) {
      bb0 = *reinterpret_cast<const unsigned*>(hess + rlo);
      bb1 = *reinterpret_cast<const unsigned*>(hess + rhi);
    }
    #pragma unroll
    for (int q = 0; q < 4; ++q) {
      if (q >= n_live) break;
      const unsigned plo = *reinterpret_cast<const unsigned*>(pp[q] + rlo);
      const unsigned phi = *reinterpret_cast<const unsigned*>(pp[q] + rhi);
      const unsigned a0 = __vcmpeq4(plo & 0x0F0F0F0Fu, bin_lo) & 0x01010101u;
      const unsigned a1 = __vcmpeq4(plo & 0xF0F0F0F0u, bin_hi) & 0x01010101u;
      const unsigned a2 = __vcmpeq4(phi & 0x0F0F0F0Fu, bin_lo) & 0x01010101u;
      const unsigned a3 = __vcmpeq4(phi & 0xF0F0F0F0u, bin_hi) & 0x01010101u;
      asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+r"(d[q][0]), "+r"(d[q][1]), "+r"(d[q][2]), "+r"(d[q][3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(bb0), "r"(bb1));
    }
  }
  if (tg == 0) {
    #pragma unroll
    for (int q = 0; q < 4; ++q) {
      if (q >= n_live) break;
      const int c0 = (quad * 4 + q) * 2;
      atomicAdd(&ghist[(c0 * kBins + g) * 2 + 0], d[q][0]);
      atomicAdd(&ghist[(c0 * kBins + g) * 2 + 1], d[q][1]);
      atomicAdd(&ghist[((c0 + 1) * kBins + g) * 2 + 0], d[q][2]);
      atomicAdd(&ghist[((c0 + 1) * kBins + g) * 2 + 1], d[q][3]);
    }
  }
  if (full_end < row_end_full && lane == 0) {
    for (int r = full_end; r < row_end_full; ++r) {
      for (int q = 0; q < n_live; ++q) {
        const int c0 = (quad * 4 + q) * 2;
        const int b0 = pp[q][r] & 0xF, b1 = pp[q][r] >> 4;
        atomicAdd(&ghist[(c0 * kBins + b0) * 2 + 0], (int)grad[r]);
        atomicAdd(&ghist[(c0 * kBins + b0) * 2 + 1], (int)hess[r]);
        atomicAdd(&ghist[((c0 + 1) * kBins + b1) * 2 + 0], (int)grad[r]);
        atomicAdd(&ghist[((c0 + 1) * kBins + b1) * 2 + 1], (int)hess[r]);
      }
    }
  }
}

// ------------------------------------------------------------------- driver --
int main(int argc, char** argv) {
  const int iters = argc > 1 ? atoi(argv[1]) : 30;
  printf("rows=%d cols=%d bins=%d iters=%d\n", kRows, kCols, kBins, iters);

  // synthetic data: numerai-ish bin distribution (skewed), int8 quant grads
  std::vector<uint8_t> h_packed((size_t)kRows * kRowBytes);
  std::vector<uint8_t> h_colmajor((size_t)kCols * kRows);
  std::vector<uint8_t> h_pairmajor((size_t)(kCols / 2) * kRows);
  std::vector<int8_t> h_grad(kRows), h_hess(kRows);
  srand(42);
  for (int r = 0; r < kRows; ++r) {
    for (int c = 0; c < kCols; c += 2) {
      const int b0 = rand() % kBins, b1 = rand() % kBins;
      h_packed[(size_t)r * kRowBytes + c / 2] = (uint8_t)(b0 | (b1 << 4));
      h_colmajor[(size_t)c * kRows + r] = (uint8_t)b0;
      h_colmajor[(size_t)(c + 1) * kRows + r] = (uint8_t)b1;
      h_pairmajor[(size_t)(c / 2) * kRows + r] = (uint8_t)(b0 | (b1 << 4));
    }
    h_grad[r] = (int8_t)((rand() % 5) - 2);   // [-2,2]
    h_hess[r] = (int8_t)(1 + rand() % 3);     // [1,3]
  }

  // CPU reference
  std::vector<long long> ref((size_t)kCols * kBins * 2, 0);
  for (int r = 0; r < kRows; ++r) {
    for (int c = 0; c < kCols; ++c) {
      const int b = h_colmajor[(size_t)c * kRows + r];
      ref[((size_t)c * kBins + b) * 2 + 0] += h_grad[r];
      ref[((size_t)c * kBins + b) * 2 + 1] += h_hess[r];
    }
  }

  uint8_t *d_packed, *d_colmajor, *d_pairmajor; int8_t *d_grad, *d_hess; int* d_hist;
  CHECK(cudaMalloc(&d_packed, h_packed.size()));
  CHECK(cudaMalloc(&d_colmajor, h_colmajor.size()));
  CHECK(cudaMalloc(&d_grad, kRows)); CHECK(cudaMalloc(&d_hess, kRows));
  CHECK(cudaMalloc(&d_hist, kCols * kBins * 2 * sizeof(int)));
  CHECK(cudaMemcpy(d_packed, h_packed.data(), h_packed.size(), cudaMemcpyHostToDevice));
  CHECK(cudaMalloc(&d_pairmajor, h_pairmajor.size()));
  CHECK(cudaMemcpy(d_colmajor, h_colmajor.data(), h_colmajor.size(), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(d_pairmajor, h_pairmajor.data(), h_pairmajor.size(), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(d_grad, h_grad.data(), kRows, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(d_hess, h_hess.data(), kRows, cudaMemcpyHostToDevice));

  auto verify = [&](const char* name) {
    std::vector<int> got(kCols * kBins * 2);
    CHECK(cudaMemcpy(got.data(), d_hist, got.size() * sizeof(int), cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < got.size(); ++i) {
      if ((long long)got[i] != ref[i]) {
        printf("%s MISMATCH at %zu: got %d want %lld\n", name, i, got[i], ref[i]);
        return false;
      }
    }
    printf("%s verify OK\n", name);
    return true;
  };
  auto bench = [&](const char* name, auto launch) {
    CHECK(cudaMemset(d_hist, 0, kCols * kBins * 2 * sizeof(int)));
    launch(); CHECK(cudaDeviceSynchronize());
    if (!verify(name)) return;
    cudaEvent_t t0, t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0);
    for (int i = 0; i < iters; ++i) {
      CHECK(cudaMemset(d_hist, 0, kCols * kBins * 2 * sizeof(int)));
      launch();
    }
    cudaEventRecord(t1); CHECK(cudaDeviceSynchronize());
    float ms; cudaEventElapsedTime(&ms, t0, t1);
    printf("%-12s %8.3f ms/pass  (%.1f Grows/s cell-updates: %.1f G/s)\n",
           name, ms / iters, kRows / (ms / iters) / 1e6,
           (double)kRows * kCols / (ms / iters) / 1e6);
  };

  bench("atomic", [&]() { HistAtomic<<<680, 384>>>(d_packed, d_grad, d_hess, kRows, d_hist); });

  // TC: strips sized so total warps ~ fill the GPU; 8 warps/block
  for (int strips : {16, 32, 64}) {
    const int rows_per_strip = (kRows + strips - 1) / strips;
    const int warps = (kCols / 2) * strips;
    const int blocks = (warps + 7) / 8;
    char label[64]; snprintf(label, sizeof label, "tc-s%d", strips);
    bench(label, [&]() {
      HistTC<<<blocks, 256, 8 * 512>>>(d_colmajor, d_grad, d_hess, kRows, rows_per_strip, d_hist);
    });
  }
  for (int strips : {16, 32, 64, 128}) {
    const int rows_per_strip = ((kRows + strips - 1) / strips + 31) & ~31;
    const int warps = (kCols / 2) * strips;
    const int blocks = (warps + 7) / 8;
    char label[64]; snprintf(label, sizeof label, "tc2-s%d", strips);
    bench(label, [&]() {
      HistTC2<<<blocks, 256>>>(d_colmajor, d_grad, d_hess, kRows, rows_per_strip, d_hist);
    });
  }
  for (int strips : {32, 64, 128, 256}) {
    const int rows_per_strip = ((kRows + strips - 1) / strips + 31) & ~31;
    const int warps = ((kCols / 2 + 3) / 4) * strips;
    const int blocks = (warps + 7) / 8;
    char label[64]; snprintf(label, sizeof label, "tc3-s%d", strips);
    bench(label, [&]() {
      HistTC3<<<blocks, 256>>>(d_pairmajor, d_grad, d_hess, kRows, rows_per_strip, d_hist);
    });
  }
  return 0;
}

// ---------------------------------------------------------------------------
// RESULTS 2026-07-31, RTX 5090, CUDA 12.9, sm_120, 2M rows x 356 cols x 8 bins
// (all variants verified exactly against the CPU reference):
//   atomic (baseline, thread-per-col shared atomics)   0.62-0.67 ms/pass  ~1.12 T upd/s
//   tc1  WMMA m16n16k16, shared-staged fragments       2.34 ms  (0.32 T)
//   tc2  PTX mma.m16n8k32, register-direct fragments   1.55-1.67 ms  (0.45 T)
//   tc3  tc2 + packed dual-col loads + 4 pairs/warp    1.28 ms  (0.58 T)
// mma issue was ~4% of TC peak in tc3 -> bound by the ~10 fragment-build
// instructions per mma, irreducible in the one-hot formulation.
// Verdict: dead end on consumer silicon -- see docs/perf-dead-ends.md.
// ---------------------------------------------------------------------------
