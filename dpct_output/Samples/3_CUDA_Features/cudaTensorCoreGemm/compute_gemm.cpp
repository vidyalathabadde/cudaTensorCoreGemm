#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <assert.h>
#include <stdio.h>

// helper functions and utilities to work with CUDA
#include <helper_cuda.h>
#include <helper_functions.h>

// Externally configurable parameters.

#ifndef CPU_DEBUG
// Set this to 1 to verify the correctness of the GPU-computed matrix.
#define CPU_DEBUG 1
#endif

#ifndef SHARED_MEMORY_LIMIT_64K
// Set this to 0 to use more than 64 Kb of shared memory to cache data, to
// improve the performance of the computations on GPU.
// Note that you need a GPU that can have more than 64 Kb of shared memory
// per multiprocessor.
#define SHARED_MEMORY_LIMIT_64K 1
#endif

// GPU configuration.

#define WARP_SIZE 16

// MMA matrix tile dimensions.

#define M 8
#define N 16
#define K 16

#define WMMA_M 8
#define WMMA_N 16
#define WMMA_K 16

// GEMM configuration.

#define M_TILES 2
#define N_TILES 1
#define K_TILES 1

#define M_GLOBAL (M * M_TILES)
#define N_GLOBAL (N * N_TILES)
#define K_GLOBAL (K * K_TILES)

#define C_LAYOUT sycl::ext::oneapi::experimental::matrix::layout::row_major

// Implementation constants.

#define WARPS_PER_BLOCK 16
#define THREADS_PER_BLOCK (WARP_SIZE * WARPS_PER_BLOCK)

#if SHARED_MEMORY_LIMIT_64K
// With only 64 Kb shared memory available, we can fit two 8-tile chunks of
// the A and B matrix data, that are 16 * 16 * 8 * 8 * 2 = 32 Kb each
// (i.e. two 8x8 arrays of tiles of 16x16 half-typed elements per CTA).
// But we cannot account the 8 Kb total skew overhead, without which the
// performance would be severely impacted. So we choose to reduce the chunk size
// in half, i.e. the amount of A and B matrix data we cache in shared memory.
// Accordingly, this doubles the number of outer iterations across the global K
// dimension, which only slightly impacts the performance.
#define CHUNK_K 4
#else
#define CHUNK_K 8
#endif

#define CHUNK_LINE_BYTES (CHUNK_K * K * sizeof(sycl::half))
#define WARP_COPY_BYTES (WARP_SIZE * sizeof(sycl::int4))
#define CHUNK_COPY_LINES_PER_WARP (WARP_COPY_BYTES / CHUNK_LINE_BYTES)
#define CHUNK_COPY_LINE_LANES (WARP_SIZE / CHUNK_COPY_LINES_PER_WARP)

#define BLOCK_ROW_WARPS 2
#define BLOCK_COL_WARPS 4

#define WARP_ROW_TILES 4
#define WARP_COL_TILES 2

#define BLOCK_ROW_TILES (WARP_ROW_TILES * BLOCK_ROW_WARPS)
#define BLOCK_COL_TILES (WARP_COL_TILES * BLOCK_COL_WARPS)

#define GLOBAL_MEM_STRIDE N_GLOBAL

#define SHMEM_STRIDE (N * BLOCK_ROW_TILES)
#define SHMEM_OFFSET (N * WARP_ROW_TILES)

// The macro below is used to shift rows of the A matrix and columns of the B matrix
// in shared memory to minimize possible bank conflicts.
// Before performing the nvcuda::wmma::mma_sync operation, the warp must load the matrix
// data using the nvcuda::wmma::load_matrix_sync operation. Although the memory access pattern
// is not specified for that function, each lane in the warp can read one or multiple matrix
// elements from different matrix rows or columns.
// For shared memory, such access can result in bank conflicts if different rows / columns
// of the matrix map to the same bank. By shifting each row and column by a few bytes, we
// make sure that they map to different banks, thus reducing the number of possible bank
// conflicts.
// The number of 16 two-byte "half" elements is chosen as the minimum possible shift because
// we must keep each row and column 256-bit aligned, as required by nvcuda::wmma::load_matrix_sync.
#define SKEW_HALF 16

/*
DPCT1001:34: The statement could not be removed.
*/
/*
DPCT1000:35: Error handling if-stmt was detected but could not be rewritten.
*/
/*
DPCT1010:36: SYCL uses exceptions to report errors and does not use the error
codes. The call was replaced with 0. You need to rewrite this code.
*/
/*
DPCT1009:37: SYCL uses exceptions to report errors and does not use the error
codes. The call was replaced by a placeholder string. You need to rewrite this
code.
*/
#define checkKernelErrors(expr)                                                \
  do {                                                                         \
    expr;                                                                      \
                                                                               \
    dpct::err0 __err = 0;                                                      \
    if (__err != 0) {                                                          \
      printf("Line %d: '%s' failed: %s\n", __LINE__, #expr,                    \
             "<Placeholder string>");                                          \
      abort();                                                                 \
    }                                                                          \
  } while (0)

void init_host_matrices(sycl::half *a, sycl::half *b, float *c) {
  for (int i = 0; i < M_GLOBAL; i++) {
    for (int j = 0; j < K_GLOBAL; j++) {
     // a[i * K_GLOBAL + j] = (sycl::half)(rand() % 3);
     if (i == j)
        a[i * K_GLOBAL + j] = 1;
        else
            a[i * K_GLOBAL + j] = 0;

    }
  }

  for (int i = 0; i < N_GLOBAL; i++) {
    for (int j = 0; j < K_GLOBAL; j++) {
      b[i * K_GLOBAL + j] = 2;//(sycl::half)(rand() % 3);
    }
  }

  for (int t = 0; t < M_GLOBAL * N_GLOBAL; t++) {
    c[t] = 1;//static_cast<float>(rand() % 3);
  }
}

template <typename T>
void matrix_transpose(unsigned int rows, unsigned int cols, T* src, T* dest)
{
    for (unsigned int i = 0; i < rows; i++) {
        for (unsigned int j = 0; j < cols; j++) {

            dest[j * rows + i] = src[i * cols + j];
        }
    }

}


/*
DPCT1110:0: The total declared local variable size in device function
compute_gemm exceeds 128 bytes and may cause high register pressure. Consult
with your hardware vendor to find the total register size available and adjust
the code, or use smaller sub-group size to avoid high register pressure.
*/
void compute_gemm(const sycl::half *A, const sycl::half *B, const float *C,
                  float *D, float alpha, float beta,
                  const sycl::nd_item<3> &item_ct1, uint8_t* dpct_local) {
  auto shmem = (sycl::half(*)[80 /*CHUNK_K * K + SKEW_HALF*/])dpct_local;

  // Warp and lane identification.
  const unsigned int warpId = item_ct1.get_local_id(2) / WARP_SIZE;
  const unsigned int laneId = item_ct1.get_local_id(2) % WARP_SIZE;

  // Offset in shared memory from which the B matrix is stored.
  const size_t shmem_idx_b_off = BLOCK_COL_TILES * M;

  // This pointer is used to access the C and D matrix tiles this warp computes.
  float *shmem_warp_tile_ptr = (float *)&shmem[0][0] +
                               (warpId / 2) * SHMEM_STRIDE * K * 2 +
                               (warpId % 2) * SHMEM_OFFSET;

  // This pointer is used to stream the C and D matrices block-wide tile to and
  // from shared memory.
  float *shmem_warp_stream_ptr =
      (float *)&shmem[0][0] + warpId * SHMEM_STRIDE * K;

  // Adjust the beta scaler, as it'll be multiplied by alpha at the end of
  // each tile computation. Technically this is not generally correct (may
  // result in a loss of precision). Zero still needs to be specially handled
  // though.
  beta /= alpha;

  // Each CTA slides along the 128 x 128 tiles from the top left corner of the
  // matrix to the right and down, and selects the next tile to compute. Once
  // there's no such tile, all warps in this CTA exit.
  for (unsigned int block_pos = item_ct1.get_group(2);;
       block_pos += item_ct1.get_group_range(2)) {
    const unsigned int block_tile_i =
        ((block_pos * BLOCK_ROW_TILES) / N_TILES) * (BLOCK_COL_TILES);
    const unsigned int block_tile_j = (block_pos * BLOCK_COL_TILES) % N_TILES;

    // Stop when there are no more D matrix tiles to compute in this CTA.
    if (block_tile_i >= M_TILES) {
      break;
    }

    // This warp's pointer to the C matrix data to copy memory from to shared
    // memory.
    const size_t gmem_idx =
        (block_tile_i + warpId) * M * GLOBAL_MEM_STRIDE + block_tile_j * N;
    const float *src_gmem_warp_stream_ptr = &C[gmem_idx];

    // Stream multiple C tiles to shared memory.
#pragma unroll
    for (int i = 0; i < K; i++) {
      typedef sycl::int4 copy_t;

      *((copy_t *)(shmem_warp_stream_ptr + SHMEM_STRIDE * i) + laneId) =
          *((copy_t *)(src_gmem_warp_stream_ptr + GLOBAL_MEM_STRIDE * i) +
            laneId);
    }

    /*
    DPCT1118:1: SYCL group functions and algorithms must be encountered in
    converged control flow. You may need to adjust the code.
    */
    /*
    DPCT1065:24: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();

    // These fragments will accumulate the result of A and B matrix fragment
    // multiplications along the K_GLOBAL dimension.
    dpct::experimental::matrix::joint_matrix<
        dpct::experimental::matrix::accumulator, M, N, K, float>
        c[WARP_COL_TILES][WARP_ROW_TILES];

    // Load the C matrix tiles into fragments from shared memory.
#pragma unroll
    for (int i = 0; i < WARP_COL_TILES; i++) {
#pragma unroll
      for (int j = 0; j < WARP_ROW_TILES; j++) {
        const float *tile_ptr =
            shmem_warp_tile_ptr + i * SHMEM_STRIDE * K + j * N;

        sycl::ext::oneapi::experimental::matrix::joint_matrix_load(
            item_ct1.get_sub_group(), c[i][j].get(),
            sycl::address_space_cast<sycl::access::address_space::generic_space,
                                     sycl::access::decorated::no, const float>(
                tile_ptr),
            SHMEM_STRIDE, C_LAYOUT);
      }
    }

    /*
    DPCT1118:2: SYCL group functions and algorithms must be encountered in
    converged control flow. You may need to adjust the code.
    */
    /*
    DPCT1065:25: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();

    // Scale the C matrix.
#pragma unroll
    for (int i = 0; i < WARP_COL_TILES; i++) {
#pragma unroll
      for (int j = 0; j < WARP_ROW_TILES; j++) {
#pragma unroll
        for (int t = 0; t < c[i][j].num_elements; t++) {
          c[i][j].x[t] *= beta;
        }
      }
    }

    // Select what warp copies what matrix to shared memory.
    // Warps 0-3 copy the A matrix, warps 4-7 copy the B matrix.
    const sycl::half *warp_ptr = (warpId < 4)
                                     ? (&A[block_tile_i * M * K_GLOBAL] +
                                        M * K_GLOBAL * (warpId % 4) * 2)
                                     : (&B[block_tile_j * N * K_GLOBAL] +
                                        N * K_GLOBAL * (warpId % 4) * 2);

    // Go through the global K dimension by a fixed step at a time.
#pragma unroll
    for (int tile_k = 0; tile_k < K_TILES; tile_k += CHUNK_K) {
      // Copy slices of the A and B matrices to shared memory.
      // The first half of the warps in the CTA copy the A matrix, the rest copy
      // the B matrix.
      size_t shmem_idx =
          warpId < (WARPS_PER_BLOCK / 2)
              ? (M * (warpId % (WARPS_PER_BLOCK / 2)) * 2)
              : (N * (warpId % (WARPS_PER_BLOCK / 2)) * 2 + shmem_idx_b_off);

      // First half of the warp copies the first row / column of the matrix,
      // the second half of the warp copies the next.
      sycl::int4 *lane_ptr =
          (sycl::int4 *)(warp_ptr + tile_k * K +
                         (laneId / CHUNK_COPY_LINE_LANES) * K_GLOBAL) +
          (laneId % CHUNK_COPY_LINE_LANES);

      // Shift the second half of the warp to the next row / column in the
      // shared memory.
      shmem_idx += laneId / CHUNK_COPY_LINE_LANES;

#pragma unroll
      for (int i = 0; i < ((WARP_SIZE / 2) / CHUNK_COPY_LINES_PER_WARP) * 2;
           i++) {
        // Copy 16 bytes at once in each lane.
        *((sycl::int4 *)&shmem[shmem_idx][0] +
          (laneId % CHUNK_COPY_LINE_LANES)) = *lane_ptr;

        // Advance the global memory pointer and the shared memory index.
        lane_ptr = (sycl::int4 *)((sycl::half *)lane_ptr +
                                  K_GLOBAL * CHUNK_COPY_LINES_PER_WARP);
        shmem_idx += CHUNK_COPY_LINES_PER_WARP;
      }

      /*
      DPCT1118:5: SYCL group functions and algorithms must be encountered in
      converged control flow. You may need to adjust the code.
      */
      /*
      DPCT1065:28: Consider replacing sycl::nd_item::barrier() with
      sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
      performance if there is no access to global memory.
      */
      item_ct1.barrier();

      // Compute a grid of C matrix tiles in each warp.
#pragma unroll
      for (int k_step = 0; k_step < CHUNK_K; k_step++) {
        dpct::experimental::matrix::joint_matrix<
            dpct::experimental::matrix::a, M, N, K, sycl::half,
            dpct::experimental::matrix::row_major>
            a[WARP_COL_TILES];
        dpct::experimental::matrix::joint_matrix<
            dpct::experimental::matrix::b, M, N, K, sycl::half,
            dpct::experimental::matrix::row_major>
            b[WARP_ROW_TILES];

#pragma unroll
        for (int i = 0; i < WARP_COL_TILES; i++) {
          size_t shmem_idx_a = (warpId / 2) * M * 2 + (i * M);
          const sycl::half *tile_ptr = &shmem[shmem_idx_a][k_step * K];

          sycl::ext::oneapi::experimental::matrix::joint_matrix_load(
              item_ct1.get_sub_group(), a[i].get(),
              sycl::address_space_cast<
                  sycl::access::address_space::generic_space,
                  sycl::access::decorated::no, const sycl::half>(tile_ptr),
              K * CHUNK_K + SKEW_HALF);

#pragma unroll
          for (int j = 0; j < WARP_ROW_TILES; j++) {
            if (i == 0) {
              // Load the B matrix fragment once, because it is going to be
              // reused against the other A matrix fragments.
              size_t shmem_idx_b = shmem_idx_b_off +
                                   (WARP_ROW_TILES * N) * (warpId % 2) +
                                   (j * N);
              const sycl::half *tile_ptr = &shmem[shmem_idx_b][k_step * K];

              sycl::ext::oneapi::experimental::matrix::joint_matrix_load(
                  item_ct1.get_sub_group(), b[j].get(),
                  sycl::address_space_cast<
                      sycl::access::address_space::generic_space,
                      sycl::access::decorated::no, const sycl::half>(tile_ptr),
                  K * CHUNK_K + SKEW_HALF);
            }

            sycl::ext::oneapi::experimental::matrix::joint_matrix_mad(
                item_ct1.get_sub_group(), c[i][j].get(), a[i].get(), b[j].get(),
                c[i][j].get());
          }
        }
      }

      /*
      DPCT1118:6: SYCL group functions and algorithms must be encountered in
      converged control flow. You may need to adjust the code.
      */
      /*
      DPCT1065:29: Consider replacing sycl::nd_item::barrier() with
      sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
      performance if there is no access to global memory.
      */
      item_ct1.barrier();
    }

      // Store the D fragments to shared memory.
#pragma unroll
    for (int i = 0; i < WARP_COL_TILES; i++) {
#pragma unroll
      for (int j = 0; j < WARP_ROW_TILES; j++) {
#pragma unroll
        // Uniform, point-wise transformations of ALL fragment elements by ALL
        // threads in the warp are well-defined even though element indices
        // within fragment storage are not defined.
        for (int t = 0; t < c[i][j].num_elements; t++) c[i][j].x[t] *= alpha;

        float *tile_ptr = shmem_warp_tile_ptr + i * SHMEM_STRIDE * K + j * N;

        sycl::ext::oneapi::experimental::matrix::joint_matrix_store(
            item_ct1.get_sub_group(), c[i][j].get(),
            sycl::address_space_cast<sycl::access::address_space::generic_space,
                                     sycl::access::decorated::no, float>(
                tile_ptr),
            SHMEM_STRIDE, C_LAYOUT);
      }
    }

    /*
    DPCT1118:3: SYCL group functions and algorithms must be encountered in
    converged control flow. You may need to adjust the code.
    */
    /*
    DPCT1065:26: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();

    // Now that shared memory contains all the D tiles, stream them to global
    // memory.
    float *dst_gmem_warp_stream_ptr = &D[gmem_idx];

#pragma unroll
    for (int i = 0; i < K; i++) {
      *((sycl::int4 *)(dst_gmem_warp_stream_ptr + GLOBAL_MEM_STRIDE * i) +
        laneId) =
          *((sycl::int4 *)(shmem_warp_stream_ptr + SHMEM_STRIDE * i) + laneId);
    }

    /*
    DPCT1118:4: SYCL group functions and algorithms must be encountered in
    converged control flow. You may need to adjust the code.
    */
    /*
    DPCT1065:27: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
  }
}

// Performs an MxNxK GEMM (C=alpha*A*B + beta*C) assuming:
//  1) Matrices are packed in memory.
//  2) M, N and K are multiples of 16.
//  3) Neither A nor B are transposed.
// Note: This is a less performant version of the compute_gemm kernel. It is
// designed for
//       demonstration purposes only to show the CUDA WMMA API use without
//       relying on availability of the shared memory.
/*
DPCT1110:7: The total declared local variable size in device function
simple_wmma_gemm exceeds 128 bytes and may cause high register pressure. Consult
with your hardware vendor to find the total register size available and adjust
the code, or use smaller sub-group size to avoid high register pressure.
*/
void simple_wmma_gemm(sycl::half *a, sycl::half *b, float *c, float *d,
                      int m_ld, int n_ld, int k_ld, float alpha, float beta,
                      const sycl::nd_item<3> &item_ct1) {
  // Leading dimensions. Packed with no transpositions.
  int lda = k_ld;
  int ldb = k_ld;
  int ldc = n_ld;

  // Tile using a 2D grid
  int warpM = (item_ct1.get_group(2) * item_ct1.get_local_range(2) +
               item_ct1.get_local_id(2)) /
              item_ct1.get_sub_group().get_local_range().get(0);
  int warpN = (item_ct1.get_group(1) * item_ct1.get_local_range(1) +
               item_ct1.get_local_id(1));

  // Declare the fragments
  dpct::experimental::matrix::joint_matrix<
      dpct::experimental::matrix::a, WMMA_M, WMMA_N, WMMA_K, sycl::half,
      dpct::experimental::matrix::row_major>
      a_frag;
  dpct::experimental::matrix::joint_matrix<
      dpct::experimental::matrix::b, WMMA_M, WMMA_N, WMMA_K, sycl::half,
      dpct::experimental::matrix::row_major>
      b_frag;
  dpct::experimental::matrix::joint_matrix<
      dpct::experimental::matrix::accumulator, WMMA_M, WMMA_N, WMMA_K, float>
      acc_frag;
  dpct::experimental::matrix::joint_matrix<
      dpct::experimental::matrix::accumulator, WMMA_M, WMMA_N, WMMA_K, float>
      c_frag;

  sycl::ext::oneapi::experimental::matrix::joint_matrix_fill(
      item_ct1.get_sub_group(), acc_frag.get(), 0.0f);

  // Loop over k
  for (int i = 0; i < k_ld; i += WMMA_K) {
    int aCol = i;
    int aRow = warpM * WMMA_M;
    int bCol = warpN * N;
    int bRow = i;

    // Bounds checking
    if (aRow < m_ld && aCol < k_ld && bRow < k_ld && bCol < n_ld) {
      // Load the inputs
      sycl::ext::oneapi::experimental::matrix::joint_matrix_load(
          item_ct1.get_sub_group(), a_frag.get(),
          sycl::address_space_cast<sycl::access::address_space::generic_space,
                                   sycl::access::decorated::no,
                                   const sycl::half>(a + aCol + aRow * lda),
          lda);
      sycl::ext::oneapi::experimental::matrix::joint_matrix_load(
          item_ct1.get_sub_group(), b_frag.get(),
          sycl::address_space_cast<sycl::access::address_space::generic_space,
                                   sycl::access::decorated::no,
                                   const sycl::half>(b + bRow + bCol * ldb),
          ldb);

      // Perform the matrix multiplication
      sycl::ext::oneapi::experimental::matrix::joint_matrix_mad(
          item_ct1.get_sub_group(), acc_frag.get(), a_frag.get(), b_frag.get(),
          acc_frag.get());
    }
  }

  // Load in the current value of c, scale it by beta, and add this our result
  // scaled by alpha
  int cCol = warpN * WMMA_N;
  int cRow = warpM * WMMA_M;

  if (cRow < m_ld && cCol < n_ld) {
    sycl::ext::oneapi::experimental::matrix::joint_matrix_load(
        item_ct1.get_sub_group(), c_frag.get(),
        sycl::address_space_cast<sycl::access::address_space::generic_space,
                                 sycl::access::decorated::no, const float>(
            c + cCol + cRow * ldc),
        ldc, sycl::ext::oneapi::experimental::matrix::layout::row_major);

    for (int i = 0; i < c_frag.num_elements; i++) {
      c_frag.x[i] = alpha * acc_frag.x[i] + beta * c_frag.x[i];
    }

    // Store the output
    sycl::ext::oneapi::experimental::matrix::joint_matrix_store(
        item_ct1.get_sub_group(), c_frag.get(),
        sycl::address_space_cast<sycl::access::address_space::generic_space,
                                 sycl::access::decorated::no, float>(
            d + cCol + cRow * ldc),
        ldc, sycl::ext::oneapi::experimental::matrix::layout::row_major);
  }
}

void matMultiplyOnHost(sycl::half *A, sycl::half *B, float *C, float alpha,
                       float beta, int numARows, int numAColumns, int numBRows,
                       int numBColumns, int numCRows, int numCColumns) {
  for (int i = 0; i < numCRows; i++) {
    for (int j = 0; j < numCColumns; j++) {
      float temp = 0.0;

      for (int k = 0; k < numAColumns; k++) {
        temp += (float)A[i * numAColumns + k] * (float)B[j * numBRows + k];
      }

      C[i * numCColumns + j] = temp * alpha + beta * C[i * numCColumns + j];
    }
  }
}

int main(int argc, char **argv) {
        int count = 0;
  printf("Initializing...\n");

  int dev = findCudaDevice(argc, (const char **)argv);
std::cout << "\nRunning on " << dpct::get_default_queue().get_device().get_info<sycl::info::device::name>()
        << "\n";

  dpct::device_info deviceProp;
  checkCudaErrors(
      DPCT_CHECK_ERROR(dpct::get_device(dev).get_device_info(deviceProp)));

  // Tensor cores require a GPU of Volta (SM7X) architecture or higher.
  /*
  DPCT1005:30: The SYCL device version is different from CUDA Compute
  Compatibility. You may need to rewrite this code.
  */
/*  if (deviceProp.get_major_version() < 7) {
    printf(
        "cudaTensorCoreGemm requires SM 7.0 or higher to use Tensor "
        "Cores.  Exiting...\n");
    exit(EXIT_WAIVED);
  }*/

  printf("M: %d (%d x %d)\n", M_GLOBAL, M, M_TILES);
  printf("N: %d (%d x %d)\n", N_GLOBAL, N, N_TILES);
  printf("K: %d (%d x %d)\n", K_GLOBAL, K, K_TILES);

  sycl::half *A_h = NULL;
  sycl::half *B_h = NULL;
  sycl::half* B_t = NULL;
  float *C_h = NULL;
#if CPU_DEBUG
  float *result_hD = NULL;
  float *result_host = NULL;
#endif

  A_h = (sycl::half *)malloc(sizeof(sycl::half) * M_GLOBAL * K_GLOBAL);
  B_h = (sycl::half *)malloc(sizeof(sycl::half) * K_GLOBAL * N_GLOBAL);
  B_t = (sycl::half*)malloc(sizeof(sycl::half) * K_GLOBAL * N_GLOBAL);
  C_h = (float *)malloc(sizeof(float) * M_GLOBAL * N_GLOBAL);
#if CPU_DEBUG
  result_hD = (float *)malloc(sizeof(float) * M_GLOBAL * N_GLOBAL);
  result_host = (float *)malloc(sizeof(float) * M_GLOBAL * N_GLOBAL);
#endif

  sycl::half *A = NULL;
  sycl::half *B = NULL;
  float *C = NULL;
  float *D = NULL;

  checkCudaErrors(DPCT_CHECK_ERROR(A = (sycl::half *)sycl::malloc_device(
                                       sizeof(sycl::half) * M_GLOBAL * K_GLOBAL,
                                       dpct::get_in_order_queue())));
  checkCudaErrors(DPCT_CHECK_ERROR(B = (sycl::half *)sycl::malloc_device(
                                       sizeof(sycl::half) * N_GLOBAL * K_GLOBAL,
                                       dpct::get_in_order_queue())));
  checkCudaErrors(DPCT_CHECK_ERROR(
      C = (float *)sycl::malloc_device(sizeof(float) * M_GLOBAL * N_GLOBAL,
                                       dpct::get_in_order_queue())));
  checkCudaErrors(DPCT_CHECK_ERROR(
      D = (float *)sycl::malloc_device(sizeof(float) * M_GLOBAL * N_GLOBAL,
                                       dpct::get_in_order_queue())));

  assert(((unsigned long long)A) % 128 == 0);
  assert(((unsigned long long)B) % 128 == 0);
  assert(((unsigned long long)C) % 128 == 0);
  assert(((unsigned long long)D) % 128 == 0);

  init_host_matrices(A_h, B_h, C_h);

  printf("Preparing data for GPU...\n");

  checkCudaErrors(DPCT_CHECK_ERROR(
      dpct::get_in_order_queue()
          .memcpy(A, A_h, sizeof(sycl::half) * M_GLOBAL * K_GLOBAL)
          .wait()));
  matrix_transpose<sycl::half>(K_GLOBAL, N_GLOBAL, B_h, B_t);
  checkCudaErrors(DPCT_CHECK_ERROR(
      dpct::get_in_order_queue()
          .memcpy(B, B_t, sizeof(sycl::half) * N_GLOBAL * K_GLOBAL)
          .wait()));
  checkCudaErrors(
      DPCT_CHECK_ERROR(dpct::get_in_order_queue()
                           .memcpy(C, C_h, sizeof(float) * M_GLOBAL * N_GLOBAL)
                           .wait()));
  checkCudaErrors(
      DPCT_CHECK_ERROR(dpct::get_in_order_queue()
                           .memset(D, 0, sizeof(float) * M_GLOBAL * N_GLOBAL)
                           .wait()));

  enum {
    // Compute the right amount of shared memory to request.
    // We need shared memory to hold per-CTA C and D matrix tiles, and to cache
    // per-CTA chunks
    // of the A and B matrices. Therefore, the right amount to request is the
    // maximum of those
    // two numbers.
    SHMEM_SZ = MAX(sizeof(sycl::half) * (BLOCK_COL_TILES * M) *
                       (CHUNK_K * K + SKEW_HALF) * 2,
                   M * (BLOCK_ROW_WARPS * WARP_ROW_TILES) * N *
                       (BLOCK_COL_WARPS * WARP_COL_TILES) * sizeof(float))
  };

  printf("Required shared memory size: %lu Kb\n", SHMEM_SZ / 1024UL);

  const float alpha = 1.1f;
  const float beta = 1.2f;

  dpct::event_ptr start, stop;

  checkCudaErrors(DPCT_CHECK_ERROR(start = new sycl::event()));
  checkCudaErrors(DPCT_CHECK_ERROR(stop = new sycl::event()));
  /*
  DPCT1024:31: The original code returned the error code that was further
  consumed by the program logic. This original code was replaced with 0. You may
  need to rewrite the program logic consuming the error code.
  */
  checkCudaErrors(DPCT_CHECK_ERROR(dpct::sync_barrier(start)));

  // If enough shared memory available on the GPU use high performant kernel
  /*
  DPCT1019:32: local_mem_size in SYCL is not a complete equivalent of
  sharedMemPerMultiprocessor in CUDA. You may need to adjust the code.
  */
  if (deviceProp.get_local_mem_size() >= SHMEM_SZ) {
    printf("Computing... using high performance kernel compute_gemm \n");

    /*
    DPCT1027:33: The call to cudaFuncSetAttribute was replaced with 0 because
    SYCL currently does not support corresponding setting.
    */
    checkCudaErrors(0);
    /*
    DPCT1038:8: When the kernel function name is used as a macro argument, the
    migration result may be incorrect. You need to verify the definition of the
    macro.
    */
    checkKernelErrors(([&]() {
      dpct::get_device(
          dpct::get_device_id(dpct::get_in_order_queue().get_device()))
          .has_capability_or_fail({sycl::aspect::fp16});

      dpct::get_in_order_queue().submit([&](sycl::handler &cgh) {
//                    sycl::stream str(8192, 4, cgh);
        /*
        DPCT1101:39: 'CHUNK_K * K + SKEW_HALF' expression was replaced with a
        value. Modify the code to use the original expression, provided in
        comments, if it is correct.
        */
        sycl::local_accessor<uint8_t , 2> dpct_local_acc_ct1(
            sycl::range<2>(SHMEM_SZ, 1), cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(
                sycl::range<3>(1, 1, deviceProp.get_max_compute_units()) *
                    sycl::range<3>(1, 1, THREADS_PER_BLOCK),
                sycl::range<3>(1, 1, THREADS_PER_BLOCK)),
            [=](sycl::nd_item<3> item_ct1) {
//          str << "ABC" << sycl::endl;
              compute_gemm(A, B, C, D, alpha, beta, item_ct1,
                            dpct_local_acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get());
            });
      });
    }()));
#if CPU_DEBUG
/*    checkCudaErrors(cudaMemcpy(result_hD, D,
                               sizeof(float) * M_GLOBAL * N_GLOBAL,
                               cudaMemcpyDeviceToHost));*/
     (dpct::get_default_queue()
             .memcpy(result_hD, D, sizeof(float) * M_GLOBAL * N_GLOBAL)
             .wait());
#endif
  } else {
    dpct::dim3 gridDim;
    dpct::dim3 blockDim;

    // blockDim.x must be a multple of warpSize
    // 128x4 means we have 16 warps and a block computes a 64x64 output tile
    blockDim.x = 128;
    blockDim.y = 4;

    gridDim.x = (M_GLOBAL + (WMMA_M * blockDim.x / 32 - 1)) /
                (WMMA_M * blockDim.x / 32);
    gridDim.y = (N_GLOBAL + WMMA_N * blockDim.y - 1) / (WMMA_N * blockDim.y);

    printf("Computing... using simple_wmma_gemm kernel\n");
    /*
    DPCT1049:9: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    {
      dpct::get_device(
          dpct::get_device_id(dpct::get_in_order_queue().get_device()))
          .has_capability_or_fail({sycl::aspect::fp16});

      dpct::get_in_order_queue().submit([&](sycl::handler &cgh) {
        auto M_GLOBAL_ct4 = M_GLOBAL;
        auto N_GLOBAL_ct5 = N_GLOBAL;
        auto K_GLOBAL_ct6 = K_GLOBAL;

        cgh.parallel_for(sycl::nd_range<3>(gridDim * blockDim, blockDim),
                         [=](sycl::nd_item<3> item_ct1) {
                           simple_wmma_gemm(A, B, C, D, M_GLOBAL_ct4,
                                            N_GLOBAL_ct5, K_GLOBAL_ct6, alpha,
                                            beta, item_ct1);
                         });
      });
    }
#if CPU_DEBUG
/*    checkCudaErrors(cudaMemcpy(result_hD, D,
                               sizeof(float) * M_GLOBAL * N_GLOBAL,
                               cudaMemcpyDeviceToHost));*/
     (dpct::get_default_queue()
             .memcpy(result_hD, D, sizeof(float) * M_GLOBAL * N_GLOBAL)
             .wait());
#endif
  }

  /*
  DPCT1024:38: The original code returned the error code that was further
  consumed by the program logic. This original code was replaced with 0. You may
  need to rewrite the program logic consuming the error code.
  */
  checkCudaErrors(DPCT_CHECK_ERROR(dpct::sync_barrier(stop)));
  checkCudaErrors(DPCT_CHECK_ERROR(stop->wait_and_throw()));

#if CPU_DEBUG
  printf("Verifying correctness of the computations...\n");

  memcpy(result_host, C_h, sizeof(float) * M_GLOBAL * N_GLOBAL);

  matMultiplyOnHost(A_h, B_h, result_host, alpha, beta, M_GLOBAL, K_GLOBAL,
                    K_GLOBAL, N_GLOBAL, M_GLOBAL, N_GLOBAL);

  for (int i = 0; i < N_GLOBAL * M_GLOBAL; i++) {
    if (fabs(result_hD[i] - result_host[i]) > 0.1f)
      printf("mismatch i=%d result_hD=%f result_host=%f\n", i, result_hD[i], result_host[i]);
    else
            count++;
  }
  if (count == M_GLOBAL * N_GLOBAL)
        printf("TEST PASSED\n");
    else
        printf("count value: %d\t TEST FAILED\n", count);

  free(result_hD);
  free(result_host);
#endif

  float milliseconds = 0;

  checkCudaErrors(DPCT_CHECK_ERROR(
      milliseconds = (stop->get_profiling_info<
                          sycl::info::event_profiling::command_end>() -
                      start->get_profiling_info<
                          sycl::info::event_profiling::command_start>()) /
                     1000000.0f));

  printf("Time: %f ms\n", milliseconds);
  printf("TFLOPS: %.2f\n", static_cast<double>((static_cast<double>(M_GLOBAL) *
                                                N_GLOBAL * K_GLOBAL * 2) /
                                               (milliseconds / 1000.)) /
                               1e12);

  free(A_h);
  free(B_h);
  free(C_h);
  checkCudaErrors(DPCT_CHECK_ERROR(dpct::dpct_free(
      reinterpret_cast<void *>(A), dpct::get_in_order_queue())));
  checkCudaErrors(DPCT_CHECK_ERROR(dpct::dpct_free(
      reinterpret_cast<void *>(B), dpct::get_in_order_queue())));
  checkCudaErrors(DPCT_CHECK_ERROR(dpct::dpct_free(
      reinterpret_cast<void *>(C), dpct::get_in_order_queue())));
  checkCudaErrors(DPCT_CHECK_ERROR(dpct::dpct_free(
      reinterpret_cast<void *>(D), dpct::get_in_order_queue())));

  return 0;
}
