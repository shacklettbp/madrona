#pragma once

#include <madrona/math.hpp>

#ifdef MADRONA_GPU_MODE
#include <madrona/mw_gpu/host_print.hpp>
#define LOG(...) mwGPU::HostPrint::log(__VA_ARGS__)
#define warp_printf(...) if (threadIdx.x % 32 == 0) { printf(__VA_ARGS__); }

#define kernel_printf(...) if (invocation_idx == 0 && threadIdx.x == 0) {printf(__VA_ARGS__);}

#else
#define LOG(...)
#endif

#ifdef MADRONA_GPU_MODE
namespace madrona::phys::cv {

namespace gpu_utils {
// For debugging purposes
template <bool transposed = false, bool host_print = false>
void printMatrix(float *mat,
                 uint32_t num_rows,
                 uint32_t num_cols,
                 const char *name = "");

template <typename DataT>
DataT warpReduceSum(DataT value);

// You pass in a pointer to all the values to sum up
template <typename DataT>
DataT warpSum(DataT *values, uint32_t num_values);

template <typename DataT>
bool checkNan(DataT *values,
              uint32_t num_rows,
              uint32_t num_cols);

template <typename DataT, typename Fn>
DataT warpSumPred(Fn &&fn, uint32_t num_values);

// Simple helper function for having a warp loop over work
template <int granularity, typename Fn>
void warpLoop(uint32_t total_num_iters, Fn &&fn);

template <typename Fn>
void warpLoop(uint32_t total_num_iters, Fn &&fn);

// Passes in 0xFFFF'FFFF to fn if invalid run
template <typename Fn>
void warpLoopSync(uint32_t total_num_iters, Fn &&fn);

static inline void warpSetZero(void *dst, uint32_t num_bytes);
static inline void warpCopy(void *dst, void *src, uint32_t num_bytes);

template <typename DataT>
float norm2Warp(DataT *values, uint32_t dim);

template <typename DataT>
float dotVectors(DataT *a_ptr, DataT *b_ptr, uint32_t dim);

template <typename DataT, typename FnA, typename FnB>
float dotVectorsPred(
        FnA &&a_fn, FnB &&b_fn, uint32_t dim);

template <typename DataT,
          uint32_t block_size,
          bool transposed>
void copyToRegs(
        DataT (&blk_tmp)[block_size][block_size],
        DataT *mtx,
        uint32_t mtx_rows, uint32_t mtx_cols,
        uint32_t blk_r, uint32_t blk_c);

template <typename DataT,
          uint32_t block_size,
          bool transposed>
void copyToRegsWithBoundary(
        DataT (&blk_tmp)[block_size][block_size], // dst
        DataT *mtx,                               // src
        uint32_t mtx_rows_start, uint32_t mtx_cols_start,
        uint32_t mtx_rows_end, uint32_t mtx_cols_end,
        uint32_t mtx_rows, uint32_t mtx_cols,
        uint32_t blk_r, uint32_t blk_c);

template <typename DataT,
          uint32_t block_size>
void copyToMem(
        DataT *mtx,
        DataT (&blk_tmp)[block_size][block_size],
        uint32_t mtx_rows, uint32_t mtx_cols,
        uint32_t blk_r, uint32_t blk_c);

template <typename DataT,
          uint32_t block_size>
void copyToMemWithOffset(
        DataT *mtx,                               // dst
        DataT (&blk_tmp)[block_size][block_size], // src
        uint32_t mtx_rows, uint32_t mtx_cols,
        uint32_t blk_r, uint32_t blk_c,
        uint32_t r_offset, uint32_t c_offset,
        uint32_t r_end, uint32_t c_end);

template <typename DataT,
          uint32_t block_size,
          bool reset_res = false>
void gmmaBlockRegs(
        DataT (&res)[block_size][block_size],
        DataT (&a)[block_size][block_size],
        DataT (&b)[block_size][block_size]);

template <typename DataT,
          uint32_t block_size>
void setBlockZero(
        DataT (&blk)[block_size][block_size]);

template <typename DataT,
          uint32_t block_size>
void gmmaWarpSmallSmem(
        DataT *res,
        DataT *a,
        DataT *b,
        uint32_t a_rows, uint32_t a_cols,
        uint32_t b_rows, uint32_t b_cols);

template <typename DataT,
          uint32_t block_size,
          bool a_transposed,
          bool b_transposed,
          bool reset_res>
void gmmaWarpSmallReg(
        DataT *res,
        DataT *a,
        DataT *b,
        uint32_t a_rows, uint32_t a_cols,
        uint32_t b_rows, uint32_t b_cols);

template <typename DataT>
DataT warpInclusivePrefixSum(DataT value);

// ReadFn : uint32_t iter -> DataT
// WriteFn : uint32_t iter, DataT epf -> void
template <typename DataT, typename ReadFn, typename WriteFn>
void warpExclusivePrefixSumPred(
        ReadFn &&read_fn,
        WriteFn &&write_fn,
        uint32_t num_elems);

template <typename DataT,
          uint32_t block_size,
          bool a_transposed,
          bool b_transposed,
          bool reset_res>
void sparseBlkDiagSmallReg(
        DataT *res,
        SparseBlkDiag *a,
        DataT *b,
        uint32_t b_rows, uint32_t b_cols);

template <typename DataT, bool dot_res_and_input>
DataT sparseBlkDiagSolve(
        DataT *res,
        SparseBlkDiag *a,
        DataT *scratch);

template <typename DataT>
void blkDiagSolve(
        DataT *res,
        DataT *a_ltdl,
        DataT *b,
        uint32_t a_dim);
    
}
}
#endif

#include "cv_gpu.inl"
