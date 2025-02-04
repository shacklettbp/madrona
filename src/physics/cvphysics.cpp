#include <algorithm>
#include <madrona/state.hpp>
#include <madrona/utils.hpp>
#include <madrona/physics.hpp>
#include <madrona/context.hpp>
#include <madrona/cvphysics.hpp>
#include <madrona/taskgraph.hpp>

#include "physics_impl.hpp"

#ifdef MADRONA_GPU_MODE
#include <madrona/mw_gpu/host_print.hpp>
#define LOG(...) mwGPU::HostPrint::log(__VA_ARGS__)
#define warp_printf(...) if (threadIdx.x % 32 == 0) { printf(__VA_ARGS__); }

#define kernel_printf(...) if (invocation_idx == 0 && threadIdx.x == 0) {printf(__VA_ARGS__);}

#else
#define LOG(...)
#endif

using namespace madrona::math;
using namespace madrona::base;

struct SparseBlkDiag {
    struct Blk {
        uint32_t dim;
        uint32_t scratch;
        float *values;

        // factorized version
        float *ltdl;
        int32_t *expandedParent;
    };

    uint32_t fullDim;
    uint32_t numBlks;
    Blk *blks;
};

#ifdef MADRONA_GPU_MODE
namespace madrona::gpu_utils {

// For debugging purposes
template <bool transposed = false, bool host_print = false>
void printMatrix(float *mat,
                 uint32_t num_rows,
                 uint32_t num_cols,
                 const char *name = "")
{
    __syncwarp();
    if (threadIdx.x % 32 == 0) {
        if constexpr (host_print) {
            LOG("printing matrix {}\n", name);
        } else {
            printf("printing matrix %s\n", name);
        }

        for (uint32_t r = 0; r < num_rows; ++r) {
            if constexpr (host_print) {
                LOG("row {}\n", r);
            }

            for (uint32_t c = 0; c < num_cols; ++c) {
                float v = 0.f;
                if constexpr (transposed) {
                    v = mat[r + c * num_rows];
                } else {
                    v = mat[c + r * num_cols];
                }

                if constexpr (host_print) {
                    LOG("{}\n", v);
                } else {
                    printf("%f ", v);
                }
            }

            if constexpr (!host_print) {
                printf("\n");
            }
        }

        printf("\n");
    }
    __syncwarp();
}

template <typename DataT>
DataT warpReduceSum(DataT value)
{
    #pragma unroll
    for (int i=16; i>=1; i/=2)
        value += __shfl_xor_sync(0xffffffff, value, i, 32);

    return value;
}

// You pass in a pointer to all the values to sum up
template <typename DataT>
DataT warpSum(DataT *values, uint32_t num_values)
{
    uint32_t num_iters = (values + 31) / 32;
    DataT running_sum = (DataT)0;

    warpLoopSync(
        num_iters,
        [&](uint32_t iter) {
            running_sum += warpReduceSum(
                    iter == 0xFFFF'FFFF ? 0 : values[iter]);
        });

    return running_sum;
}

template <typename DataT>
bool checkNan(DataT *values,
              uint32_t num_rows,
              uint32_t num_cols)
{
    uint32_t total_num_values = num_rows * num_cols;

    warpLoopSync(
        total_num_values,
        [&](uint32_t iter) {
            float v = (iter == 0xFFFF'FFFF) ? 0.f : values[iter];
            bool invalid = isnan(v);


        });
}

template <typename DataT, typename Fn>
DataT warpSumPred(Fn &&fn, uint32_t num_values)
{
    DataT running_sum = (DataT)0;

    warpLoopSync(
        num_values,
        [&](uint32_t iter) {
            running_sum += warpReduceSum(
                    (iter == 0xFFFF'FFFF) ? 0 : fn(iter));
        });

    return running_sum;
}

// Simple helper function for having a warp loop over work
template <int granularity, typename Fn>
void warpLoop(uint32_t total_num_iters, Fn &&fn)
{
    uint32_t iter = granularity * (threadIdx.x % 32);
    while (iter < total_num_iters) {
        #pragma unroll
        for (int i = 0; i < granularity; ++i) {
            fn(iter + i);
        }

        iter += 32 * granularity;
    }
}

template <typename Fn>
void warpLoop(uint32_t total_num_iters, Fn &&fn)
{
    uint32_t iter = threadIdx.x % 32;
    while (iter < total_num_iters) {
        fn(iter);

        iter += 32;
    }
}

// Passes in 0xFFFF'FFFF to fn if invalid run
template <typename Fn>
void warpLoopSync(uint32_t total_num_iters, Fn &&fn)
{
    uint32_t iter = threadIdx.x % 32;
    bool run = (iter < total_num_iters);

    while (__any_sync(0xFFFF'FFFF, run)) {
        fn(run ? iter : 0xFFFF'FFFF);
        iter += 32;

        run = (iter < total_num_iters);
    }
}

void warpSetZero(void *dst, uint32_t num_bytes)
{
    int32_t lane_id = threadIdx.x % 32;
    int32_t bytes_per_warp = (num_bytes + 31) / 32;
    int32_t bytes_to_set =
        max(0, min((int32_t)num_bytes - lane_id * bytes_per_warp, bytes_per_warp));

    memset(
        (uint8_t *)dst + bytes_per_warp * lane_id,
        0,
        bytes_to_set);
}

void warpCopy(void *dst, void *src, uint32_t num_bytes, bool dbg = false)
{
    int32_t lane_id = threadIdx.x % 32;
    int32_t bytes_per_warp = (num_bytes + 31) / 32;
    int32_t bytes_to_cpy =
        max(0, min((int32_t)num_bytes - lane_id * bytes_per_warp, bytes_per_warp));

    memcpy(
        (uint8_t *)dst + bytes_per_warp * lane_id,
        (uint8_t *)src + bytes_per_warp * lane_id,
        bytes_to_cpy);
}

template <typename DataT>
float norm2Warp(DataT *values, uint32_t dim)
{
    float cur_sum = 0.f;

    warpLoopSync(dim, [&](uint32_t iter) {
        float v = (iter != 0xFFFF'FFFF) ? values[iter] : 0.f;

        v = v * v;
        v = warpReduceSum(v) + cur_sum;
        cur_sum = v;
    });

    return cur_sum;
}

template <typename DataT>
float dotVectors(
        DataT *a_ptr, DataT *b_ptr, uint32_t dim)
{
    float cur_sum = 0.f;

    warpLoopSync(dim, [&](uint32_t iter) {
        float a = (iter != 0xFFFF'FFFF) ? a_ptr[iter] : 0.f;
        float b = (iter != 0xFFFF'FFFF) ? b_ptr[iter] : 0.f;

        float v = a * b;
        v = warpReduceSum(v) + cur_sum;
        cur_sum = v;
    });

    return cur_sum;
}

template <typename DataT, typename FnA, typename FnB>
float dotVectorsPred(
        FnA &&a_fn, FnB &&b_fn, uint32_t dim)
{
    float cur_sum = 0.f;

    warpLoopSync(dim, [&](uint32_t iter) {
        DataT a = (iter != 0xFFFF'FFFF) ? a_fn(iter) : 0.f;
        DataT b = (iter != 0xFFFF'FFFF) ? b_fn(iter) : 0.f;

        DataT v = a * b;
        v = warpReduceSum(v) + cur_sum;
        cur_sum = v;
    });

    return cur_sum;
}

template <typename DataT,
          uint32_t block_size,
          bool transposed>
void copyToRegs(
        DataT (&blk_tmp)[block_size][block_size],
        DataT *mtx,
        uint32_t mtx_rows, uint32_t mtx_cols,
        uint32_t blk_r, uint32_t blk_c)
{
    uint32_t col_start = blk_c * block_size;
    uint32_t row_start = blk_r * block_size;

    #pragma unroll
    for (uint32_t blk_row = 0; blk_row < block_size; ++blk_row) {
        #pragma unroll
        for (uint32_t blk_col = 0; blk_col < block_size; ++blk_col) {
            if constexpr (transposed) {
                blk_tmp[blk_row][blk_col] = 
                    (col_start + blk_col < mtx_cols &&
                     row_start + blk_row < mtx_rows) ?
                    mtx[row_start + blk_row + mtx_rows * (col_start + blk_col)] :
                    0.f;
            } else {
                blk_tmp[blk_row][blk_col] = 
                    (col_start + blk_col < mtx_cols &&
                     row_start + blk_row < mtx_rows) ?
                    mtx[col_start + blk_col + mtx_cols * (row_start + blk_row)] :
                    0.f;
            }
        }
    }
}

template <typename DataT,
          uint32_t block_size,
          bool transposed>
void copyToRegsWithBoundary(
        DataT (&blk_tmp)[block_size][block_size], // dst
        DataT *mtx,                               // src
        uint32_t mtx_rows_start, uint32_t mtx_cols_start,
        uint32_t mtx_rows_end, uint32_t mtx_cols_end,
        uint32_t mtx_rows, uint32_t mtx_cols,
        uint32_t blk_r, uint32_t blk_c)
{
    uint32_t col_start = blk_c * block_size + mtx_cols_start;
    uint32_t row_start = blk_r * block_size + mtx_rows_start;

    #pragma unroll
    for (uint32_t blk_row = 0; blk_row < block_size; ++blk_row) {
        #pragma unroll
        for (uint32_t blk_col = 0; blk_col < block_size; ++blk_col) {
            if constexpr (transposed) {
                blk_tmp[blk_row][blk_col] = 
                    (col_start + blk_col < mtx_cols_end &&
                     row_start + blk_row >= mtx_rows_start &&
                     row_start + blk_row < mtx_rows_end) ?
                    mtx[row_start + blk_row + mtx_rows * (col_start + blk_col)] :
                    0.f;
            } else {
                blk_tmp[blk_row][blk_col] = 
                    (col_start + blk_col < mtx_cols_end &&
                     row_start + blk_row >= mtx_rows_start &&
                     row_start + blk_row < mtx_rows_end) ?
                    mtx[col_start + blk_col + mtx_cols * (row_start + blk_row)] :
                    0.f;
            }
        }
    }
}

template <typename DataT,
          uint32_t block_size>
void copyToMem(
        DataT *mtx,
        DataT (&blk_tmp)[block_size][block_size],
        uint32_t mtx_rows, uint32_t mtx_cols,
        uint32_t blk_r, uint32_t blk_c)
{
    uint32_t col_start = blk_c * block_size;
    uint32_t row_start = blk_r * block_size;

    #pragma unroll
    for (uint32_t blk_row = 0; blk_row < block_size; ++blk_row) {
        #pragma unroll
        for (uint32_t blk_col = 0; blk_col < block_size; ++blk_col) {
            if (col_start + blk_col < mtx_cols &&
                 row_start + blk_row < mtx_rows) {
                mtx[col_start + blk_col + mtx_cols * (row_start + blk_row)] =
                    blk_tmp[blk_row][blk_col];
            }
        }
    }
}

template <typename DataT,
          uint32_t block_size>
void copyToMemWithOffset(
        DataT *mtx,                               // dst
        DataT (&blk_tmp)[block_size][block_size], // src
        uint32_t mtx_rows, uint32_t mtx_cols,
        uint32_t blk_r, uint32_t blk_c,
        uint32_t r_offset, uint32_t c_offset,
        uint32_t r_end, uint32_t c_end)
{
    uint32_t col_start = blk_c * block_size + c_offset;
    uint32_t row_start = blk_r * block_size + r_offset;

    #pragma unroll
    for (uint32_t blk_row = 0; blk_row < block_size; ++blk_row) {
        #pragma unroll
        for (uint32_t blk_col = 0; blk_col < block_size; ++blk_col) {
            if (col_start + blk_col < c_end &&
                 row_start + blk_row < r_end) {
                mtx[col_start + blk_col + mtx_cols * (row_start + blk_row)] =
                    blk_tmp[blk_row][blk_col];
            }
        }
    }
}

template <typename DataT,
          uint32_t block_size,
          bool reset_res = false>
void gmmaBlockRegs(
        DataT (&res)[block_size][block_size],
        DataT (&a)[block_size][block_size],
        DataT (&b)[block_size][block_size])
{
    #pragma unroll
    for (int i = 0; i < block_size; i++) {
        #pragma unroll
        for (int j = 0; j < block_size; j++) {
            if constexpr (reset_res) {
                res[i][j] = 0;
            }

            #pragma unroll
            for (int k = 0; k < block_size; k++) {
                res[i][j] += a[i][k] * b[k][j];
            }
        }
    }
}

template <typename DataT,
          uint32_t block_size>
void setBlockZero(
        DataT (&blk)[block_size][block_size])
{
    #pragma unroll
    for (int i = 0; i < block_size; i++) {
        #pragma unroll
        for (int j = 0; j < block_size; j++) {
            blk[i][j] = 0.f;
        }
    }
}

template <typename DataT,
          uint32_t block_size>
void gmmaWarpSmallSmem(
        DataT *res,
        DataT *a,
        DataT *b,
        uint32_t a_rows, uint32_t a_cols,
        uint32_t b_rows, uint32_t b_cols)
{
    const int32_t num_smem_bytes_per_warp =
        mwGPU::SharedMemStorage::numBytesPerWarp();
    auto smem_buf = (uint8_t *)mwGPU::SharedMemStorage::buffer +
                    num_smem_bytes_per_warp * warp_id;

    // This function requires (block_size x block_size) * 3 bytes of Smem
    assert(3 * (block_size * block_size) < num_smem_bytes_per_warp);

    DataT *a_blk_tmp = (DataT *)smem_buf;
    DataT *b_blk_tmp = (DataT *)smem_buf + (block_size * block_size);
    DataT *res_blk_tmp = (DataT *)smem_buf + (block_size * block_size) * 2;

    // Get value in block tmp matrix
    // TODO: Double check the compiler spits out not dumb stuff.
    auto bv = [](DataT *d, uint32_t row, uint32_t col) -> DataT & {
        return d[col + row * block_size];
    };

    
}

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
        uint32_t b_rows, uint32_t b_cols)
{
    // TODO: Make sure non of this stuff spills
    DataT a_blk_tmp[block_size][block_size];
    DataT b_blk_tmp[block_size][block_size];
    DataT res_blk_tmp[block_size][block_size];

    uint32_t res_rows = a_rows;
    uint32_t res_cols = b_cols;

    uint32_t num_iters_r = (res_rows + block_size - 1) / block_size;
    uint32_t num_iters_c = (res_cols + block_size - 1) / block_size;
    uint32_t total_num_iters = num_iters_r * num_iters_c;

    uint32_t num_blks_b = (b_rows + block_size - 1) / block_size;

    uint32_t lane_id = threadIdx.x % 32;
    uint32_t cur_iter = lane_id;
    
    while (cur_iter < total_num_iters) {
        uint32_t res_blk_r = cur_iter / num_iters_c;
        uint32_t res_blk_c = cur_iter % num_iters_c;

        if constexpr (reset_res) {
            setBlockZero(res_blk_tmp);
        } else {
            copyToRegs<DataT, block_size, false>(
                    res_blk_tmp, res, res_rows, res_cols,
                    res_blk_r, res_blk_c);
        }

        for (uint32_t blk_j = 0; blk_j < num_blks_b; ++blk_j) {
            copyToRegs<DataT, block_size, a_transposed>(
                    a_blk_tmp, a, a_rows, a_cols, res_blk_r, blk_j);
            copyToRegs<DataT, block_size, b_transposed>(
                    b_blk_tmp, b, b_rows, b_cols, blk_j, res_blk_c);

            gmmaBlockRegs(res_blk_tmp, a_blk_tmp, b_blk_tmp);
        }

        copyToMem(res, res_blk_tmp, res_rows, 
                  res_cols, res_blk_r, res_blk_c);

        cur_iter += 32;
    }
}

template <typename DataT>
DataT warpInclusivePrefixSum(DataT value)
{
    uint32_t lane_id = threadIdx.x % 32;

    #pragma unroll
    for (uint32_t i = 1; i <= 32; i *= 2) {
        DataT prev_blk = __shfl_up_sync(0xFFFF'FFFF, value, i, 32);
        if (lane_id >= i) value += prev_blk;
    }

    return value;
}

template <typename DataT,
          uint32_t block_size,
          bool a_transposed,
          bool b_transposed,
          bool reset_res>
void sparseBlkDiagSmallReg(
        DataT *res,
        SparseBlkDiag *a,
        DataT *b,
        uint32_t b_rows, uint32_t b_cols)
{
    DataT a_blk_tmp[block_size][block_size];
    DataT b_blk_tmp[block_size][block_size];
    DataT res_blk_tmp[block_size][block_size];

    uint32_t res_rows = a->fullDim;
    uint32_t res_cols = b_cols;

    auto get_num_iters = [b_cols](SparseBlkDiag::Blk blk) {
        uint32_t res_rows = blk.dim;
        uint32_t res_cols = b_cols;
        uint32_t num_iters_r = (res_rows + block_size - 1) / block_size;
        uint32_t num_iters_c = (res_cols + block_size - 1) / block_size;
        return num_iters_r * num_iters_c;
    };

    uint32_t lane_id = threadIdx.x % 32;

    // Everyone starts on blk 0
    uint32_t cur_blk = 0;
    uint32_t cur_num_iters = get_num_iters(a->blks[cur_blk]);
    uint32_t cur_iter = lane_id;
    uint32_t total_cur_iter = lane_id;

    bool work_finished = false;

    uint32_t processed_dims = 0;

    while (!work_finished) {
        while (cur_iter >= cur_num_iters) {
            processed_dims += a->blks[cur_blk].dim;

            cur_blk++;
            cur_iter -= cur_num_iters;

            if (cur_blk < a->numBlks) {
                cur_num_iters = get_num_iters(a->blks[cur_blk]);
            } else {
                work_finished = true;
                break;
            }
        }

        if (work_finished)
            break;

        auto blk = a->blks[cur_blk];
        uint32_t blk_num_iters = get_num_iters(blk);

        uint32_t num_iters_c = (b_cols + block_size - 1) / block_size;

        // These are blocks within the blk
        uint32_t res_blk_r = cur_iter / num_iters_c;
        uint32_t res_blk_c = cur_iter % num_iters_c;

        uint32_t num_blks_b = (blk.dim + block_size - 1) / block_size;

        if constexpr (reset_res) {
            setBlockZero(res_blk_tmp);
        } else {
#if 0
            copyToRegs<DataT, block_size, false>(
                    res_blk_tmp, res, res_rows, res_cols,
                    res_blk_r, res_blk_c);
#endif
        }

        for (uint32_t blk_j = 0; blk_j < num_blks_b; ++blk_j) {
            // This is trivial
            copyToRegs<DataT, block_size, a_transposed>(
                    a_blk_tmp, blk.values, blk.dim, blk.dim, 
                    res_blk_r, blk_j);

            // This is a little more complicated
            copyToRegsWithBoundary<DataT, block_size, b_transposed>(
                    b_blk_tmp, 
                    b, 
                    processed_dims, 0,
                    processed_dims + blk.dim, b_cols,
                    b_rows, b_cols,
                    blk_j, res_blk_c);

            gmmaBlockRegs(res_blk_tmp, a_blk_tmp, b_blk_tmp);
        }

#if 0
        printf("iter %d (offset %d), values: %f %f\n",
                total_cur_iter,
                processed_dims,
                )
#endif
        copyToMemWithOffset(res, res_blk_tmp, 
                            res_rows, res_cols, 
                            res_blk_r, res_blk_c,
                            processed_dims, 0,
                            processed_dims + blk.dim, res_cols);

#if 0 // Save for later to potenatially reduce divergence
        bool valid_work;
        uint32_t cur_num_iters;

        if (cur_blk < a->numBlks) {
            cur_num_iters = get_num_iters(a->blks[cur_blk]);
            valid_work = (cur_iter < cur_num_iters);
        } else {
            valid_work = true;
            work_finished = true;

            cur_iter = 0;
            cur_num_iters = 0xFFFF'FFFF;
        }
        
        while (!__all_sync(0xFFFF'FFFF, valid_work)) {
            if (cur_iter >= cur_num_iters) {
                cur_blk++;
                cur_iter -= cur_num_iters;

                if (cur_blk < a->numBlks)
                    cur_num_iters = get_num_iters(a->blks[cur_blk]);
            }
        }
#endif

        cur_iter += 32;
        total_cur_iter += 32;
    }
}

template <typename DataT, bool dot_res_and_input>
DataT sparseBlkDiagSolve(
        DataT *res,
        SparseBlkDiag *a,
        DataT *scratch)
{
    uint32_t lane_id = threadIdx.x % 32;
    uint32_t cur_dim_offset = 0;

    if constexpr (dot_res_and_input) {
        warpCopy(scratch, res, a->fullDim * sizeof(DataT));
    }

    // I think for now, we are just going to naively assign each warp
    // to a block to invert.
    warpLoopSync(a->numBlks, [&](uint32_t iter) {
        auto [num_dims, blk] = [&]() -> 
            std::pair<uint32_t, SparseBlkDiag::Blk> {
            if (iter == 0xFFFF'FFFF) {
                return { 0, SparseBlkDiag::Blk{} };
            } else {
                auto blk = a->blks[iter];
                return { blk.dim, blk };
            }
        } ();

        uint32_t dim_offset = warpInclusivePrefixSum(num_dims) +
                              cur_dim_offset;
        if (lane_id == 31) {
            cur_dim_offset = dim_offset;
        }

        CountT total_dofs = blk.dim;

        int32_t *expandedParent = blk.expandedParent;
        float *massMatrixLTDL = blk.ltdl;
        float *x = res + dim_offset - num_dims;

        auto ltdl = [&](int32_t row, int32_t col) -> float& {
            return massMatrixLTDL[row + total_dofs * col];
        };

        for (int32_t i = (int32_t) total_dofs - 1; i >= 0; --i) {
            int32_t j = expandedParent[i];
            while (j != -1) {
                x[j] -= ltdl(i, j) * x[i];
                j = expandedParent[j];
            }
        }

        for (int32_t i = 0; i < total_dofs; ++i) {
            if (ltdl(i, i) == 0.f) {
                printf("nan going to happen ltdl!\n");
            }
            x[i] /= ltdl(i, i);
        }

        for (int32_t i = 0; i < total_dofs; ++i) {
            int32_t j = expandedParent[i];
            while (j != -1) {
                x[i] -= ltdl(i, j) * x[j];
                j = expandedParent[j];
            }
        }
    });

    float ret = 0.f;
    if constexpr (dot_res_and_input) {
        ret = dotVectors(res, scratch, a->fullDim);
    }

    return ret;
}

template <typename DataT>
void blkDiagSolve(
        DataT *res,
        DataT *a_ltdl,
        DataT *b,
        uint32_t a_dim)
{

}

}
#endif

namespace madrona::phys::cv {

StateManager * getStateManager(Context &ctx)
{
#ifdef MADRONA_GPU_MODE
    return mwGPU::getStateManager();
#else
    return ctx.getStateManager();
#endif
}

StateManager * getStateManager()
{
#ifdef MADRONA_GPU_MODE
    return mwGPU::getStateManager();
#else
    assert(false);
#endif
}

struct MRElement128b {
    uint8_t d[128];
};

struct SolverScratch256b {
    uint8_t d[256];
};

float * DofObjectTmpState::getPhiFull(Context &ctx)
{
    uint8_t *bytes =
        (uint8_t *)ctx.memoryRangePointer<MRElement128b>(dynData) +
        phiFullOffset;
    return (float *)bytes;
}

float * BodyGroupHierarchy::getMassMatrix(Context &ctx)
{
    uint8_t *bytes =
        (uint8_t *)ctx.memoryRangePointer<MRElement128b>(dynData) +
        massMatrixOffset;
    return (float *)bytes;
}

float * BodyGroupHierarchy::getMassMatrixLTDL(Context &ctx)
{
    uint8_t *bytes =
        (uint8_t *)ctx.memoryRangePointer<MRElement128b>(dynData) +
        massMatrixLTDLOffset;
    return (float *)bytes;
}

float * BodyGroupHierarchy::getBias(Context &ctx)
{
    uint8_t *bytes =
        (uint8_t *)ctx.memoryRangePointer<MRElement128b>(dynData) +
        biasOffset;
    return (float *)bytes;
}

int32_t * BodyGroupHierarchy::getExpandedParent(Context &ctx)
{
    uint8_t *bytes =
        (uint8_t *)ctx.memoryRangePointer<MRElement128b>(dynData) +
        expandedParentOffset;
    return (int32_t *)bytes;
}

Entity * BodyGroupHierarchy::bodies(Context &ctx)
{
    uint8_t *bytes =
        (uint8_t *)ctx.memoryRangePointer<MRElement128b>(mrBodies);
    return (Entity *)bytes;
}

uint32_t * BodyGroupHierarchy::getDofPrefixSum(Context &ctx)
{
    uint8_t *bytes =
        (uint8_t *)ctx.memoryRangePointer<MRElement128b>(dynData) +
        dofPrefixSumOffset;
    return (uint32_t *)bytes;
}

BodyObjectData *BodyGroupHierarchy::getCollisionData(Context &ctx)
{
    uint8_t *bytes =
        (uint8_t *)ctx.memoryRangePointer<MRElement128b>(mrCollisionVisual);
    return (BodyObjectData *)bytes;
}

BodyObjectData *BodyGroupHierarchy::getVisualData(Context &ctx)
{
    uint8_t *bytes =
        (uint8_t *)ctx.memoryRangePointer<MRElement128b>(mrCollisionVisual) +
        sizeof(BodyObjectData) * collisionObjsCounter;
    return (BodyObjectData *)bytes;
}

struct Contact : Archetype<
    ContactConstraint,
    ContactTmpState
> {};

struct Joint : Archetype<
    JointConstraint
> {};

struct DummyComponent {

};

struct CVRigidBodyState : Bundle<
    DummyComponent
> {};

struct CVSolveData {
    uint32_t numBodyGroups;
    uint32_t *dofOffsets;

    uint32_t totalNumDofs;
    uint32_t numContactPts;
    float h;

    // Values
    float *mass;
    float *freeAcc;
    float *vel;
    float *J_c;
    float *J_e;
    float *mu;
    float *penetrations;
    float *eqResiduals;
    // Diagonal approximations of A = J * M^-1 * J^T
    float *diagApprox_c;
    float *diagApprox_e;

    uint32_t massDim;
    uint32_t freeAccDim;
    uint32_t velDim;

    uint32_t numRowsJc;
    uint32_t numColsJc;

    uint32_t numRowsJe;
    uint32_t numColsJe;

    uint32_t muDim;
    uint32_t penetrationsDim;

    // Sum of diagonals of mass matrix
    float totalMass;

    enum StateFlags {
        // Is a_ref stored in shared memory?
        ARefSmem = 1 << 0
    };

    uint32_t flags;

    SparseBlkDiag massSparse;

    CVXSolve *cvxSolve;

    uint8_t *solverScratchMem;
    uint8_t *accRefMem;
    uint8_t *prepMem;
    uint32_t scratchAllocatedBytes;
    uint32_t accRefAllocatedBytes;
    uint32_t prepAllocatedBytes;
#if 0
    // Only relevant for the GPU implementation
    MemoryRange solverScratchMemory;

    MemoryRange accRefMemory;

    // This has mass matrix, full vel, free acc, jacobian, mu,
    // penetrations
    MemoryRange prepMemory;
#endif

    static constexpr uint32_t kNumRegisters = 8;

    struct RegInfo {
        uint64_t size;
        bool inSmem;
        void *ptr;
    };

    RegInfo regInfos[kNumRegisters];

#ifdef MADRONA_GPU_MODE
    SparseBlkDiag::Blk * getMassBlks(StateManager *state_mgr)
    {
        uint8_t *bytes =
            //(uint8_t *)state_mgr->memoryRangePointer<SolverScratch256b>(prepMemory);
            prepMem;
        return (SparseBlkDiag::Blk *)bytes;
    }

    float * getFullVel(StateManager *state_mgr)
    {
        uint8_t *bytes =
            //(uint8_t *)state_mgr->memoryRangePointer<SolverScratch256b>(prepMemory) +
            prepMem +
            sizeof(SparseBlkDiag::Blk) * numBodyGroups;
        return (float *)bytes;
    }

    float * getFreeAcc(StateManager *state_mgr)
    {
        uint8_t *bytes =
            // (uint8_t *)state_mgr->memoryRangePointer<SolverScratch256b>(prepMemory) +
            prepMem +
            sizeof(SparseBlkDiag::Blk) * numBodyGroups +
            sizeof(float) * totalNumDofs;
        return (float *)bytes;
    }

    float * getMu(StateManager *state_mgr)
    {
        uint8_t *bytes =
            // (uint8_t *)state_mgr->memoryRangePointer<SolverScratch256b>(prepMemory) +
            prepMem +
            sizeof(SparseBlkDiag::Blk) * numBodyGroups +
            sizeof(float) * totalNumDofs +
            sizeof(float) * totalNumDofs;
        return (float *)bytes;
    }

    float * getPenetrations(StateManager *state_mgr)
    {
        uint8_t *bytes =
            //(uint8_t *)state_mgr->memoryRangePointer<SolverScratch256b>(prepMemory) +
            prepMem +
            sizeof(SparseBlkDiag::Blk) * numBodyGroups +
            sizeof(float) * totalNumDofs +
            sizeof(float) * totalNumDofs +
            sizeof(float) * numContactPts;
        return (float *)bytes;
    }

    float * getJacobian(StateManager *state_mgr)
    {
        uint8_t *bytes =
            // (uint8_t *)state_mgr->memoryRangePointer<SolverScratch256b>(prepMemory) +
            prepMem +
            sizeof(SparseBlkDiag::Blk) * numBodyGroups +
            sizeof(float) * totalNumDofs +
            sizeof(float) * totalNumDofs +
            sizeof(float) * numContactPts +
            sizeof(float) * numContactPts;
        return (float *)bytes;
    }
#endif
};

namespace tasks {

// Computes the expanded parent array (based on Table 6.4 of Featherstone)
inline void computeExpandedParent(Context &ctx,
                                  BodyGroupHierarchy &body_grp) {
    CountT numBodies = body_grp.numBodies;

    int32_t *expandedParent = body_grp.getExpandedParent(ctx);

    // Initialize n-N_B elements
    for(int32_t i = 0; i < body_grp.numDofs; ++i) {
        expandedParent[i] = i - 1;
    }

    // Create a mapping from body index to start of block
    int32_t *map = (int32_t *)ctx.tmpAlloc(sizeof(int32_t) * numBodies);

    map[0] = -1;

    for(int32_t i = 1; i < numBodies; ++i) {
        uint32_t n_i = ctx.get<DofObjectNumDofs>(body_grp.bodies(ctx)[i]).numDofs;
        map[i] = map[i - 1] + (int32_t) n_i;
    }
    // Finish expanded parent array
    for(int32_t i = 1; i < numBodies; ++i) {
        int32_t parent_idx = ctx.get<DofObjectHierarchyDesc>(
            body_grp.bodies(ctx)[i]).parentIndex;
        expandedParent[map[i - 1] + 1] = map[parent_idx];
    }
}

inline void forwardKinematics(Context &ctx,
                              BodyGroupHierarchy &body_grp)
{
    { // Set the parent's state
        auto &position = ctx.get<DofObjectPosition>(body_grp.bodies(ctx)[0]);
        auto &tmp_state = ctx.get<DofObjectTmpState>(body_grp.bodies(ctx)[0]);

        tmp_state.composedRot = {
            position.q[3],
            position.q[4],
            position.q[5],
            position.q[6]
        };

        // This is the origin of the body
        tmp_state.comPos = {
            position.q[0],
            position.q[1],
            position.q[2]
        };

        // omega remains unchanged, and v only depends on the COM position
        tmp_state.phi.v[0] = tmp_state.comPos[0];
        tmp_state.phi.v[1] = tmp_state.comPos[1];
        tmp_state.phi.v[2] = tmp_state.comPos[2];
    }

    // Forward pass from parent to children
    for (int i = 1; i < body_grp.numBodies; ++i) {
        Entity body = body_grp.bodies(ctx)[i];
        auto &position = ctx.get<DofObjectPosition>(body);
        auto &num_dofs = ctx.get<DofObjectNumDofs>(body);
        auto &tmp_state = ctx.get<DofObjectTmpState>(body);
        auto &hier_desc = ctx.get<DofObjectHierarchyDesc>(body);

        float s = hier_desc.globalScale;

        Entity parent_e = hier_desc.parent;
        DofObjectTmpState &parent_tmp_state =
            ctx.get<DofObjectTmpState>(parent_e);

        // We can calculate our stuff.
        switch (num_dofs.type) {
        case DofType::Hinge: {
            // Find the hinge axis orientation in world space
            Vector3 rotated_hinge_axis =
                parent_tmp_state.composedRot.rotateVec(
                        hier_desc.parentToChildRot.rotateVec(hier_desc.axis));

            // Calculate the composed rotation applied to the child entity.
            tmp_state.composedRot = parent_tmp_state.composedRot *
                                    hier_desc.parentToChildRot *
                                    Quat::angleAxis(position.q[0], hier_desc.axis);

            // Calculate the composed COM position of the child
            //  (parent COM + R_{parent} * (rel_pos_parent + R_{hinge} * rel_pos_local))
            tmp_state.comPos = parent_tmp_state.comPos +
                s * parent_tmp_state.composedRot.rotateVec(
                        hier_desc.relPositionParent +
                        hier_desc.parentToChildRot.rotateVec(
                            Quat::angleAxis(position.q[0], hier_desc.axis).
                                rotateVec(hier_desc.relPositionLocal))
                );

            // All we are getting here is the position of the hinge point
            // which is relative to the parent's COM.
            tmp_state.anchorPos = parent_tmp_state.comPos +
                s * parent_tmp_state.composedRot.rotateVec(
                        hier_desc.relPositionParent);

            // Phi only depends on the hinge axis and the hinge point
            tmp_state.phi.v[0] = rotated_hinge_axis[0];
            tmp_state.phi.v[1] = rotated_hinge_axis[1];
            tmp_state.phi.v[2] = rotated_hinge_axis[2];
            tmp_state.phi.v[3] = tmp_state.anchorPos[0];
            tmp_state.phi.v[4] = tmp_state.anchorPos[1];
            tmp_state.phi.v[5] = tmp_state.anchorPos[2];
        } break;

        case DofType::Slider: {
            Vector3 rotated_axis =
                parent_tmp_state.composedRot.rotateVec(
                        hier_desc.parentToChildRot.rotateVec(hier_desc.axis));

            // The composed rotation for this body is the same as the parent's
            tmp_state.composedRot = parent_tmp_state.composedRot *
                                    hier_desc.parentToChildRot;

            tmp_state.comPos = parent_tmp_state.comPos +
                s * parent_tmp_state.composedRot.rotateVec(
                        hier_desc.relPositionParent +
                        hier_desc.parentToChildRot.rotateVec(
                            hier_desc.relPositionLocal +
                            position.q[0] * hier_desc.axis)
                );

            // This is the same as the comPos I guess?
            tmp_state.anchorPos = tmp_state.comPos;

            Vector3 axis = rotated_axis.normalize();

            tmp_state.phi.v[0] = axis[0];
            tmp_state.phi.v[1] = axis[1];
            tmp_state.phi.v[2] = axis[2];
        } break;

        case DofType::Ball: {
            Quat joint_rot = Quat{
                position.q[0], position.q[1], position.q[2], position.q[3]
            };

            // Calculate the composed rotation applied to the child entity.
            tmp_state.composedRot = parent_tmp_state.composedRot *
                                    hier_desc.parentToChildRot *
                                    joint_rot;

            // Calculate the composed COM position of the child
            //  (parent COM + R_{parent} * (rel_pos_parent + R_{ball} * rel_pos_local))
            tmp_state.comPos = parent_tmp_state.comPos +
                s * parent_tmp_state.composedRot.rotateVec(
                        hier_desc.relPositionParent +
                        hier_desc.parentToChildRot.rotateVec(
                            joint_rot.rotateVec(hier_desc.relPositionLocal))
                );

            // All we are getting here is the position of the ball point
            // which is relative to the parent's COM.
            tmp_state.anchorPos = parent_tmp_state.comPos +
                s * parent_tmp_state.composedRot.rotateVec(
                        hier_desc.relPositionParent);

            // Phi only depends on the hinge point and parent rotation
            tmp_state.phi.v[0] = tmp_state.anchorPos[0];
            tmp_state.phi.v[1] = tmp_state.anchorPos[1];
            tmp_state.phi.v[2] = tmp_state.anchorPos[2];
            tmp_state.phi.v[3] = parent_tmp_state.composedRot.w;
            tmp_state.phi.v[4] = parent_tmp_state.composedRot.x;
            tmp_state.phi.v[5] = parent_tmp_state.composedRot.y;
            tmp_state.phi.v[6] = parent_tmp_state.composedRot.z;
        } break;

        case DofType::FixedBody: {
            tmp_state.composedRot = 
                parent_tmp_state.composedRot;

            // This is the origin of the body
            tmp_state.comPos =
                parent_tmp_state.comPos +
                s * parent_tmp_state.composedRot.rotateVec(
                        hier_desc.relPositionParent +
                        hier_desc.parentToChildRot.rotateVec(
                            hier_desc.relPositionLocal)
                );

            // omega remains unchanged, and v only depends on the COM position
            tmp_state.phi.v[0] = tmp_state.comPos[0];
            tmp_state.phi.v[1] = tmp_state.comPos[1];
            tmp_state.phi.v[2] = tmp_state.comPos[2];
        } break;

        default: {
            // Only hinges have parents
            assert(false);
        } break;
        }
    }
}

// Init tasks
inline void initHierarchies(Context &ctx,
                            BodyGroupHierarchy &grp)
{
    CountT num_dofs = grp.numDofs;

    // We are going to do all dynamic allocations in one memory range to
    // minimize fragmentation.
    uint64_t required_bytes = 0;

    // Expanded parent array
    grp.expandedParentOffset = required_bytes;
    required_bytes += num_dofs * sizeof(int32_t);

    // Bias vector
    grp.biasOffset = (uint32_t)required_bytes;
    required_bytes += num_dofs * sizeof(float);

    // Mass matrix
    grp.massMatrixOffset = (uint32_t)required_bytes;
    required_bytes += num_dofs * num_dofs * sizeof(float);

    // LTDL mass matrix
    grp.massMatrixLTDLOffset = (uint32_t)required_bytes;
    required_bytes += num_dofs * num_dofs * sizeof(float);

    grp.dofPrefixSumOffset = (uint32_t)required_bytes;
    required_bytes += grp.numBodies * sizeof(uint32_t);

    // All the bodies' data
    uint32_t accum_dofs = 0;
    for (CountT j = 0; j < grp.numBodies; ++j) {
        Entity body = grp.bodies(ctx)[j];
        auto &num_body_dofs = ctx.get<DofObjectNumDofs>(body);
        auto &tmp_state = ctx.get<DofObjectTmpState>(body);

        tmp_state.phiFullOffset = (uint32_t)required_bytes;
        tmp_state.dofOffset = accum_dofs;

        accum_dofs += num_body_dofs.numDofs;

        // Space for both phi and phi_dot
        CountT num_phi_vals = 2 * 6 * num_body_dofs.numDofs;
        required_bytes += num_phi_vals * sizeof(float);
    }

    // Do memory range allocation now
    uint64_t num_elems = (required_bytes + sizeof(MRElement128b) - 1) /
        sizeof(MRElement128b);
    grp.dynData = ctx.allocMemoryRange<MRElement128b>(num_elems);

    uint32_t ps_offset = 0;
    uint32_t *prefix_sum = grp.getDofPrefixSum(ctx);
    for (CountT j = 0; j < grp.numBodies; ++j) {
        Entity body = grp.bodies(ctx)[j];
        auto &tmp_state = ctx.get<DofObjectTmpState>(body);
        auto &num_dofs = ctx.get<DofObjectNumDofs>(body);

        tmp_state.dynData = grp.dynData;
        prefix_sum[j] = ps_offset;

        ps_offset += num_dofs.numDofs;
    }

    // Now do some post-allocation computations
    tasks::computeExpandedParent(ctx, grp);

    // Forward kinematics to get positions
    tasks::forwardKinematics(ctx, grp);
}

inline Mat3x3 skewSymmetricMatrix(Vector3 v)
{
    return {
        {
            { 0.f, v.z, -v.y },
            { -v.z, 0.f, v.x },
            { v.y, -v.x, 0.f }
        }
    };
}

#ifdef MADRONA_GPU_MODE
inline void computePhiTrans(
        const DofObjectNumDofs num_dofs,
        DofObjectTmpState &tmp_state,
        Vector3 origin,
        float (&S)[18])
{
    Phi phi = tmp_state.phi;

    if (num_dofs.type == DofType::FreeBody) {
        // S = [1_3x3 r^x; 0 1_3x3], column-major
        // memset(S, 0.f, 6 * 3 * sizeof(float));

        // Diagonal identity
        #pragma unroll
        for(CountT i = 0; i < 3; ++i) {
            S[i * 3 + i] = 1.f;
        }
        // r^x Skew symmetric matrix
        Vector3 comPos = {phi.v[0], phi.v[1], phi.v[2]};
        comPos -= origin;
        S[0 + 3 * 4] = -comPos.z;
        S[0 + 3 * 5] = comPos.y;
        S[1 + 3 * 3] = comPos.z;
        S[1 + 3 * 5] = -comPos.x;
        S[2 + 3 * 3] = -comPos.y;
        S[2 + 3 * 4] = comPos.x;
    }
    else if (num_dofs.type == DofType::Hinge) {
        // S = [r \times hinge; hinge]
        Vector3 hinge = {phi.v[0], phi.v[1], phi.v[2]};
        Vector3 anchorPos = {phi.v[3], phi.v[4], phi.v[5]};
        anchorPos -= origin;
        Vector3 r_cross_hinge = anchorPos.cross(hinge);
        S[0] = r_cross_hinge.x;
        S[1] = r_cross_hinge.y;
        S[2] = r_cross_hinge.z;
        // S[3] = hinge.x;
        // S[4] = hinge.y;
        // S[5] = hinge.z;
    }
    else if (num_dofs.type == DofType::Ball) {
        // This will just get right-multiplied by the angular velocity
        Vector3 anchor_pos = {phi.v[0], phi.v[1], phi.v[2]};
        anchor_pos -= origin;

        // We need to right multiply these by the parent composed rotation
        // matrix.
        Mat3x3 rx = skewSymmetricMatrix(anchor_pos);
        Quat parent_composed_rot = Quat{
            phi.v[3], phi.v[4], phi.v[5], phi.v[6]
        };
        Mat3x3 parent_rot = Mat3x3::fromQuat(parent_composed_rot);

        rx *= parent_rot;

        #pragma unroll
        for (int col = 0; col < 3; ++col) {
            S[col * 3 + 0] = rx[col][0];
            S[col * 3 + 1] = rx[col][1];
            S[col * 3 + 2] = rx[col][2];
        }
    }
    else {
        MADRONA_UNREACHABLE();
    }
}
#else
inline void computePhiTrans(
        const DofObjectNumDofs num_dofs,
        DofObjectTmpState &tmp_state,
        Vector3 origin,
        float *S)
{
    Phi phi = tmp_state.phi;

    if (num_dofs.type == DofType::FreeBody) {
        // S = [1_3x3 r^x; 0 1_3x3], column-major
        memset(S, 0.f, 6 * 3 * sizeof(float));
        // Diagonal identity
        for(CountT i = 0; i < 3; ++i) {
            S[i * 3 + i] = 1.f;
        }
        // r^x Skew symmetric matrix
        Vector3 comPos = {phi.v[0], phi.v[1], phi.v[2]};
        comPos -= origin;
        S[0 + 3 * 4] = -comPos.z;
        S[0 + 3 * 5] = comPos.y;
        S[1 + 3 * 3] = comPos.z;
        S[1 + 3 * 5] = -comPos.x;
        S[2 + 3 * 3] = -comPos.y;
        S[2 + 3 * 4] = comPos.x;
    }
    else if (num_dofs.type == DofType::Slider) {
        S[0] = phi.v[0];
        S[1] = phi.v[1];
        S[2] = phi.v[2];
    }
    else if (num_dofs.type == DofType::Hinge) {
        // S = [r \times hinge; hinge]
        Vector3 hinge = {phi.v[0], phi.v[1], phi.v[2]};
        Vector3 anchorPos = {phi.v[3], phi.v[4], phi.v[5]};
        anchorPos -= origin;
        Vector3 r_cross_hinge = anchorPos.cross(hinge);
        S[0] = r_cross_hinge.x;
        S[1] = r_cross_hinge.y;
        S[2] = r_cross_hinge.z;
        // S[3] = hinge.x;
        // S[4] = hinge.y;
        // S[5] = hinge.z;
    }
    else if (num_dofs.type == DofType::Ball) {
        // This will just get right-multiplied by the angular velocity
        Vector3 anchor_pos = {phi.v[0], phi.v[1], phi.v[2]};
        anchor_pos -= origin;

        // We need to right multiply these by the parent composed rotation
        // matrix.
        Mat3x3 rx = skewSymmetricMatrix(anchor_pos);
        Quat parent_composed_rot = Quat{
            phi.v[3], phi.v[4], phi.v[5], phi.v[6]
        };
        Mat3x3 parent_rot = Mat3x3::fromQuat(parent_composed_rot);

        rx *= parent_rot;

        for (int col = 0; col < 3; ++col) {
            S[col * 3 + 0] = rx[col][0];
            S[col * 3 + 1] = rx[col][1];
            S[col * 3 + 2] = rx[col][2];
        }
    }
    else {
        // MADRONA_UNREACHABLE();
    }
}
#endif

#ifdef MADRONA_GPU_MODE
struct GaussMinimizationNode : NodeBase {
    GaussMinimizationNode(StateManager *state_mgr);

    // Make sure that res has shape sd->freeAccDim.
    // Make sure scratch has shape max(sd->freeAccDim, sd->numRowsJc)
    //
    // This is also going to return a couple other things to reduce
    // computation
    template <bool calc_package = false>
    void dobjWarp(
        float *res,
        float *x,
        CVSolveData *sd,
        float *scratch,
        float *jaccref,
        float *Mxmin,
        float *acc_ref,
        bool dbg = false);

    // Call this after calling dobjWarp
    float objWarp(
        float *x,
        CVSolveData *sd,
        float *jaccref,
        float *Mxmin);

    float exactLineSearch(
            CVSolveData *sd,
            float *jaccref,
            float *Mxmin,
            float *p,
            float *x,
            float tol,
            float *scratch,
            bool dbg = false);

    void computeContactJacobian(
            BodyGroupHierarchy *grp,
            DofObjectHierarchyDesc *hier_desc,
            Mat3x3 C,
            Vector3 origin,
            float *j_c,
            uint32_t body_dof_offset,
            uint32_t jac_row,
            uint32_t j_num_rows,
            float coeff,
            bool dbg);

    void prepareRegInfos(CVSolveData *sd);

    // Nodes in the taskgraph:
    void allocateScratch(int32_t invocation_idx);
    // Prepares mass matrix and contact jacobian
    void prepareSolver(int32_t invocation_idx);
    void computeAccRef(int32_t invocation_idx);
    void nonlinearCG(int32_t invocation_idx);




    // Let's test some of these helper functions
    void testNodeMul(int32_t invocation_idx);
    void testNodeTransposeMul(int32_t invocation_idx);
    void testNodeIdenMul(int32_t invocation_idx);
    void testNodeSparseMul(int32_t invocation_idx);
    void testWarpStuff(int32_t invocation_idx);

    static TaskGraph::NodeID addToGraph(
            TaskGraph::Builder &builder,
            Span<const TaskGraph::NodeID> dependencies);

    // Each `CVSolveData` contains the matrices / vectors we need.
    CVSolveData *solveDatas;
};

float GaussMinimizationNode::objWarp(
    float *x,
    CVSolveData *sd,
    float *jaccref,
    float *Mxmin)
{
    using namespace gpu_utils;

    // Need to calculate the following:
    // 0.5 * x_min_a_free.T @ (M @ x_min_a_free) + s(J @ x - a_ref)
    float res =
        0.5f * dotVectorsPred<float>(
            [&](uint32_t iter) {
                return x[iter] - sd->freeAcc[iter];
            },
            [&](uint32_t iter) {
                return Mxmin[iter];
            },
            sd->freeAccDim);

    // Calculate s(...) now:
    warpLoopSync(
        sd->numRowsJc / 3,
        [&](uint32_t iter) {
            float curr_val = 0.f;
            if (iter != 0xFFFF'FFFF) {
                float n = jaccref[iter * 3];
                float t1 = jaccref[iter * 3 + 1];
                float t2 = jaccref[iter * 3 + 2];

                float t = sqrtf(t1 * t1 + t2 * t2);
                float mu = sd->mu[iter];
                float mw = 1.f / (1.f + mu * mu);

                if (n >= mu * t) {
                    // Do nothing (top zone)
                } else if (mu * n + t <= 0.f) {
                    // Bottom zone
                    curr_val = 0.5f * (n * n + t1 * t1 + t2 * t2);
                } else {
                    // Middle zone
                    curr_val = 0.5f * mw * (n - mu * t) * (n - mu * t);
                }
            }

            res += warpReduceSum(curr_val);
        });

    return res;
}

template <bool calc_package>
void GaussMinimizationNode::dobjWarp(
        float *res,
        float *x,
        CVSolveData *sd,
        float *scratch,
        float *jaccref,
        float *Mxmin,
        float *acc_ref,
        bool dbg)
{
    using namespace gpu_utils;

    // x - acc_free
    warpLoop(sd->freeAccDim, [&](uint32_t iter) {
        scratch[iter] = x[iter] - sd->freeAcc[iter];
    });
    __syncwarp();

    sparseBlkDiagSmallReg<float, 4, false, false, true>(
            res,
            &sd->massSparse,
            scratch,
            sd->freeAccDim, 1);
    __syncwarp();

    if (dbg) {
        printMatrix(res, 1, sd->freeAccDim, "mxmin new");
    }

    // By now, res has M @ x_min_acc_free

    if constexpr (calc_package) {
        warpCopy(Mxmin, res, sd->freeAccDim * sizeof(float));
        __syncwarp();

        if (dbg) {
            printMatrix(Mxmin, 1, sd->freeAccDim, "mxmin new copied");
        }
    }

#if 0
     // Creating J @ x - acc_ref to feed to ds
    warpLoop(sd->numRowsJc, [&](uint32_t iter) {
        scratch[iter] = -acc_ref[iter];
    });
    __syncwarp();

    if (dbg) {
        printMatrix(scratch, 1, sd->numRowsJc, "-accref");
    }

    gmmaWarpSmallReg<float, 4, true, false, false>(
            scratch,
            sd->J_c,
            x,
            sd->numRowsJc,
            sd->numColsJc,
            sd->freeAccDim,
            1);
    __syncwarp();
#endif

    gmmaWarpSmallReg<float, 4, true, false, true>(
            scratch,
            sd->J_c,
            x,
            sd->numRowsJc,
            sd->numColsJc,
            sd->freeAccDim,
            1);
    __syncwarp();

    if (dbg) {
        printMatrix(Mxmin, 1, sd->freeAccDim, "after Jx");
    }

#if 0
    if (dbg) {
        printMatrix(scratch, 1, sd->numRowsJc, "J @ x");
    }
#endif

    warpLoop(sd->numRowsJc, [&](uint32_t iter) {
        scratch[iter] -= acc_ref[iter];
    });
    __syncwarp();

    if (dbg) {
        printMatrix(Mxmin, 1, sd->freeAccDim, "after Jx-aref");
    }

    if constexpr (calc_package) {
        if (dbg) {
            warp_printf("jaccref at %p; mxmin at %p\n", jaccref, Mxmin);
            warp_printf("memcpy of %d bytes\n", sd->numRowsJc * sizeof(float));
        }

        warpCopy(jaccref, scratch, sd->numRowsJc * sizeof(float), dbg);
        __syncwarp();
    }

    if (dbg) {
        printMatrix(Mxmin, 1, sd->freeAccDim, "after copy Jx-aref");
    }

#if 0
    if (dbg) {
        printMatrix(scratch, 1, sd->numRowsJc, "J@x - accref");
    }
#endif

    if (dbg) {
        printMatrix(Mxmin, 1, sd->freeAccDim, "mxmin new copied before ds");
    }

    // ds
    warpLoop(sd->numRowsJc / 3, [&](uint32_t iter) {
        float n = scratch[iter * 3];
        float t1 = scratch[iter * 3 + 1];
        float t2 = scratch[iter * 3 + 2];

        float t = sqrtf(t1 * t1 + t2 * t2);
        float mu = sd->mu[iter];
        float mw = 1.f / (1.f + mu * mu);

        if (n >= mu * t) {
            scratch[3 * iter] = 0.f;
            scratch[3 * iter + 1] = 0.f;
            scratch[3 * iter + 2] = 0.f;
        } else if (mu * n + t <= 0.f) {
            scratch[3 * iter] = n;
            scratch[3 * iter + 1] = t1;
            scratch[3 * iter + 2] = t2;
        } else {
            float tmp = mw * (n - mu * t);
            scratch[3 * iter] = tmp;

            if (t == 0.f) {
                printf("nan going to happen dobj!\n");
            }
            scratch[3 * iter + 1] = -tmp * mu * t1 / t;
            scratch[3 * iter + 2] = -tmp * mu * t2 / t;
        }
    });
    __syncwarp();
    // By now, we will have ds(...). Just need to multiply by J.T

    // Accumulate J.T @ ds(...) into res
    gmmaWarpSmallReg<float, 4, false, false, false>(
            res,
            sd->J_c,
            scratch,
            sd->numColsJc,
            sd->numRowsJc,
            sd->numRowsJc,
            1);
    __syncwarp();

    if (dbg) {
        printMatrix(Mxmin, 1, sd->freeAccDim, "mxmin new copied eof");
    }
}

GaussMinimizationNode::GaussMinimizationNode(
        StateManager *s)
    : solveDatas(s->getSingletonColumn<CVSolveData>())
{
}

// Let's test some of these helper functions
void GaussMinimizationNode::testNodeTransposeMul(int32_t invocation_idx)
{
    using namespace gpu_utils;

    uint32_t warp_id = threadIdx.x / 32;
    uint32_t lane_id = threadIdx.x % 32;

    if (invocation_idx == 0) {
        float *test_a_mat = nullptr;
        float *test_b_mat = nullptr;
        float *test_res_mat = nullptr;
        uint32_t a_mat_rows, a_mat_cols;
        uint32_t b_mat_rows, b_mat_cols;

        if (lane_id == 0) {
            a_mat_rows = 12;
            a_mat_cols = 16;

            b_mat_rows = 16;
            b_mat_cols = 18;

            test_a_mat = (float *)mwGPU::TmpAllocator::get().alloc(
                sizeof(float) * a_mat_rows * a_mat_cols);
            test_b_mat = (float *)mwGPU::TmpAllocator::get().alloc(
                sizeof(float) * b_mat_rows * b_mat_cols);
            test_res_mat = (float *)mwGPU::TmpAllocator::get().alloc(
                sizeof(float) * a_mat_rows * b_mat_cols);

            for (int j = 0; j < a_mat_cols; ++j) {
                for (int i = 0; i < a_mat_rows; ++i) {
                    //test_a_mat[i * a_mat_cols + j] = (i * a_mat_cols + j) % 5;
                    test_a_mat[i + j * a_mat_rows] = (i * a_mat_cols + j) % 5;
                }
            }

#if 0
            for (int i = 0; i < a_mat_rows; ++i) {
                for (int j = 0; j < a_mat_cols; ++j) {
                    test_a_mat[i * a_mat_cols + j] = (i * a_mat_cols + j) % 5;
                }
            }
#endif

            for (int i = 0; i < b_mat_rows; ++i) {
                for (int j = 0; j < b_mat_cols; ++j) {
                    test_b_mat[i * b_mat_cols + j] = (i * b_mat_cols + j) % 7;
                }
            }
        }

        __syncwarp();

        test_a_mat = (float *)__shfl_sync(0xFFFF'FFFF, (uint64_t)test_a_mat, 0);
        test_b_mat = (float *)__shfl_sync(0xFFFF'FFFF, (uint64_t)test_b_mat, 0);
        test_res_mat = (float *)__shfl_sync(0xFFFF'FFFF, (uint64_t)test_res_mat, 0);
        a_mat_rows = __shfl_sync(0xFFFF'FFFF, a_mat_rows, 0);
        a_mat_cols = __shfl_sync(0xFFFF'FFFF, a_mat_cols, 0);
        b_mat_rows = __shfl_sync(0xFFFF'FFFF, b_mat_rows, 0);
        b_mat_cols = __shfl_sync(0xFFFF'FFFF, b_mat_cols, 0);

        gmmaWarpSmallReg<float, 4, true, false, true>(
                test_res_mat,
                test_a_mat,
                test_b_mat,
                a_mat_rows,
                a_mat_cols,
                b_mat_rows,
                b_mat_cols);

        if (lane_id == 0) {
            for (int i = 0; i < a_mat_rows; ++i) {
                for (int j = 0; j < b_mat_cols; ++j) {
                    float v = test_res_mat[i * b_mat_cols + j];

                    printf("%f\t", v);
                }

                printf("\n");
            }
            printf("\n");
        }
    }
}

void GaussMinimizationNode::testNodeMul(int32_t invocation_idx)
{
    using namespace gpu_utils;

    uint32_t warp_id = threadIdx.x / 32;
    uint32_t lane_id = threadIdx.x % 32;

    if (warp_id == 0) {
        float *test_a_mat = nullptr;
        float *test_b_mat = nullptr;
        float *test_res_mat = nullptr;
        uint32_t a_mat_rows, a_mat_cols;
        uint32_t b_mat_rows, b_mat_cols;

        if (lane_id == 0) {
            a_mat_rows = 12;
            a_mat_cols = 16;

            b_mat_rows = 16;
            b_mat_cols = 18;

            test_a_mat = (float *)mwGPU::TmpAllocator::get().alloc(
                sizeof(float) * a_mat_rows * a_mat_cols);
            test_b_mat = (float *)mwGPU::TmpAllocator::get().alloc(
                sizeof(float) * b_mat_rows * b_mat_cols);
            test_res_mat = (float *)mwGPU::TmpAllocator::get().alloc(
                sizeof(float) * a_mat_rows * b_mat_cols);

            for (int i = 0; i < a_mat_rows; ++i) {
                for (int j = 0; j < a_mat_cols; ++j) {
                    test_a_mat[i * a_mat_cols + j] = (i * a_mat_cols + j) % 5;
                }
            }

            for (int i = 0; i < b_mat_rows; ++i) {
                for (int j = 0; j < b_mat_cols; ++j) {
                    test_b_mat[i * b_mat_cols + j] = (i * b_mat_cols + j) % 7;
                }
            }
        }

        __syncwarp();

        test_a_mat = (float *)__shfl_sync(0xFFFF'FFFF, (uint64_t)test_a_mat, 0);
        test_b_mat = (float *)__shfl_sync(0xFFFF'FFFF, (uint64_t)test_b_mat, 0);
        test_res_mat = (float *)__shfl_sync(0xFFFF'FFFF, (uint64_t)test_res_mat, 0);
        a_mat_rows = __shfl_sync(0xFFFF'FFFF, a_mat_rows, 0);
        a_mat_cols = __shfl_sync(0xFFFF'FFFF, a_mat_cols, 0);
        b_mat_rows = __shfl_sync(0xFFFF'FFFF, b_mat_rows, 0);
        b_mat_cols = __shfl_sync(0xFFFF'FFFF, b_mat_cols, 0);

        gmmaWarpSmallReg<float, 4, false, false, true>(
                test_res_mat,
                test_a_mat,
                test_b_mat,
                a_mat_rows,
                a_mat_cols,
                b_mat_rows,
                b_mat_cols);

        if (lane_id == 0) {
            for (int i = 0; i < a_mat_rows; ++i) {
                for (int j = 0; j < b_mat_cols; ++j) {
                    float v = test_res_mat[i * b_mat_cols + j];

                    printf("%f\t", v);
                }

                printf("\n");
            }
            printf("\n");
        }
    }
}

void GaussMinimizationNode::testNodeSparseMul(int32_t invocation_idx)
{
    using namespace gpu_utils;

    uint32_t warp_id = threadIdx.x / 32;
    uint32_t lane_id = threadIdx.x % 32;

    if (warp_id == 0) {
        const int32_t num_smem_bytes_per_warp =
            mwGPU::SharedMemStorage::numBytesPerWarp();
        auto smem_buf = (uint8_t *)mwGPU::SharedMemStorage::buffer +
                        num_smem_bytes_per_warp * warp_id;

        // The first block is 4x4, the second is 6x6
        SparseBlkDiag *sps = (SparseBlkDiag *)smem_buf;

        SparseBlkDiag::Blk *blk_array = (SparseBlkDiag::Blk *)
            (smem_buf + sizeof(SparseBlkDiag));
        float *blk1 = (float *)
            (smem_buf + sizeof(SparseBlkDiag) +
                        2 * sizeof(SparseBlkDiag::Blk));
        float *blk2 = blk1 + sizeof(float) * 5 * 5;

        uint32_t b_rows = 12;
        uint32_t b_cols = 1;

        float *test_res = nullptr;
        float *test_mat = nullptr;

        if (lane_id == 0) {
            test_res = (float *)mwGPU::TmpAllocator::get().alloc(
                sizeof(float) * b_rows * b_cols);
            test_mat = (float *)mwGPU::TmpAllocator::get().alloc(
                sizeof(float) * b_rows * b_cols);

            *sps = SparseBlkDiag {
                .fullDim = 12,
                .numBlks = 2,
                .blks = blk_array
            };

            sps->blks[0] = SparseBlkDiag::Blk {
                .dim = 5,
                .scratch = 0,
                .values = blk1,
                .ltdl = nullptr,
                .expandedParent = nullptr
            };

            sps->blks[1] = SparseBlkDiag::Blk {
                .dim = 7,
                .scratch = 0,
                .values = blk2,
                .ltdl = nullptr,
                .expandedParent = nullptr
            };

            // Fill in the values of blocks
#if 1
            for (int i = 0; i < 5*5; ++i) {
                sps->blks[0].values[i] = (float)i;
            }
            for (int i = 0; i < 7*7; ++i) {
                sps->blks[1].values[i] = (float)i;
            }
#endif

            for (int i = 0; i < b_rows; ++i) {
                for (int j = 0; j < b_cols; ++j) {
                    test_mat[i * b_cols + j] = (i * b_cols + j);
                }
            }
        }

        __syncwarp();

        test_res = (float *)__shfl_sync(0xFFFF'FFFF, (uint64_t)test_res, 0);
        test_mat = (float *)__shfl_sync(0xFFFF'FFFF, (uint64_t)test_mat, 0);

        // Now try to do the multiplication
        sparseBlkDiagSmallReg<float, 4, false, false, true>(
                test_res,
                sps,
                test_mat,
                b_rows, b_cols);
        __syncwarp();

        if (lane_id == 0) {
            for (int i = 0; i < b_rows; ++i) {
                for (int j = 0; j < b_cols; ++j) {
                    float v = test_res[i * b_cols + j];

                    printf("%f\t", v);
                }

                printf("\n");
            }
            printf("\n");
        }
    }
}

void GaussMinimizationNode::testWarpStuff(int32_t invocation_idx)
{
    using namespace gpu_utils;

    uint32_t value = threadIdx.x % 32;
    uint32_t sum = warpReduceSum(value);
    printf("sum = %u\n", sum);
}

void GaussMinimizationNode::testNodeIdenMul(int32_t invocation_idx)
{
    using namespace gpu_utils;

    uint32_t warp_id = threadIdx.x / 32;
    uint32_t lane_id = threadIdx.x % 32;

    if (warp_id == 0) {
        float *test_a_mat = nullptr;
        float *test_b_mat = nullptr;
        float *test_res_mat = nullptr;
        uint32_t a_mat_rows, a_mat_cols;
        uint32_t b_mat_rows, b_mat_cols;

        if (lane_id == 0) {
            a_mat_rows = 7;
            a_mat_cols = 7;

            b_mat_rows = 7;
            b_mat_cols = 6;

            test_a_mat = (float *)mwGPU::TmpAllocator::get().alloc(
                sizeof(float) * a_mat_rows * a_mat_cols);
            test_b_mat = (float *)mwGPU::TmpAllocator::get().alloc(
                sizeof(float) * b_mat_rows * b_mat_cols);
            test_res_mat = (float *)mwGPU::TmpAllocator::get().alloc(
                sizeof(float) * a_mat_rows * b_mat_cols);

            memset(test_a_mat, 0, sizeof(float) * a_mat_rows * a_mat_cols);

            for (int i = 0; i < a_mat_rows; ++i) {
                test_a_mat[i * a_mat_cols + i] = 1.f;
            }

            for (int i = 0; i < b_mat_rows; ++i) {
                for (int j = 0; j < b_mat_cols; ++j) {
                    test_b_mat[i * b_mat_cols + j] = (i * b_mat_cols + j);
                }
            }
        }

        __syncwarp();

        test_a_mat = (float *)__shfl_sync(0xFFFF'FFFF, (uint64_t)test_a_mat, 0);
        test_b_mat = (float *)__shfl_sync(0xFFFF'FFFF, (uint64_t)test_b_mat, 0);
        test_res_mat = (float *)__shfl_sync(0xFFFF'FFFF, (uint64_t)test_res_mat, 0);
        a_mat_rows = __shfl_sync(0xFFFF'FFFF, a_mat_rows, 0);
        a_mat_cols = __shfl_sync(0xFFFF'FFFF, a_mat_cols, 0);
        b_mat_rows = __shfl_sync(0xFFFF'FFFF, b_mat_rows, 0);
        b_mat_cols = __shfl_sync(0xFFFF'FFFF, b_mat_cols, 0);

        gmmaWarpSmallReg<float, 4, false, false, true>(
                test_res_mat,
                test_a_mat,
                test_b_mat,
                a_mat_rows,
                a_mat_cols,
                b_mat_rows,
                b_mat_cols);

        if (lane_id == 0) {
            for (int i = 0; i < a_mat_rows; ++i) {
                for (int j = 0; j < b_mat_cols; ++j) {
                    float v = test_res_mat[i * b_mat_cols + j];

                    printf("%f\t", v);
                }

                printf("\n");
            }
            printf("\n");
        }
    }
}

float GaussMinimizationNode::exactLineSearch(
        CVSolveData *sd,
        float *jaccref,
        float *Mxmin,
        float *p,
        float *x,
        float tol,
        float *scratch,
        bool dbg)
{
#define dbg_warp_printf(...) if (dbg) { warp_printf(__VA_ARGS__); }
#define dbg_matrix_printf(...) if (dbg) { printMatrix(__VA_ARGS__); }

    using namespace gpu_utils;

    float pMx_free = dotVectors(p, Mxmin, sd->freeAccDim);

    dbg_warp_printf("@@@@@@@@@@@@@ ENTERING FAUlTY LINE SEARCH\n");

    dbg_matrix_printf(p, 1, sd->freeAccDim, "p");
    dbg_matrix_printf(Mxmin, 1, sd->freeAccDim, "Mx_free");
    dbg_matrix_printf(jaccref, 1, sd->numRowsJc, "jaccref");

    dbg_warp_printf("pMx_free = %f\n", pMx_free);

    float xmin_M_xmin = dotVectorsPred<float>(
        [&](uint32_t iter) {
            return x[iter] - sd->freeAcc[iter];
        },
        [&](uint32_t iter) {
            return Mxmin[iter];
        },
        sd->freeAccDim);
    __syncwarp();

    dbg_warp_printf("xmin_M_xmin = %f\n", xmin_M_xmin);

    float pMp = 0.f;
    { // Ok, now we don't need to store Mxmin anymore
        float *Mp = Mxmin;

        sparseBlkDiagSmallReg<float, 4, false, false, true>(
                Mp, &sd->massSparse, p,
                sd->freeAccDim, 1);
        __syncwarp();

        pMp = dotVectors(p, Mp, sd->freeAccDim);
    }
    __syncwarp();

    dbg_warp_printf("pMp = %f\n", pMp);

    // Store J @ p in the scratch space we used for Mp
    float *Jp = Mxmin;
    { // Calculate Jp
        gmmaWarpSmallReg<float, 4, true, false, true>(
            Jp, sd->J_c, p,
            sd->numRowsJc, sd->numColsJc,
            sd->numColsJc, 1);
    }
    __syncwarp();

    // function, first deriv and second deriv evals
    struct Evals {
        float fun;
        float grad;
        float hess;
    };

   auto fdh_phi = [&](float alpha, bool print = false) -> Evals {
        float fun = 0.5f * alpha * alpha * pMp +
                    alpha * pMx_free + 0.5f * xmin_M_xmin;
        float grad = alpha * pMp + pMx_free;
        float hess = pMp;

        if (print) {
            warp_printf("fun0 = %f; grad0 = %f; hess0 = %f\n",
                    fun, grad, hess);
        }

        warpLoopSync(sd->numRowsJc / 3,
            [&](uint32_t iter) {
                struct Diff {
                    float dfun;
                    float dgrad;
                    float dhess;
                };

                auto d = [&]() -> Diff {
                    if (iter == 0xFFFF'FFFF) {
                        return {
                            0.f, 0.f, 0.f
                        };
                    } else {
                        float n = jaccref[iter * 3];
                        float t1 = jaccref[iter * 3 + 1];
                        float t2 = jaccref[iter * 3 + 2];
                        float mu = sd->mu[iter];
                        float mw = 1.f / (1.f + mu * mu);

                        float p0 = Jp[iter * 3];
                        float p1 = Jp[iter * 3 + 1];
                        float p2 = Jp[iter * 3 + 2];
                        float np = n + alpha * p0;
                        float t1p = t1 + alpha * p1;
                        float t2p = t2 + alpha * p2;
                        float tp = sqrtf(t1p * t1p + t2p * t2p);

                        if (np >= mu * tp) {
                            // Don't add anything up
                            return {0.f, 0.f, 0.f};

                            if (print)
                                printf("iter = %d; first return (0)\n", iter);
                        } else if (mu * np + tp <= 0.f) {
                            float p_sq = p0 * p0 + p1 * p1 + p2 * p2;

                            Diff diff = {
                                np * np + tp * tp,
                                p0 * n + p1 * t1 + p2 * t2 + alpha * p_sq,
                                p_sq,
                            };

                            if (print)
                                printf("iter = %d; second return; f=%f; g=%f; h=%f\n",
                                        iter, diff.dfun, diff.dgrad, diff.dhess);

                            return diff;
                        } else {
                            float dnp_da = p0;
                            float dtp_da = (p1 * t1 + p2 * t2 +
                                            alpha * (p1 * p1 + p2 * p2)) / tp;
                            if (tp == 0.f) {
                                printf("nan going to happen exactLineSearch tp\n");
                            }
                            float d2tp_da2 = ((p2 * t1 - p1 * t2) * (p2 * t1 - p1 * t2)) /
                                (tp * tp * tp);
                            if (tp * tp * tp == 0.f) {
                                printf("nan going to happen exactLineSearch tp^3\n");
                            }
                            float tmp = np - mu * tp;
                            float d_tmp = dnp_da - mu * dtp_da;

                            Diff diff = {
                                mw * tmp * tmp,
                                mw * tmp * d_tmp,
                                mw * (d_tmp * d_tmp + tmp * (-mu * d2tp_da2))
                            };

                            if (print)
                                printf("iter = %d; second return; f=%f; g=%f; h=%f\n",
                                        iter, diff.dfun, diff.dgrad, diff.dhess);

                            return diff;
                        }
                    }
                } ();

                // These are summed from the current iteration
                float dfun = warpReduceSum(d.dfun);
                float dgrad = warpReduceSum(d.dgrad);
                float dhess = warpReduceSum(d.dhess);

                // Now, fun, grad and hess will contain the full sum.
                fun += dfun;
                grad += dgrad;
                hess += dhess;
            });

        if (print) {
            warp_printf("final: fun=%f; grad=%f; hess=%f\n", fun, grad, hess);
        }

        return Evals {
            fun, grad, hess
        };
    };

    float alpha = 0.f;
    Evals evals_alpha = fdh_phi(alpha);

    // Newton step
    float alpha1 = alpha - evals_alpha.grad / evals_alpha.hess;
    if (evals_alpha.hess == 0.f) {
        printf("nan going to happen exactLineSeaarch hess\n");
    }

    Evals evals_alpha1 = fdh_phi(alpha1);

    if (evals_alpha.fun < evals_alpha1.fun) {
        alpha1 = alpha;
    }

    evals_alpha1 = fdh_phi(alpha1, dbg);

    if (dbg) {
        dbg_warp_printf("first return\n");
        dbg_warp_printf("d_alpha1 = %f\n", evals_alpha1.grad);
        dbg_warp_printf("tol = %f\n", tol);
        dbg_warp_printf("alpha1 = %f\n", alpha1);
    }

    // Initial convergence
    if (fabs(evals_alpha1.grad) < tol) {
        return alpha1;
    }

    float a_dir = (evals_alpha1.grad < 0.f) ? 1.f : -1.f;

    // Line search iterations
    uint32_t ls_iters = 50;
    uint32_t iters = 0;
    for (; iters < ls_iters; ++iters) {
        __syncwarp();

        evals_alpha1 = fdh_phi(alpha1);

        if (evals_alpha1.grad * a_dir > -tol)
            break;
        if (fabs(evals_alpha1.grad)  < tol) {
            dbg_warp_printf("second return\n");
            return alpha1;
        }

        alpha1 -= evals_alpha1.grad / evals_alpha1.hess;
        if (evals_alpha1.hess == 0.f) {
            printf("nan going to happen exactLineSeaarch hess1\n");
        }
    }

    if (iters == ls_iters) {
        // Failed to bracket...
        dbg_warp_printf("third return\n");
        return alpha1;
    }

    float alpha_low = alpha1;
    float alpha_high = alpha1 - evals_alpha1.grad / evals_alpha1.hess;
    if (evals_alpha1.hess == 0.f) {
        printf("nan going to happen exactLineSeaarch hess2\n");
    }

    Evals evals_alpha_mid = fdh_phi(alpha_low);
    if (evals_alpha_mid.grad > 0.f) {
        std::swap(alpha_low, alpha_high);
    }

    float alpha_mid;

    for (iters = 0; iters < ls_iters; ++iters) {
        alpha_mid = 0.5f * (alpha_low + alpha_high);
        evals_alpha_mid = fdh_phi(alpha_mid);

        if (fabs(evals_alpha_mid.grad) < tol) {
            dbg_warp_printf("fourth return\n");
            return alpha_mid;
        }

        if (evals_alpha_mid.grad > 0.f)
            alpha_high = alpha_mid;
        else
            alpha_low = alpha_mid;

        if (fabs(alpha_high - alpha_low) < tol) {
            dbg_warp_printf("fifth return\n");
            return alpha_mid;
        }
    }

    if (iters >= ls_iters) {
        // Failed to converge...
        dbg_warp_printf("sixth return\n");
        return alpha_mid;
    }

    MADRONA_UNREACHABLE();
}

void GaussMinimizationNode::allocateScratch(int32_t invocation_idx)
{
    if (threadIdx.x % 32 == 0) {
        const int32_t num_smem_bytes_per_warp =
            mwGPU::SharedMemStorage::numBytesPerWarp();

        // We want to fit as much data as possible into shared memory
        uint32_t world_id = invocation_idx;
        CVSolveData *curr_sd = &solveDatas[world_id];

        uint32_t max_num_comps = max(curr_sd->freeAccDim, curr_sd->numRowsJc);

        uint32_t num_sizes = 0;

        CVSolveData::RegInfo *sizes = curr_sd->regInfos;

        // x
        sizes[num_sizes++] = { sizeof(float) * curr_sd->freeAccDim, false };
        // m_grad
        sizes[num_sizes++] = { sizeof(float) * curr_sd->freeAccDim, false };
        // p
        sizes[num_sizes++] = { sizeof(float) * curr_sd->freeAccDim, false };
        // scratch1
        sizes[num_sizes++] = { sizeof(float) * max_num_comps, false };
        // scratch2
        sizes[num_sizes++] = { sizeof(float) * max_num_comps, false };
        // scratch3
        sizes[num_sizes++] = { sizeof(float) * max_num_comps, false };
        // scratch4
        sizes[num_sizes++] = { sizeof(float) * max_num_comps, false };
        // scratch5
        sizes[num_sizes++] = { sizeof(float) * max_num_comps, false };

        assert(num_sizes == CVSolveData::kNumRegisters);

        uint32_t total_size = 0;
        uint32_t size_in_glob_mem = 0;
        for (uint32_t i = 0; i < num_sizes; ++i) {
            total_size += sizes[i].size;

            if (total_size < num_smem_bytes_per_warp) {
                sizes[i].inSmem = true;
            } else {
                size_in_glob_mem += sizes[i].size;
            }
        }

        StateManager *state_mgr = getStateManager();

        curr_sd->solverScratchMem = (uint8_t *)mwGPU::TmpAllocator::get().alloc(
                size_in_glob_mem);
        curr_sd->scratchAllocatedBytes = size_in_glob_mem;

#if 0
        // Everything that isn't in shared memory, will have to go into a memory range.
        if (curr_sd->scratchAllocatedBytes == 0 && size_in_glob_mem != 0) {
            // Allocate the memory range
            curr_sd->solverScratchMemory = state_mgr->allocMemoryRange(
                    TypeTracker::typeID<SolverScratch256b>(),
                    (size_in_glob_mem + sizeof(SolverScratch256b)-1) /
                        sizeof(SolverScratch256b));
            curr_sd->scratchAllocatedBytes = size_in_glob_mem;
        } else if (curr_sd->scratchAllocatedBytes < size_in_glob_mem) {
            state_mgr->freeMemoryRange(curr_sd->solverScratchMemory);

            curr_sd->solverScratchMemory = state_mgr->allocMemoryRange(
                    TypeTracker::typeID<SolverScratch256b>(),
                    (size_in_glob_mem + sizeof(SolverScratch256b)) /
                        sizeof(SolverScratch256b));
            curr_sd->scratchAllocatedBytes = size_in_glob_mem;
        }
#endif


        { // AccRef allocation
            uint32_t acc_ref_bytes = sizeof(float) * curr_sd->numRowsJc;

            curr_sd->accRefMem = (uint8_t *)mwGPU::TmpAllocator::get().alloc(
                    acc_ref_bytes);
            curr_sd->accRefAllocatedBytes = acc_ref_bytes;
#if 0
            if (curr_sd->accRefAllocatedBytes == 0 && curr_sd->numRowsJc != 0) {
                curr_sd->accRefMemory = state_mgr->allocMemoryRange(
                    TypeTracker::typeID<SolverScratch256b>(),
                    (acc_ref_bytes + sizeof(SolverScratch256b)-1) /
                        sizeof(SolverScratch256b));
                curr_sd->accRefAllocatedBytes = acc_ref_bytes;
            } else if (curr_sd->accRefAllocatedBytes < acc_ref_bytes) {
                state_mgr->freeMemoryRange(curr_sd->accRefMemory);

                curr_sd->accRefMemory = state_mgr->allocMemoryRange(
                        TypeTracker::typeID<SolverScratch256b>(),
                        (acc_ref_bytes + sizeof(SolverScratch256b)) /
                            sizeof(SolverScratch256b));
                curr_sd->accRefAllocatedBytes = acc_ref_bytes;
            }
#endif
        }

        { // Mass matrix allocation
            CountT num_grps = state_mgr->numRows<BodyGroup>(world_id);
            BodyGroupHierarchy *hiers = state_mgr->getWorldComponents<
                BodyGroup, BodyGroupHierarchy>(world_id);
            ContactConstraint *contacts = state_mgr->getWorldComponents<
                Contact, ContactConstraint>(world_id);
            CountT num_contacts = state_mgr->numRows<Contact>(world_id);

            uint32_t total_num_dofs = 0;
            for (int i = 0; i < num_grps; ++i) {
                total_num_dofs += hiers[i].numDofs;
            }

            uint32_t total_contact_pts = 0;
            for (int i = 0; i < num_contacts; ++i) {
                total_contact_pts += contacts[i].numPoints;
            }

            uint32_t prep_bytes =
                // Mass matrix
                sizeof(SparseBlkDiag::Blk) * num_grps +
                // full vel
                sizeof(float) * total_num_dofs +
                // free acc
                sizeof(float) * total_num_dofs +
                // mu
                sizeof(float) * total_contact_pts +
                // penetrations
                sizeof(float) * total_contact_pts +
                // TODO: Make this sparse
                sizeof(float) * 3 * total_contact_pts * total_num_dofs;

            curr_sd->prepMem = (uint8_t *)mwGPU::TmpAllocator::get().alloc(
                    prep_bytes);
            curr_sd->prepAllocatedBytes = prep_bytes;

#if 0
            if (curr_sd->prepAllocatedBytes == 0) {
                curr_sd->prepMemory = state_mgr->allocMemoryRange(
                    TypeTracker::typeID<SolverScratch256b>(),
                    (prep_bytes + sizeof(SolverScratch256b)-1) /
                        sizeof(SolverScratch256b));
                curr_sd->prepAllocatedBytes = prep_bytes;
            } else if (curr_sd->prepAllocatedBytes < prep_bytes) {
                state_mgr->freeMemoryRange(curr_sd->prepMemory);

                curr_sd->prepMemory = state_mgr->allocMemoryRange(
                        TypeTracker::typeID<SolverScratch256b>(),
                        (prep_bytes + sizeof(SolverScratch256b)) /
                            sizeof(SolverScratch256b));
                curr_sd->prepAllocatedBytes = prep_bytes;
            }
#endif
        }
    }
}

void GaussMinimizationNode::prepareSolver(int32_t invocation_idx)
{
    using namespace gpu_utils;

    uint32_t warp_id = threadIdx.x / 32;
    uint32_t lane_id = threadIdx.x % 32;

    uint32_t world_id = invocation_idx;

    StateManager *state_mgr = getStateManager();

    CVSolveData *curr_sd = &solveDatas[world_id];

    BodyGroupHierarchy *hiers = state_mgr->getWorldComponents<
        BodyGroup, BodyGroupHierarchy>(world_id);

    { // Get the total mass in the world
        float total_mass = 0.f;

        BodyGroupHierarchy *hiers = state_mgr->getWorldComponents<
            BodyGroup, BodyGroupHierarchy>(world_id);
        uint32_t num_masses = state_mgr->numRows<BodyGroup>(world_id);

        total_mass = warpSumPred<float>(
            [&](uint32_t iter) -> float {
                return hiers[iter].inertiaSum;
            }, num_masses);

        if (lane_id == 0) {
            curr_sd->totalMass = total_mass;
        }
    }

    { // Prepare the mass matrix
        SparseBlkDiag mass_sparse;
        mass_sparse.fullDim = curr_sd->totalNumDofs;
        mass_sparse.numBlks = curr_sd->numBodyGroups;
        mass_sparse.blks = curr_sd->getMassBlks(state_mgr);

        warpLoop(curr_sd->numBodyGroups,
                [&](uint32_t i) {
                    mass_sparse.blks[i].dim = hiers[i].numDofs;
                    mass_sparse.blks[i].values = (float *)(
                        (uint8_t *)state_mgr->memoryRangePointer<MRElement128b>(hiers[i].dynData) +
                        hiers[i].massMatrixOffset);
                    mass_sparse.blks[i].ltdl = (float*)(
                        (uint8_t *)state_mgr->memoryRangePointer<MRElement128b>(hiers[i].dynData) +
                        hiers[i].massMatrixLTDLOffset);
                    mass_sparse.blks[i].expandedParent = (int32_t *)(
                        (uint8_t *)state_mgr->memoryRangePointer<MRElement128b>(hiers[i].dynData) +
                        hiers[i].expandedParentOffset);
                });
        __syncwarp();

        if (lane_id == 0) {
            curr_sd->massSparse = mass_sparse;
        }
    }

    { // Prepare free acc
        float *free_acc = curr_sd->getFreeAcc(state_mgr);

        uint32_t processed_dofs = 0;
        for (uint32_t body = 0; body < curr_sd->numBodyGroups; ++body) {
            __syncwarp();

            float *local_free_acc = (float *)(
                (uint8_t *)state_mgr->memoryRangePointer<MRElement128b>(hiers[body].dynData) +
                hiers[body].biasOffset);

            warpLoop(hiers[body].numDofs,
                    [&](uint32_t i) {
                        free_acc[processed_dofs + i] = local_free_acc[i];
                    });

            processed_dofs += hiers[body].numDofs;
        }

        if (lane_id == 0) {
            curr_sd->freeAcc = free_acc;
        }
    }

    { // Prepare full velocity
        float *full_vel = curr_sd->getFullVel(state_mgr);

        uint32_t processed_dofs = 0;
        for (uint32_t grp = 0; grp < curr_sd->numBodyGroups; ++grp) {
            Entity *bodies = (Entity *)
                (state_mgr->memoryRangePointer<MRElement128b>(hiers[grp].mrBodies));

            warpLoop(hiers[grp].numBodies,
                [&](uint32_t iter) {
                    Entity body = bodies[iter];

                    DofObjectVelocity vel = state_mgr->getUnsafe<DofObjectVelocity>(body);
                    uint32_t dof_offset = state_mgr->getUnsafe<DofObjectTmpState>(body).dofOffset;
                    uint32_t num_dofs = state_mgr->getUnsafe<DofObjectNumDofs>(body).numDofs;

                    #pragma unroll
                    for (uint32_t i = 0; i < 6; ++i) {
                        if (i < num_dofs)
                            full_vel[processed_dofs + dof_offset + i] = vel.qv[i];
                    }
                });

            __syncwarp();
            processed_dofs += hiers[grp].numDofs;
        }

        if (lane_id == 0) {
            curr_sd->vel = full_vel;
        }
    }

    uint32_t num_contacts = state_mgr->numRows<Contact>(world_id);

    ContactConstraint *contacts = state_mgr->getWorldComponents<
        Contact, ContactConstraint>(world_id);
    ContactTmpState *contacts_tmp_state = state_mgr->getWorldComponents<
        Contact, ContactTmpState>(world_id);

    { // Prepare mu
        float *full_mu = curr_sd->getMu(state_mgr);
        float *full_penetrations = curr_sd->getPenetrations(state_mgr);

        uint32_t processed_pts = 0;

        warpLoopSync(
            num_contacts,
            [&](uint32_t iter) {
                ContactTmpState tmp_state = (iter == 0xFFFF'FFFF) ?
                    ContactTmpState{} : contacts_tmp_state[iter];
                ContactConstraint contact = (iter == 0xFFFF'FFFF) ?
                    ContactConstraint{} : contacts[iter];

                uint32_t num_pts_ipf = warpInclusivePrefixSum(
                    (iter == 0xFFFF'FFFF) ? 0 : contact.numPoints);

                if (iter != 0xFFFF'FFFF) {
                    uint32_t offset = processed_pts + num_pts_ipf - contact.numPoints;
                    for (int i = 0; i < contact.numPoints; ++i) {
                        full_mu[offset + i] = tmp_state.mu;
                        full_penetrations[offset + i] = contact.points[i].w;
                    }
                }

                processed_pts += __shfl_sync(0xFFFF'FFFF, num_pts_ipf, 31);
            });

        curr_sd->mu = full_mu;
        curr_sd->penetrations = full_penetrations;
    }

#if 1
    { // Prepare the contact jacobian
        const int32_t num_smem_bytes_per_warp =
            mwGPU::SharedMemStorage::numBytesPerWarp();
        auto smem_buf = (uint8_t *)mwGPU::SharedMemStorage::buffer +
                        num_smem_bytes_per_warp * warp_id;

        uint32_t *world_block_start = (uint32_t *)smem_buf;
        assert(curr_sd->numBodyGroups * sizeof(uint32_t) < num_smem_bytes_per_warp);

        { // Quick prefix sum
            uint32_t accum = 0;
            if (lane_id == 0) {
                for (uint32_t i = 0; i < curr_sd->numBodyGroups; ++i) {
                    world_block_start[i] = accum;
                    accum += hiers[i].numDofs;
                    hiers[i].tmpIdx0 = i;
                }
            }
            __syncwarp();
        }

        float *j_c = curr_sd->getJacobian(state_mgr);
        curr_sd->J_c = j_c;

        warpSetZero(j_c, sizeof(float) *
                         3 * curr_sd->numContactPts *
                         curr_sd->totalNumDofs);

        // TODO: Maybe experiment with different axes of parallelism
        // Right now, trying to parallelize over contacts
        struct ContactInfo {
            ContactConstraint contact;
            ContactTmpState tmpState;
            Entity ref;
            Entity alt;
            uint32_t refNumDofs;
            uint32_t altNumDofs;
            bool refFixed;
            bool altFixed;
            uint32_t numPoints;
            float jaccCoeff;
        };

        auto get_contact_info = [&](uint32_t ct_idx) -> ContactInfo {
            auto contact = contacts[ct_idx];

            Entity ref = state_mgr->getUnsafe<LinkParentDofObject>(
                    contact.ref).parentDofObject;
            Entity alt = state_mgr->getUnsafe<LinkParentDofObject>(
                    contact.alt).parentDofObject;

            auto ref_num_dofs = state_mgr->getUnsafe<DofObjectNumDofs>(
                    ref).numDofs;
            auto alt_num_dofs = state_mgr->getUnsafe<DofObjectNumDofs>(
                    alt).numDofs;

            return {
                contact,
                contacts_tmp_state[ct_idx],
                ref,
                alt,
                ref_num_dofs,
                alt_num_dofs,
                (ref_num_dofs == 0),
                (alt_num_dofs == 0),
                contacts[ct_idx].numPoints
            };
        };

        // uint32_t jacc_row = 0;
        uint32_t processed_pts = 0;

        warpLoopSync(
            num_contacts,
            [&](uint32_t ct_idx) {
                ContactInfo c_info = (ct_idx == 0xFFFF'FFFF) ?
                    ContactInfo {} : get_contact_info(ct_idx);

                uint32_t wave_pt_ips = warpInclusivePrefixSum(c_info.contact.numPoints);
                uint32_t wave_num_pts = __shfl_sync(0xFFFF'FFFF, wave_pt_ips, 31);

                uint32_t curr_pt = processed_pts + wave_pt_ips - c_info.contact.numPoints;

                processed_pts += wave_num_pts;

                #pragma unroll
                for (uint32_t pt_idx = 0; pt_idx < ContactConstraint::kMaxPoints; ++pt_idx) {
                    uint32_t curr_jacc_row = (curr_pt + pt_idx) * 3;

                    if (pt_idx < c_info.contact.numPoints) {
                        Vector3 contact_pt = c_info.contact.points[pt_idx].xyz();

                        if (!c_info.refFixed) {
                            DofObjectHierarchyDesc *hier =
                                &state_mgr->getUnsafe<DofObjectHierarchyDesc>(
                                    c_info.ref);
                            BodyGroupHierarchy *grp =
                                &state_mgr->getUnsafe<BodyGroupHierarchy>(hier->bodyGroup);

                            computeContactJacobian(
                                    grp,
                                    hier,
                                    c_info.tmpState.C,
                                    contact_pt,
                                    j_c,
                                    world_block_start[grp->tmpIdx0],
                                    curr_jacc_row,
                                    curr_sd->numRowsJc,
                                    -1.f,
                                    false);//(ct_idx == 0 && pt_idx == 0));
                        }

                        if (!c_info.altFixed) {
                            DofObjectHierarchyDesc *hier =
                                &state_mgr->getUnsafe<DofObjectHierarchyDesc>(
                                    c_info.alt);
                            BodyGroupHierarchy *grp =
                                &state_mgr->getUnsafe<BodyGroupHierarchy>(hier->bodyGroup);

                            computeContactJacobian(
                                    grp,
                                    hier,
                                    c_info.tmpState.C,
                                    contact_pt,
                                    j_c,
                                    world_block_start[grp->tmpIdx0],
                                    curr_jacc_row,
                                    curr_sd->numRowsJc,
                                    1.f,
                                    false);//(ct_idx == 0 && pt_idx == 0));
                        }
                    }
                }
            });
    }
#endif
}

void GaussMinimizationNode::computeContactJacobian(
        BodyGroupHierarchy *grp,
        DofObjectHierarchyDesc *hier_desc,
        Mat3x3 C,
        Vector3 origin,
        float *j_c,
        uint32_t body_dof_offset,
        uint32_t jac_row,
        uint32_t j_num_rows,
        float coeff,
        bool dbg)
{
    StateManager *state_mgr = mwGPU::getStateManager();

    Entity *bodies = (Entity *)
        (state_mgr->memoryRangePointer<MRElement128b>(grp->mrBodies));
    uint32_t *block_start = (uint32_t *)
        ((uint8_t *)state_mgr->memoryRangePointer<MRElement128b>(grp->dynData) +
         grp->dofPrefixSumOffset);

    int32_t curr_idx = hier_desc->index;
    while(curr_idx != -1) {
        Entity body = bodies[curr_idx];

        auto &curr_tmp_state =
            state_mgr->getUnsafe<DofObjectTmpState>(body);
        auto &curr_num_dofs =
            state_mgr->getUnsafe<DofObjectNumDofs>(body);
        auto &curr_hier_desc =
            state_mgr->getUnsafe<DofObjectHierarchyDesc>(body);

        // Populate columns of J_C
        float S[18] = {};
        computePhiTrans(
                curr_num_dofs, curr_tmp_state, origin, S);

        // Only use translational part of S
        for(CountT i = 0; i < curr_num_dofs.numDofs; ++i) {
            float *J_col = j_c +
                j_num_rows * (body_dof_offset + block_start[curr_idx] + i) +
                jac_row;

            #pragma unroll
            for(CountT j = 0; j < 3; ++j) {
                J_col[j] = S[3 * i + j];
            }
        }

        curr_idx = curr_hier_desc.parentIndex;
    }


    // Multiply by C^T to project into contact space
    for(CountT i = 0; i < grp->numDofs; ++i) {
        float *J_col = j_c +
                j_num_rows * (body_dof_offset + i) +
                jac_row;

        Vector3 J_col_vec = { J_col[0], J_col[1], J_col[2] };

        J_col_vec = C.transpose() * J_col_vec;

        J_col[0] = coeff * J_col_vec.x;
        J_col[1] = coeff * J_col_vec.y;
        J_col[2] = coeff * J_col_vec.z;

#if 0
        if (dbg) {
            printf("(not working) (row=%d, col=%d) dof=%d, J_col=(%f %f %f)\n",
                    jac_row, body_dof_offset + i,
                    i, J_col[0], J_col[1], J_col[2]);
        }
#endif
    }
}

// Might be overkill to allocate a warp per world but we can obviously
// experiment.
void GaussMinimizationNode::computeAccRef(int32_t invocation_idx)
{
    using namespace gpu_utils;

    uint32_t total_resident_warps = (blockDim.x * gridDim.x) / 32;
    uint32_t total_num_worlds = mwGPU::GPUImplConsts::get().numWorlds;

    // This is the global warp ID
    // uint32_t warp_id = (blockDim.x * blockIdx.x + threadIdx.x) / 32;
    uint32_t warp_id = threadIdx.x / 32;
    uint32_t lane_id = threadIdx.x % 32;

    assert(blockDim.x == consts::numMegakernelThreads);

    StateManager *state_mgr = getStateManager();

    uint32_t world_id = invocation_idx;

    { // Do the actual computation
        CVSolveData *curr_sd = &solveDatas[world_id];

        // We want acc_ref to have priority in shared memory
        auto [acc_ref, in_smem] = [&]() -> std::pair<float *, bool> {
            const int32_t num_smem_bytes_per_warp =
                mwGPU::SharedMemStorage::numBytesPerWarp();

            uint32_t acc_ref_bytes = sizeof(float) * curr_sd->numRowsJc;
            auto smem_buf = (uint8_t *)mwGPU::SharedMemStorage::buffer +
                            num_smem_bytes_per_warp * warp_id;

            if (acc_ref_bytes == 0) {
                return { (float *)nullptr, false };
            } else if (acc_ref_bytes < num_smem_bytes_per_warp) {
                return { (float *)smem_buf, true };
            } else {
                return {
#if 0
                    (float *)state_mgr->memoryRangePointer<
                        SolverScratch256b>(curr_sd->accRefMemory),
#endif
                    (float *)curr_sd->accRefMem,
                    false
                };
            }
        } ();

        if (curr_sd->numRowsJc > 0) {
            float time_const = 2.f * curr_sd->h;
            constexpr float damp_ratio = 1.f;
            constexpr float d_min = 0.9f,
                            d_max = 0.95f,
                            width = 0.001f,
                            mid = 0.5f,
                            power = 2.f;

            // First store J @ v
            gmmaWarpSmallReg<float, 4, true, false, true>(
                    acc_ref,
                    curr_sd->J_c,
                    curr_sd->vel,
                    curr_sd->numRowsJc,
                    curr_sd->numColsJc,
                    curr_sd->velDim,
                    1);

            warpLoop(curr_sd->numRowsJc, [&](uint32_t iter) {
                float r = (iter % 3 == 0) ?
                    -curr_sd->penetrations[iter / 3] : 0.f;
                float imp_x = fabs(r) / width;
                float imp_a = (1.f / powf(mid, power-1.f)) * powf(imp_x, power);
                float imp_b = 1.f - (1.f / powf(1.f - mid, power - 1)) *
                              powf(1.f - imp_x, power);
                float imp_y = (imp_x < mid) ? imp_a : imp_b;
                float imp = d_min + imp_y * (d_max - d_min);
                if (imp < d_min)
                    imp = d_min;
                else if (imp > d_max)
                    imp = d_max;
                imp = (imp_x > 1.f) ? d_max : imp;

                float k = 1.f / (d_max * d_max *
                                 time_const * time_const *
                                 damp_ratio * damp_ratio);
                float b = 2.f / (d_max * time_const);

                acc_ref[iter] *= -b;
                acc_ref[iter] -= k * imp * r;
            });

            if (acc_ref && in_smem) {
                float * acc_ref_glob =
#if 0
                    (float *)state_mgr->memoryRangePointer<SolverScratch256b>(
                            curr_sd->accRefMemory);
#endif
                    (float *)curr_sd->accRefMem;

                warpCopy(acc_ref_glob, acc_ref,
                         sizeof(float) * curr_sd->numRowsJc);
                acc_ref = (float *)acc_ref_glob;
            }
        }

        __syncwarp();
    }
}

void GaussMinimizationNode::prepareRegInfos(CVSolveData *sd)
{
    const int32_t num_smem_bytes_per_warp =
        mwGPU::SharedMemStorage::numBytesPerWarp();

    uint32_t warp_id = threadIdx.x / 32;

    auto smem_buf = (uint8_t *)mwGPU::SharedMemStorage::buffer +
                    num_smem_bytes_per_warp * warp_id;

    StateManager *state_mgr = getStateManager();

    if (threadIdx.x % 32 == 0) {
        uint32_t current_smem_size = 0;
        uint32_t current_mr_size = 0;

        uint32_t curr_reg = 0;

        uint8_t *mr_mem = [&]() -> uint8_t * {
            if (sd->scratchAllocatedBytes == 0) {
                return (uint8_t *)nullptr;
            } else {
#if 0
                return (uint8_t *)state_mgr->memoryRangePointer<
                    SolverScratch256b>(sd->solverScratchMemory);
#endif
                return (uint8_t *)sd->solverScratchMem;
            }
        } ();

        for (; curr_reg < CVSolveData::kNumRegisters; ++curr_reg) {
            if (sd->regInfos[curr_reg].inSmem) {
                sd->regInfos[curr_reg].ptr = smem_buf + current_smem_size;
                current_smem_size += sd->regInfos[curr_reg].size;
            } else {
                sd->regInfos[curr_reg].ptr = mr_mem + current_mr_size;
                current_mr_size += sd->regInfos[curr_reg].size;
            }
        }
    }

    __syncwarp();
}

void GaussMinimizationNode::nonlinearCG(int32_t invocation_idx)
{
#define iter_warp_printf(...) if (iter < 4) { warp_printf(__VA_ARGS__); }
#define iter_matrix_printf(...) if (iter < 4) { printMatrix(__VA_ARGS__); }

    using namespace gpu_utils;

    constexpr float kTolerance = 1e-8f;
    constexpr float lsTolerance = 0.01f;
    constexpr float MINVAL = 1e-12f;

    uint32_t total_resident_warps = (blockDim.x * gridDim.x) / 32;

    uint32_t total_num_worlds = mwGPU::GPUImplConsts::get().numWorlds;

    // Global warp ID
    // uint32_t warp_id = (blockDim.x * blockIdx.x + threadIdx.x) / 32;
    uint32_t warp_id = threadIdx.x / 32;
    uint32_t lane_id = threadIdx.x % 32;

    assert(blockDim.x == consts::numMegakernelThreads);

    StateManager *state_mgr = getStateManager();

    uint32_t world_id = invocation_idx;

    { // Do the computation
        CVSolveData *curr_sd = &solveDatas[world_id];

        float tol_scale = 1.f / curr_sd->totalMass;

        prepareRegInfos(curr_sd);

        float *x = (float *)curr_sd->regInfos[0].ptr;
        float *m_grad = (float *)curr_sd->regInfos[1].ptr;
        float *p = (float *)curr_sd->regInfos[2].ptr;
        float *scratch1 = (float *)curr_sd->regInfos[3].ptr;
        float *scratch2 = (float *)curr_sd->regInfos[4].ptr;
        float *scratch3 = (float *)curr_sd->regInfos[5].ptr;
        float *jaccref = (float *)curr_sd->regInfos[6].ptr;
        float *mxmin = (float *)curr_sd->regInfos[7].ptr;

        float *acc_ref = [&]() -> float * {
            if (curr_sd->accRefAllocatedBytes == 0) {
                return (float *)nullptr;
            } else {
                return (float *)curr_sd->accRefMem;
            }
        } ();

        // printMatrix(acc_ref, 1, curr_sd->numRowsJc, "accref");

        // We are using freeAcc as initial guess
        warpCopy(x, curr_sd->freeAcc, sizeof(float) * curr_sd->freeAccDim);
        __syncwarp();

        // printMatrix(x, 1, curr_sd->freeAccDim, "x");

        dobjWarp<true>(m_grad, x, curr_sd, scratch1,
                       jaccref, mxmin, acc_ref, false);
        __syncwarp();

        float curr_fun = objWarp(x, curr_sd, jaccref, mxmin);
        __syncwarp();

        // warp_printf("curr_fun = %f\n", curr_fun);

        // Keep track of the norm2 of g (m_grad currently has g)
        float g_norm = sqrtf(norm2Warp(m_grad, curr_sd->freeAccDim));

        float g_dot_m_grad = sparseBlkDiagSolve<float, true>(
                m_grad, &curr_sd->massSparse, scratch1);

        // printMatrix(m_grad, 1, curr_sd->freeAccDim, "m_grad");

        // warp_printf("g_dot_m_grad = %f\n", g_dot_m_grad);

        // By now, m_grad actually has m_grad
        __syncwarp();

        warpLoop(curr_sd->freeAccDim, [&](uint32_t iter) {
            p[iter] = -m_grad[iter];
        });
        __syncwarp();

        uint32_t max_iters = 100;
        uint32_t iter = 0;

        for (; iter < max_iters; ++iter) {
            // iter_warp_printf("##################### CG iter=%d\n", iter);

            // TODO: add improvement check
            if (tol_scale * g_norm < kTolerance)
                break;

            float p_norm = sqrtf(norm2Warp(p, curr_sd->freeAccDim));

            // iter_warp_printf("p_norm = %f\n", p_norm);

            if (p_norm < MINVAL)
                break;

           // float lsTol = lsTolerance * kTolerance * p_norm / tol_scale;
           float lsTol = lsTolerance;
           float alpha = exactLineSearch(
                curr_sd, jaccref, mxmin, p, x, lsTol, scratch1, false);
            __syncwarp();

            // iter_warp_printf("alpha = %f\n", alpha);

            // warp_printf("alpha = %f\n", alpha);

            // No improvement
            if (alpha == 0.f)
                break;

            // Update x to the new value after alpha was found
            warpLoop(curr_sd->freeAccDim,
                [&](uint32_t iter) {
                    x[iter] += alpha * p[iter];
                });

            __syncwarp();

            // iter_matrix_printf(x, 1, curr_sd->freeAccDim, "x_new");

            float *g_new = m_grad;
            float new_fun = 0.f;
            { // Get the new gradient
                warpCopy(scratch2, m_grad, curr_sd->freeAccDim * sizeof(float));

                dobjWarp<true>(g_new, x, curr_sd, scratch1,
                               jaccref, mxmin, acc_ref, false);

                // iter_matrix_printf(g_new, 1, curr_sd->freeAccDim, "g_new");
                __syncwarp();

                new_fun = objWarp(x, curr_sd, jaccref, mxmin);
                __syncwarp();

                g_norm = sqrtf(norm2Warp(g_new, curr_sd->freeAccDim));
            }

            // warp_printf("curr_fun = %f\n", new_fun);

            if (tol_scale * (curr_fun - new_fun) < kTolerance) {
#if 0
                warp_printf(
                    "tol_scale (%.17f) * (curr_fun (%.17f) - new_fun (%.17f)) < kTolerance (%.17f)\n",
                    tol_scale,
                    curr_fun,
                    new_fun,
                    kTolerance);
#endif
                break;
            }

            {
                // Now we have scratch1 and scratch2 to play with
                // We need have access to these three at the same time:
                // g_new, Mgrad_new M_grad
                warpCopy(scratch3, g_new, curr_sd->freeAccDim * sizeof(float));

                float g_dot_m_grad_new = sparseBlkDiagSolve<float, true>(
                        m_grad, &curr_sd->massSparse, scratch1);

                // iter_matrix_printf(m_grad, 1, curr_sd->freeAccDim, "m_grad_new");

                // By now, m_grad actually has m_grad_new,
                //         scratch2 has m_grad
                //         scratch3 has g_new
                __syncwarp();

                // dot(g_new, (M_gradnew - m_grad))
                float g_new_dot_mgradmin =
                    dotVectorsPred<float>(
                        [&](uint32_t iter) {
                            return scratch3[iter];
                        },
                        [&](uint32_t iter) {
                            return m_grad[iter] - scratch2[iter];
                        },
                        curr_sd->freeAccDim);

                float beta = g_new_dot_mgradmin / fmax(g_dot_m_grad, MINVAL);

                g_dot_m_grad = g_dot_m_grad_new;

                beta = fmax(0.f, beta);

                // iter_warp_printf("beta = %f\n", beta);

                warpLoop(
                    curr_sd->freeAccDim,
                    [&](uint32_t iter) {
                        p[iter] = -m_grad[iter] + beta * p[iter];
                    });

                // iter_matrix_printf(p, 1, curr_sd->freeAccDim, "p_new");

#if 0
                if (iter < 4) {
                    warp_printf("beta=%f\n", beta);
                    printMatrix(p, 1, curr_sd->freeAccDim, "p");
                }
#endif
            }

            curr_fun = new_fun;
        }

#if 1
        if (lane_id == 0 && curr_sd->numRowsJc > 0) {
            if (iter > 20) {
                printf("world = %d; num CG iterations: %d; g_norm %f\n", world_id, iter, g_norm);
            }

#if 0
            if (iter == 100) {
                printf(
                    "world_id = %d; num_iters = %d; g_norm = %f; g_dot_m_grad = %f;\n",
                    world_id, iter, g_norm, g_dot_m_grad);
                printf(
                    "tol_scale (%f) * g_norm (%f) < ktolerance (%f)?\n",
                    tol_scale, g_norm, kTolerance);
            }
#endif
        }
#endif

        { // Now, we need to copy x into the right components
            auto get_bodies = [state_mgr](BodyGroupHierarchy &hier)
                -> Entity * {
                uint8_t *bytes =
                    (uint8_t *)state_mgr->memoryRangePointer<
                        MRElement128b>(hier.mrBodies);
                return (Entity *)bytes;
            };

            BodyGroupHierarchy *hiers = state_mgr->getWorldComponents<
                BodyGroup, BodyGroupHierarchy>(world_id);
            CountT num_grps = state_mgr->numRows<BodyGroup>(world_id);

            // Make sure to have a phase where we first calculate prefix sums
            // of all the DOFs of all bodies so we can get around this shit.

            if (lane_id == 0) {
                uint32_t processed_dofs = 0;
                for (CountT i = 0; i < num_grps; ++i) {
                    Entity * bodies = get_bodies(hiers[i]);
                    for (CountT j = 0; j < hiers[i].numBodies; j++) {
                        auto body = bodies[j];
                        auto numDofs = state_mgr->getUnsafe<
                                DofObjectNumDofs>(body).numDofs;
                        auto &acceleration = state_mgr->getUnsafe<
                                DofObjectAcceleration>(body);

                        for (CountT k = 0; k < numDofs; k++) {
                            acceleration.dqv[k] = x[processed_dofs];
                            processed_dofs++;
                        }
                    }
                }
            }
        }

        world_id += total_resident_warps;
    }
}

TaskGraph::NodeID GaussMinimizationNode::addToGraph(
        TaskGraph::Builder &builder,
        Span<const TaskGraph::NodeID> deps)
{
    using namespace mwGPU;

    StateManager *state_mgr = getStateManager();

    auto data_id = builder.constructNodeData<GaussMinimizationNode>(
            state_mgr);
    auto &gauss_data = builder.getDataRef(data_id);

    uint32_t num_invocations = mwGPU::GPUImplConsts::get().numWorlds;

#if 1
    TaskGraph::NodeID cur_node = builder.addNodeFn<
        &GaussMinimizationNode::allocateScratch>(data_id, {},
                Optional<TaskGraph::NodeID>::none(),
                num_invocations,
                // This is the thread block dimension
                32);

    cur_node = builder.addNodeFn<
        &GaussMinimizationNode::prepareSolver>(data_id, {cur_node},
                Optional<TaskGraph::NodeID>::none(),
                num_invocations,
                32);

    cur_node = builder.addNodeFn<
        &GaussMinimizationNode::computeAccRef>(data_id, {cur_node},
                Optional<TaskGraph::NodeID>::none(),
                num_invocations,
                32);

    cur_node = builder.addNodeFn<
        &GaussMinimizationNode::nonlinearCG>(data_id, {cur_node},
                Optional<TaskGraph::NodeID>::none(),
                num_invocations,
                32);

    return cur_node;
#endif

#if 0
    TaskGraph::NodeID a_ref_node = builder.addNodeFn<
        &GaussMinimizationNode::testNodeSparseMul>(data_id, {},
                Optional<TaskGraph::NodeID>::none(),
                1,
                // This is the thread block dimension
                32);

    return a_ref_node;
#endif

}
#else
inline void solveCPU(Context &ctx,
                     CVSolveData &cv_sing)
{
    uint32_t world_id = ctx.worldID().idx;

    StateManager *state_mgr = getStateManager(ctx);
    PhysicsSystemState &physics_state = ctx.singleton<PhysicsSystemState>();

    // Call the solver
    if (cv_sing.cvxSolve && cv_sing.cvxSolve->fn) {
        cv_sing.cvxSolve->totalNumDofs = cv_sing.totalNumDofs;
        cv_sing.cvxSolve->numContactPts = cv_sing.numContactPts;
        cv_sing.cvxSolve->h = cv_sing.h;
        cv_sing.cvxSolve->mass = cv_sing.mass;
        cv_sing.cvxSolve->free_acc = cv_sing.freeAcc;
        cv_sing.cvxSolve->vel = cv_sing.vel;
        cv_sing.cvxSolve->J_c = cv_sing.J_c;
        cv_sing.cvxSolve->J_e = cv_sing.J_e;
        cv_sing.cvxSolve->numEqualityRows = cv_sing.numRowsJe;
        cv_sing.cvxSolve->mu = cv_sing.mu;
        cv_sing.cvxSolve->penetrations = cv_sing.penetrations;
        cv_sing.cvxSolve->eqResiduals = cv_sing.eqResiduals;
        cv_sing.cvxSolve->diagApprox_c = cv_sing.diagApprox_c;
        cv_sing.cvxSolve->diagApprox_e = cv_sing.diagApprox_e;

        cv_sing.cvxSolve->callSolve.store_release(1);
        while (cv_sing.cvxSolve->callSolve.load_acquire() != 2);
        cv_sing.cvxSolve->callSolve.store_relaxed(0);

        float *res = cv_sing.cvxSolve->resPtr;

        if (res) {
            BodyGroupHierarchy *hiers = state_mgr->getWorldComponents<
                BodyGroup, BodyGroupHierarchy>(world_id);

            CountT num_grps = state_mgr->numRows<BodyGroup>(world_id);

            // Update the body accelerations
            uint32_t processed_dofs = 0;
            for (CountT i = 0; i < num_grps; ++i)
            {
                for (CountT j = 0; j < hiers[i].numBodies; j++)
                {
                    auto body = hiers[i].bodies(ctx)[j];
                    auto numDofs = ctx.get<DofObjectNumDofs>(body).numDofs;
                    auto &acceleration = ctx.get<DofObjectAcceleration>(body);
                    for (CountT k = 0; k < numDofs; k++) {
                        acceleration.dqv[k] = res[processed_dofs];
                        processed_dofs++;
                    }
                }
            }
        }
    }
}
#endif

inline void computeBodyCOM(Context &ctx,
                           DofObjectTmpState &tmp_state,
                           const DofObjectInertial inertial,
                           const DofObjectNumDofs num_dofs)
{
    if (num_dofs.type == DofType::FixedBody) {
#if 0
        tmp_state.scratch[0] =
            tmp_state.scratch[1] =
            tmp_state.scratch[2] =
            tmp_state.scratch[3] = 0.f;
#endif
        tmp_state.scratch[0] = inertial.mass * tmp_state.comPos.x;
        tmp_state.scratch[1] = inertial.mass * tmp_state.comPos.y;
        tmp_state.scratch[2] = inertial.mass * tmp_state.comPos.z;
        tmp_state.scratch[3] = inertial.mass;
    } else {
        tmp_state.scratch[0] = inertial.mass * tmp_state.comPos.x;
        tmp_state.scratch[1] = inertial.mass * tmp_state.comPos.y;
        tmp_state.scratch[2] = inertial.mass * tmp_state.comPos.z;
        tmp_state.scratch[3] = inertial.mass;
    }
}

inline void computeTotalCOM(Context &ctx,
                            BodyGroupHierarchy &body_grp)
{
    Vector3 hierarchy_com = Vector3::zero();
    float total_mass = 0.f;

    Entity *bodies = body_grp.bodies(ctx);

    for (int i = 0; i < body_grp.numBodies; ++i) {
        auto &tmp_state = ctx.get<DofObjectTmpState>(bodies[i]);

        hierarchy_com += Vector3 {
            tmp_state.scratch[0],
            tmp_state.scratch[1],
            tmp_state.scratch[2]
        };

        total_mass += tmp_state.scratch[3];
    }

    body_grp.comPos = hierarchy_com / total_mass;
}

inline void computeSpatialInertias(Context &ctx,
                                   DofObjectTmpState &tmp_state,
                                   const DofObjectHierarchyDesc hier_desc,
                                   const DofObjectInertial inertial,
                                   const DofObjectNumDofs num_dofs)
{
#if 0
    if(num_dofs.type == DofType::FixedBody) {
        tmp_state.spatialInertia.mass = 0.f;
        return;
    }
#endif

    Vector3 body_grp_com_pos = ctx.get<BodyGroupHierarchy>(
            hier_desc.bodyGroup).comPos;

    Diag3x3 inertia = inertial.inertia * hier_desc.globalScale * hier_desc.globalScale;
    float mass = inertial.mass;

    // We need to find inertia tensor in world space orientation
    Mat3x3 rot_mat = Mat3x3::fromQuat(tmp_state.composedRot);
    // I_world = R * I * R^T (since R^T transforms from world to local)
    Mat3x3 i_world_frame = rot_mat * inertia * rot_mat.transpose();

    // Compute the 3x3 skew-symmetric matrix (r^x)
    // (where r is from Plücker origin to COM)
    Vector3 adjustedCom = tmp_state.comPos - body_grp_com_pos;
    Mat3x3 sym_mat = skewSymmetricMatrix(adjustedCom);
    // (I_world - m r^x r^x)
    Mat3x3 inertia_mat = i_world_frame - (mass * sym_mat * sym_mat);

    // Take only the upper triangular part (since it's symmetric)
    tmp_state.spatialInertia.spatial_inertia[0] = inertia_mat[0][0];
    tmp_state.spatialInertia.spatial_inertia[1] = inertia_mat[1][1];
    tmp_state.spatialInertia.spatial_inertia[2] = inertia_mat[2][2];
    tmp_state.spatialInertia.spatial_inertia[3] = inertia_mat[1][0];
    tmp_state.spatialInertia.spatial_inertia[4] = inertia_mat[2][0];
    tmp_state.spatialInertia.spatial_inertia[5] = inertia_mat[2][1];

    // Rest of parameters
    tmp_state.spatialInertia.mass = mass;
    tmp_state.spatialInertia.mCom = mass * adjustedCom;
}

// Compute spatial inertia of subtree rooted at the body.
inline void combineSpatialInertias(Context &ctx,
                                   BodyGroupHierarchy &body_grp)
{
    // Backward pass from children to parent
    for (CountT i = body_grp.numBodies-1; i > 0; --i) {
        Entity body_i = body_grp.bodies(ctx)[i];

        auto &current_hier_desc = ctx.get<DofObjectHierarchyDesc>(body_i);
        auto &current_tmp_state = ctx.get<DofObjectTmpState>(body_i);

        auto &parent_tmp_state = ctx.get<DofObjectTmpState>(
                body_grp.bodies(ctx)[current_hier_desc.parentIndex]);

        parent_tmp_state.spatialInertia += current_tmp_state.spatialInertia;
    }
}

// Computes the Phi matrix from generalized velocities to Plücker coordinates
inline float* computePhi(Context &ctx,
                         const DofObjectNumDofs num_dofs,
                         DofObjectTmpState &tmp_state,
                         Vector3 origin)
{
    Phi phi = tmp_state.phi;

    float *S = tmp_state.getPhiFull(ctx);

    if (num_dofs.type == DofType::FreeBody) {
        // S = [1_3x3 r^x; 0 1_3x3], column-major
        memset(S, 0.f, 6 * 6 * sizeof(float));
        // Diagonal identity
        for(CountT i = 0; i < 6; ++i) {
            S[i * 6 + i] = 1.f;
        }
        // r^x Skew symmetric matrix
        Vector3 comPos = {phi.v[0], phi.v[1], phi.v[2]};
        comPos -= origin;
        S[0 + 6 * 4] = -comPos.z;
        S[0 + 6 * 5] = comPos.y;
        S[1 + 6 * 3] = comPos.z;
        S[1 + 6 * 5] = -comPos.x;
        S[2 + 6 * 3] = -comPos.y;
        S[2 + 6 * 4] = comPos.x;
    } else if (num_dofs.type == DofType::Slider) {
        // This is just the axis of the slider.
        S[0] = phi.v[0];
        S[1] = phi.v[1];
        S[2] = phi.v[2];
        S[3] = 0.f;
        S[4] = 0.f;
        S[5] = 0.f;
    } else if (num_dofs.type == DofType::Hinge) {
        // S = [r \times hinge; hinge]
        Vector3 hinge = {phi.v[0], phi.v[1], phi.v[2]};
        Vector3 anchorPos = {phi.v[3], phi.v[4], phi.v[5]};
        anchorPos -= origin;
        Vector3 r_cross_hinge = anchorPos.cross(hinge);
        S[0] = r_cross_hinge.x;
        S[1] = r_cross_hinge.y;
        S[2] = r_cross_hinge.z;
        S[3] = hinge.x;
        S[4] = hinge.y;
        S[5] = hinge.z;
    } else if (num_dofs.type == DofType::Ball) {
        // This will just get right-multiplied by the angular velocity
        Vector3 anchor_pos = {phi.v[0], phi.v[1], phi.v[2]};
        anchor_pos -= origin;

        // We need to right multiply these by the parent composed rotation
        // matrix.
        Mat3x3 rx = skewSymmetricMatrix(anchor_pos);
        Quat parent_composed_rot = Quat{
            phi.v[3], phi.v[4], phi.v[5], phi.v[6]
        };
        Mat3x3 parent_rot = Mat3x3::fromQuat(parent_composed_rot);

        rx *= parent_rot;

        for (int col = 0; col < 3; ++col) {
            S[col * 6 + 0] = rx[col][0];
            S[col * 6 + 1] = rx[col][1];
            S[col * 6 + 2] = rx[col][2];

            S[col * 6 + 3] = parent_rot[col][0];
            S[col * 6 + 4] = parent_rot[col][1];
            S[col * 6 + 5] = parent_rot[col][2];
        }
    } else {
        // MADRONA_UNREACHABLE();
    }

    return S;
}

inline float* computePhiDot(Context &ctx,
                            DofObjectNumDofs &num_dofs,
                            DofObjectTmpState &tmp_state,
                            SpatialVector &v_hat)
{
    float *S = tmp_state.getPhiFull(ctx);

    // Same storage location, but will need to be incremented based on size of S
    float *S_dot = S + 6 * num_dofs.numDofs;

    if (num_dofs.type == DofType::FreeBody) {
        // S_dot = [0_3x3 v^x; 0_3x3 0_3x3], column-major
        memset(S_dot, 0.f, 6 * 6 * sizeof(float));
        // v^x Skew symmetric matrix
        Vector3 v_trans = v_hat.linear;
        S_dot[0 + 6 * 4] = -v_trans.z;
        S_dot[0 + 6 * 5] = v_trans.y;
        S_dot[1 + 6 * 3] = v_trans.z;
        S_dot[1 + 6 * 5] = -v_trans.x;
        S_dot[2 + 6 * 3] = -v_trans.y;
        S_dot[2 + 6 * 4] = v_trans.x;
    }
    else if (num_dofs.type == DofType::Slider) {
        // S_dot = v [spatial cross] S
        SpatialVector S_sv = SpatialVector::fromVec(S);
        SpatialVector S_dot_sv = v_hat.cross(S_sv);
        S_dot[0] = S_dot_sv.linear.x;
        S_dot[1] = S_dot_sv.linear.y;
        S_dot[2] = S_dot_sv.linear.z;
        S_dot[3] = S_dot_sv.angular.x;
        S_dot[4] = S_dot_sv.angular.y;
        S_dot[5] = S_dot_sv.angular.z;
    }
    else if (num_dofs.type == DofType::Hinge) {
        // S_dot = v [spatial cross] S
        SpatialVector S_sv = SpatialVector::fromVec(S);
        SpatialVector S_dot_sv = v_hat.cross(S_sv);
        S_dot[0] = S_dot_sv.linear.x;
        S_dot[1] = S_dot_sv.linear.y;
        S_dot[2] = S_dot_sv.linear.z;
        S_dot[3] = S_dot_sv.angular.x;
        S_dot[4] = S_dot_sv.angular.y;
        S_dot[5] = S_dot_sv.angular.z;
    }
    else if (num_dofs.type == DofType::Ball) {
        // S_dot = v [spatial cross] S
        for (int i = 0; i < 3; ++i) {
            SpatialVector S_sv = SpatialVector::fromVec(S + (i * 6));
            SpatialVector S_dot_sv = v_hat.cross(S_sv);

            S_dot[i * 6 + 0] = S_dot_sv.linear.x;
            S_dot[i * 6 + 1] = S_dot_sv.linear.y;
            S_dot[i * 6 + 2] = S_dot_sv.linear.z;
            S_dot[i * 6 + 3] = S_dot_sv.angular.x;
            S_dot[i * 6 + 4] = S_dot_sv.angular.y;
            S_dot[i * 6 + 5] = S_dot_sv.angular.z;
        }
    }
    return S_dot;
}

// First pass to compute Phi with body COM as origin
inline void computePhiHierarchy(Context &ctx,
                                DofObjectTmpState &tmp_state,
                                const DofObjectNumDofs num_dofs,
                                const DofObjectHierarchyDesc &desc)
{
    if (num_dofs.numDofs > 0) {
        Vector3 com_pos = ctx.get<BodyGroupHierarchy>(desc.bodyGroup).comPos;
        computePhi(ctx, num_dofs, tmp_state, com_pos);
    }
}

// J_C = C^T[e_{b1} S_1, e_{b2} S_2, ...], col-major
//  where e_{bi} = 1 if body i is an ancestor of b
//  C^T projects into the contact space
inline float* computeContactJacobian(Context &ctx,
                                     BodyGroupHierarchy &body_grp,
                                     DofObjectHierarchyDesc &hier_desc,
                                     Mat3x3 &C,
                                     Vector3 &origin,
                                     float *J,
                                     uint32_t body_dof_offset,
                                     uint32_t jac_row,
                                     uint32_t j_num_rows,
                                     float coeff,
                                     bool dbg)
{
    // Compute prefix sum to determine the start of the block for each body
    uint32_t *block_start = body_grp.getDofPrefixSum(ctx);

    // Populate J_C by traversing up the hierarchy
    int32_t curr_idx = hier_desc.index;
    while(curr_idx != -1) {
        Entity body = body_grp.bodies(ctx)[curr_idx];
        auto &curr_tmp_state = ctx.get<DofObjectTmpState>(body);
        auto &curr_num_dofs = ctx.get<DofObjectNumDofs>(body);
        auto &curr_hier_desc = ctx.get<DofObjectHierarchyDesc>(body);

        // Populate columns of J_C
        float S[18] = {};
        computePhiTrans(curr_num_dofs, curr_tmp_state, origin, S);
        // Only use translational part of S
        for(CountT i = 0; i < curr_num_dofs.numDofs; ++i) {
            float *J_col = J +
                    j_num_rows * (body_dof_offset + block_start[curr_idx] + i) +
                    jac_row;

            float *S_col = S + 3 * i;
            for(CountT j = 0; j < 3; ++j) {
                J_col[j] = S_col[j];
            }
        }
        curr_idx = curr_hier_desc.parentIndex;
    }

    // Multiply by C^T to project into contact space
    for(CountT i = 0; i < body_grp.numDofs; ++i) {
        float *J_col = J +
                j_num_rows * (body_dof_offset + i) +
                jac_row;

        Vector3 J_col_vec = { J_col[0], J_col[1], J_col[2] };
        J_col_vec = C.transpose() * J_col_vec;
        J_col[0] = coeff * J_col_vec.x;
        J_col[1] = coeff * J_col_vec.y;
        J_col[2] = coeff * J_col_vec.z;
    }
    return J;
}


// J = [e_{b1} S_1, e_{b2} S_2, ...], col-major
//  where e_{bi} = 1 if body i is an ancestor of b
inline float* computeBodyJacobian(Context &ctx,
                                  BodyGroupHierarchy &body_grp,
                                  DofObjectHierarchyDesc &hier_desc,
                                  Vector3 &origin,
                                  float *J,
                                  uint32_t body_dof_offset,
                                  uint32_t jac_row,
                                  uint32_t j_num_rows) {

    // Compute prefix sum to determine the start of the block for each body
    uint32_t *block_start = body_grp.getDofPrefixSum(ctx);

    // Populate J_C by traversing up the hierarchy
    int32_t curr_idx = hier_desc.index;
    while(curr_idx != -1) {
        Entity body = body_grp.bodies(ctx)[curr_idx];
        auto &curr_tmp_state = ctx.get<DofObjectTmpState>(body);
        auto &curr_num_dofs = ctx.get<DofObjectNumDofs>(body);
        auto &curr_hier_desc = ctx.get<DofObjectHierarchyDesc>(body);

        // Populate columns of J_C
        float *S = computePhi(ctx, curr_num_dofs, curr_tmp_state, origin);
        for(CountT i = 0; i < curr_num_dofs.numDofs; ++i) {
            float *J_col = J +
                    j_num_rows * (body_dof_offset + block_start[curr_idx] + i) +
                    jac_row;

            float *S_col = S + 6 * i;
            for(CountT j = 0; j < 6; ++j) {
                J_col[j] = S_col[j];
            }
        }
        curr_idx = curr_hier_desc.parentIndex;
    }
    return J;
}

// y = Mx. Based on Table 6.5 in Featherstone
inline void mulM(Context &ctx, BodyGroupHierarchy &body_grp,
    float *x, float *y) {
    CountT total_dofs = body_grp.numDofs;

    int32_t *expandedParent = body_grp.getExpandedParent(ctx);

    float *massMatrix = body_grp.getMassMatrix(ctx);

    auto M = [&](int32_t row, int32_t col) -> float& {
        return massMatrix[row + total_dofs * col];
    };

    for(int32_t i = 0; i < total_dofs; ++i) {
        y[i] = M(i, i) * x[i];
    }

    for(int32_t i = (int32_t) total_dofs - 1; i >= 0; --i) {
        int32_t j = expandedParent[i];
        while(j != -1) {
            y[i] += M(i, j) * x[j];
            y[j] += M(i, j) * x[i];
            j = expandedParent[j];
        }
    }
}

// Solves M^{-1}x, overwriting x. Based on Table 6.5 in Featherstone
inline void solveM(Context &ctx, BodyGroupHierarchy &body_grp, float *x) {
    CountT total_dofs = body_grp.numDofs;

    int32_t *expandedParent = body_grp.getExpandedParent(ctx);
    float *massMatrixLTDL = body_grp.getMassMatrixLTDL(ctx);

    auto ltdl = [&](int32_t row, int32_t col) -> float& {
        return massMatrixLTDL[row + total_dofs * col];
    };
    // M=L^TDL, so first solve L^{-T} x
    for (int32_t i = (int32_t) total_dofs - 1; i >= 0; --i) {
        int32_t j = expandedParent[i];
        while (j != -1) {
            x[j] -= ltdl(i, j) * x[i];
            j = expandedParent[j];
        }
    }
    // D^{-1} x
    for (int32_t i = 0; i < total_dofs; ++i) {
        x[i] /= ltdl(i, i);
    }
    // L^{-1} x
    for (int32_t i = 0; i < total_dofs; ++i) {
        int32_t j = expandedParent[i];
        while (j != -1) {
            x[i] -= ltdl(i, j) * x[j];
            j = expandedParent[j];
        }
    }
}


inline void computeInvMass(Context &ctx,
                           BodyGroupHierarchy &grp) {
    // For each body, find translational and rotational inverse weight
    //  by computing A = J M^{-1} J^T
    float A[36] = {}; // 6x6
    float J[6 * grp.numDofs]; // col-major (shape 6 x numDofs)
    float MinvJT[grp.numDofs * 6]; // col-major (shape numDofs x 6)

    // Compute the inverse weight for each body
    for (CountT i_body = 0; i_body < grp.numBodies; ++i_body) {
        Entity body = grp.bodies(ctx)[i_body];
        auto body_dofs = ctx.get<DofObjectNumDofs>(body);
        auto tmp_state = ctx.get<DofObjectTmpState>(body);
        auto hier_desc = ctx.get<DofObjectHierarchyDesc>(body);
        auto &dof_inertial = ctx.get<DofObjectInertial>(body);

#if 0
        if (body_dofs.type == DofType::FixedBody) {
            dof_inertial.approxInvMassRot = 0.f;
            dof_inertial.approxInvMassTrans = 0.f;
            continue;
        }
#endif

        // Compute J
        memset(J, 0.f, 6 * grp.numDofs * sizeof(float));
        computeBodyJacobian(ctx, grp, hier_desc, tmp_state.comPos, J, 0, 0, 6);
        // Helper
        auto Jb = [&](int32_t row, int32_t col) -> float& {
            return J[row + 6 * col];
        };
        auto MinvJTb = [&](int32_t row, int32_t col) -> float& {
            return MinvJT[row + grp.numDofs * col];
        };
        auto Ab = [&](int32_t row, int32_t col) -> float& {
            return A[row + 6 * col];
        };

        // Copy into J^T
        for (CountT i = 0; i < 6; ++i) {
            for (CountT j = 0; j < grp.numDofs; ++j) {
                MinvJTb(j, i) = Jb(i, j);
            }
        }
        // M^{-1} J^T
        for (CountT i = 0; i < 6; ++i) {
            float *col = MinvJT + i * 6;
            solveM(ctx, grp, col);
        }
        // A = J M^{-1} J^T
        memset(A, 0.f, 36 * sizeof(float));
        for (CountT i = 0; i < 6; ++i) {
            for (CountT j = 0; j < 6; ++j) {
                for (CountT k = 0; k < grp.numDofs; ++k) {
                    Ab(i, j) += Jb(i, k) * MinvJTb(k, j);
                }
            }
        }
        // Compute the inverse weight
        dof_inertial.approxInvMassTrans = (Ab(0, 0) + Ab(1, 1) + Ab(2, 2)) / 3.f;
        dof_inertial.approxInvMassRot = (Ab(3, 3) + Ab(4, 4) + Ab(5, 5)) / 3.f;
    }

    // For each DOF, find the inverse weight
    uint32_t dof_offset = 0;
    for (CountT i_body = 0; i_body < grp.numBodies; ++i_body) {
        Entity body = grp.bodies(ctx)[i_body];
        auto body_dofs = ctx.get<DofObjectNumDofs>(body);
        auto &dof_inertial = ctx.get<DofObjectInertial>(body);
#if 0
        if (body_dofs.type == DofType::FixedBody) {
            continue;
        }
#endif

        // Jacobian size (body dofs x total dofs)
        memset(J, 0.f, body_dofs.numDofs * grp.numDofs * sizeof(float));
        auto Jd = [&](int32_t row, int32_t col) -> float& {
            return J[row + body_dofs.numDofs * col];
        };
        // J^T and M^{-1}J^T (total dofs x body dofs)
        auto MinvJTd = [&](int32_t row, int32_t col) -> float& {
            return MinvJT[row + grp.numDofs * col];
        };
        // A = JM^{-1}J^T. (body dofs x body dofs)
        auto Ad = [&](int32_t row, int32_t col) -> float& {
            return A[row + body_dofs.numDofs * col];
        };

        // Fill in 1 for the corresponding body dofs
        for (CountT i = 0; i < body_dofs.numDofs; ++i) {
            Jd(i, i + dof_offset) = 1.f;
        }

        // Copy into J^T
        for (CountT i = 0; i < body_dofs.numDofs; ++i) {
            for (CountT j = 0; j < grp.numDofs; ++j) {
                MinvJTd(j, i) = Jd(i, j);
            }
        }

        // M^{-1} J^T. (J^T is total dofs x body dofs)
        for (CountT i = 0; i < body_dofs.numDofs; ++i) {
            float *col = MinvJT + i * grp.numDofs;
            solveM(ctx, grp, col);
        }

        // A = J M^{-1} J^T
        memset(A, 0.f, body_dofs.numDofs * body_dofs.numDofs * sizeof(float));
        for (CountT i = 0; i < body_dofs.numDofs; ++i) {
            for (CountT j = 0; j < body_dofs.numDofs; ++j) {
                for (CountT k = 0; k < grp.numDofs; ++k) {
                    Ad(i, j) += Jd(i, k) * MinvJTd(k, j);
                }
            }
        }

        // Update the inverse mass of each DOF
        if (body_dofs.numDofs == 6) {
           dof_inertial.approxInvMassDof[0] = dof_inertial.approxInvMassDof[1] = dof_inertial.approxInvMassDof[2] =
               (Ad(0, 0) + Ad(1, 1) + Ad(2, 2)) / 3.f;
           dof_inertial.approxInvMassDof[3] = dof_inertial.approxInvMassDof[4] = dof_inertial.approxInvMassDof[5] =
                (Ad(3, 3) + Ad(4, 4) + Ad(5, 5)) / 3.f;
        } else if (body_dofs.numDofs == 3) {
            dof_inertial.approxInvMassDof[0] = dof_inertial.approxInvMassDof[1] = dof_inertial.approxInvMassDof[2] =
                (Ad(0, 0) + Ad(1, 1) + Ad(2, 2)) / 3.f;
        } else {
            dof_inertial.approxInvMassDof[0] = Ad(0, 0);
        }

        dof_offset += body_dofs.numDofs;
    }
}



// CRB: Compute the Mass Matrix (n_dofs x n_dofs)
#ifdef MADRONA_GPU_MODE
inline void compositeRigidBody(Context &ctx,
                               BodyGroupHierarchy &body_grp)
{
    uint32_t world_id = ctx.worldID().idx;
    StateManager *state_mgr = getStateManager(ctx);

    // Mass Matrix of this entire body group, column-major
    uint32_t total_dofs = body_grp.numDofs;

    float *M = body_grp.getMassMatrix(ctx);

    memset(M, 0.f, total_dofs * total_dofs * sizeof(float));

    // Compute prefix sum to determine the start of the block for each body
    uint32_t *block_start = body_grp.getDofPrefixSum(ctx);

    // Backward pass
    for (CountT i = body_grp.numBodies-1; i >= 0; --i) {
        Entity body = body_grp.bodies(ctx)[i];

        auto &i_hier_desc = ctx.get<DofObjectHierarchyDesc>(body);
        auto &i_tmp_state = ctx.get<DofObjectTmpState>(body);
        auto &i_num_dofs = ctx.get<DofObjectNumDofs>(body);

        float *S_i = i_tmp_state.getPhiFull(ctx);

        float *F = body_grp.scratch;

        for(CountT col = 0; col < i_num_dofs.numDofs; ++col) {
            float *S_col = S_i + 6 * col;
            float *F_col = F + 6 * col;
            i_tmp_state.spatialInertia.multiply(S_col, F_col);
        }

        // M_{ii} = S_i^T I_i^C S_i = F^T S_i
        float *M_ii = M + block_start[i] * total_dofs + block_start[i];
        for(CountT row = 0; row < i_num_dofs.numDofs; ++row) {
            float *F_col = F + 6 * row; // take col for transpose
            for(CountT col = 0; col < i_num_dofs.numDofs; ++col) {
                float *S_col = S_i + 6 * col;
                for(CountT k = 0; k < 6; ++k) {
                    M_ii[row + total_dofs * col] += F_col[k] * S_col[k];
                }
            }
        }

        // Traverse up hierarchy
        uint32_t j = i;
        auto parent_j = i_hier_desc.parent;
        while(parent_j != Entity::none()) {
            j = ctx.get<DofObjectHierarchyDesc>(
                body_grp.bodies(ctx)[j]).parentIndex;
            Entity body = body_grp.bodies(ctx)[j];
            auto &j_tmp_state = ctx.get<DofObjectTmpState>(body);
            auto &j_num_dofs = ctx.get<DofObjectNumDofs>(body);

            float *S_j = j_tmp_state.getPhiFull(ctx);

            // M_{ij} = M{ji} = F^T S_j
            float *M_ij = M + block_start[i] + total_dofs * block_start[j]; // row i, col j
            float *M_ji = M + block_start[j] + total_dofs * block_start[i]; // row j, col i
            for(CountT row = 0; row < i_num_dofs.numDofs; ++row) {
                float *F_col = F + 6 * row; // take col for transpose
                for(CountT col = 0; col < j_num_dofs.numDofs; ++col) {
                    float *S_col = S_j + 6 * col;
                    for(CountT k = 0; k < 6; ++k) {
                        M_ij[row + total_dofs * col] += F_col[k] * S_col[k];
                        M_ji[col + total_dofs * row] += F_col[k] * S_col[k];
                    }
                }
            }
            parent_j = ctx.get<DofObjectHierarchyDesc>(
                body_grp.bodies(ctx)[j]).parent;
        }
    }
}
#else
inline void compositeRigidBody(Context &ctx,
                               BodyGroupHierarchy &body_grp)
{
    uint32_t world_id = ctx.worldID().idx;
    StateManager *state_mgr = getStateManager(ctx);

    // Mass Matrix of this entire body group, column-major
    uint32_t total_dofs = body_grp.numDofs;

    float *M = body_grp.getMassMatrix(ctx);

    memset(M, 0.f, total_dofs * total_dofs * sizeof(float));

    // Compute prefix sum to determine the start of the block for each body
    uint32_t *block_start = body_grp.getDofPrefixSum(ctx);

    // Backward pass
    for (CountT i = body_grp.numBodies-1; i >= 0; --i) {
        Entity body = body_grp.bodies(ctx)[i];

        auto &i_hier_desc = ctx.get<DofObjectHierarchyDesc>(body);
        auto &i_tmp_state = ctx.get<DofObjectTmpState>(body);
        auto &i_num_dofs = ctx.get<DofObjectNumDofs>(body);

        float *S_i = i_tmp_state.getPhiFull(ctx);

        float *F = body_grp.scratch;

        for(CountT col = 0; col < i_num_dofs.numDofs; ++col) {
            float *S_col = S_i + 6 * col;
            float *F_col = F + 6 * col;
            i_tmp_state.spatialInertia.multiply(S_col, F_col);
        }

        // M_{ii} = S_i^T I_i^C S_i = F^T S_i
        float *M_ii = M + block_start[i] * total_dofs + block_start[i];
        for(CountT row = 0; row < i_num_dofs.numDofs; ++row) {
            float *F_col = F + 6 * row; // take col for transpose
            for(CountT col = 0; col < i_num_dofs.numDofs; ++col) {
                float *S_col = S_i + 6 * col;
                for(CountT k = 0; k < 6; ++k) {
                    M_ii[row + total_dofs * col] += F_col[k] * S_col[k];
                }
            }
        }

        // Traverse up hierarchy
        uint32_t j = i;
        auto parent_j = i_hier_desc.parent;
        while(parent_j != Entity::none()) {
            j = ctx.get<DofObjectHierarchyDesc>(
                body_grp.bodies(ctx)[j]).parentIndex;
            Entity body = body_grp.bodies(ctx)[j];
            auto &j_tmp_state = ctx.get<DofObjectTmpState>(body);
            auto &j_num_dofs = ctx.get<DofObjectNumDofs>(body);

            float *S_j = j_tmp_state.getPhiFull(ctx);

            // M_{ij} = M{ji} = F^T S_j
            float *M_ij = M + block_start[i] + total_dofs * block_start[j]; // row i, col j
            float *M_ji = M + block_start[j] + total_dofs * block_start[i]; // row j, col i
            for(CountT row = 0; row < i_num_dofs.numDofs; ++row) {
                float *F_col = F + 6 * row; // take col for transpose
                for(CountT col = 0; col < j_num_dofs.numDofs; ++col) {
                    float *S_col = S_j + 6 * col;
                    for(CountT k = 0; k < 6; ++k) {
                        M_ij[row + total_dofs * col] += F_col[k] * S_col[k];
                        M_ji[col + total_dofs * row] += F_col[k] * S_col[k];
                    }
                }
            }
            parent_j = ctx.get<DofObjectHierarchyDesc>(
                body_grp.bodies(ctx)[j]).parent;
        }
    }
}
#endif

// Computes the LTDL factorization of the mass matrix
inline void factorizeMassMatrix(Context &ctx,
                                BodyGroupHierarchy &body_grp) {
    uint32_t world_id = ctx.worldID().idx;
    StateManager *state_mgr = getStateManager(ctx);

    // First copy in the mass matrix
    uint32_t total_dofs = body_grp.numDofs;

    float *LTDL = body_grp.getMassMatrixLTDL(ctx);
    float *massMatrix = body_grp.getMassMatrix(ctx);

    memcpy(LTDL, massMatrix, total_dofs * total_dofs * sizeof(float));

    // Helper
    auto ltdl = [&](int32_t row, int32_t col) -> float& {
        return LTDL[row + total_dofs * col];
    };

    int32_t *expandedParent = body_grp.getExpandedParent(ctx);

    // Backward pass through DOFs
    for (int32_t k = (int32_t) total_dofs - 1; k >= 0; --k) {
        int32_t i = expandedParent[k];
        while (i != -1) {
            // Temporary storage
            float a = ltdl(k, i) / ltdl(k, k);
            int32_t j = i;
            while (j != -1) {
                ltdl(i, j) = ltdl(i, j) - a * ltdl(k, j);
                j = expandedParent[j];
            }
            ltdl(k, i) = a;
            i = expandedParent[i];
        }
    }
}

// Computes the unconstrained acceleration of each body
inline void computeFreeAcceleration(Context &ctx,
                                    BodyGroupHierarchy &body_grp) {
    // Negate the bias forces, solve
    CountT num_dofs = body_grp.numDofs;

    float *bias = body_grp.getBias(ctx);

    for (CountT i = 0; i < num_dofs; ++i) {
        bias[i] = -bias[i];
    }
    // This overwrites bias with the acceleration
    solveM(ctx, body_grp, bias);
}

#if defined(MADRONA_GPU_MODE) && 1
inline void recursiveNewtonEuler(Context &ctx,
                                BodyGroupHierarchy &body_grp)
{
    using namespace gpu_utils;

    uint32_t lane_id = threadIdx.x % 32;

    PhysicsSystemState &physics_state = ctx.singleton<PhysicsSystemState>();
    uint32_t total_dofs = body_grp.numDofs;

    if (lane_id == 0) {
        // Forward pass. Find in Plücker coordinates:
        //  1. velocities. v_i = v_{parent} + S * \dot{q_i}
        //  2. accelerations. a_i = a_{parent} + \dot{S} * \dot{q_i} + S * \ddot{q_i}
        //  3. forces. f_i = I_i a_i + v_i [spatial star cross] I_i v_i
        for (int i = 0; i < body_grp.numBodies; ++i) {
            Entity body = body_grp.bodies(ctx)[i];

            auto num_dofs = ctx.get<DofObjectNumDofs>(body);
            auto velocity = ctx.get<DofObjectVelocity>(body);

            auto &tmp_state = ctx.get<DofObjectTmpState>(body);
            auto &hier_desc = ctx.get<DofObjectHierarchyDesc>(body);

            float *S = tmp_state.getPhiFull(ctx);

            if (num_dofs.type == DofType::FreeBody) {
                // Free bodies must be root of their hierarchy
                SpatialVector v_body = {
                    {velocity.qv[0], velocity.qv[1], velocity.qv[2]},
                    Vector3::zero()
                };
                float *S_dot = computePhiDot(ctx, num_dofs, tmp_state, v_body);

                // v_0 = 0, a_0 = -g (fictitious upward acceleration)
                tmp_state.sAcc = {-physics_state.g, Vector3::zero()};
                tmp_state.sVel = {Vector3::zero(), Vector3::zero()};

                // S\dot{q_i} and \dot{S}\dot{q_i}
                for (int j = 0; j < 6; ++j) {
                    for (int k = 0; k < num_dofs.numDofs; ++k) {
                        tmp_state.sVel[j] += S[j + 6 * k] * velocity.qv[k];
                        tmp_state.sAcc[j] += S_dot[j + 6 * k] * velocity.qv[k];
                    }
                }
            }
            else if (num_dofs.type == DofType::FixedBody) {
                // Fixeds bodies must also be root of their hierarchy
                tmp_state.sVel = {Vector3::zero(), Vector3::zero()};
                tmp_state.sAcc = {-physics_state.g, Vector3::zero()};
            }
            else if (num_dofs.type == DofType::Hinge) {
                assert(hier_desc.parent != Entity::none());
                auto parent_tmp_state = ctx.get<DofObjectTmpState>(hier_desc.parent);
                tmp_state.sVel = parent_tmp_state.sVel;
                tmp_state.sAcc = parent_tmp_state.sAcc;

                // v_i = v_{parent} + S * \dot{q_i}, compute S * \dot{q_i}
                // a_i = a_{parent} + \dot{S} * \dot{q_i} [+ S * \ddot{q_i}, which is 0]
                // Note: we are using the parent velocity here (for hinge itself)
                float *S_dot = computePhiDot(ctx, num_dofs, tmp_state,
                    parent_tmp_state.sVel);

                float q_dot = velocity.qv[0];
                for (int j = 0; j < 6; ++j) {
                    tmp_state.sVel[j] += S[j] * q_dot;
                    tmp_state.sAcc[j] += S_dot[j] * q_dot;
                }
            }
            else if (num_dofs.type == DofType::Ball) {
                assert(hier_desc.parent != Entity::none());
                auto parent_tmp_state = ctx.get<DofObjectTmpState>(hier_desc.parent);
                tmp_state.sVel = parent_tmp_state.sVel;
                tmp_state.sAcc = parent_tmp_state.sAcc;

                // v_i = v_{parent} + S * \dot{q_i}, compute S * \dot{q_i}
                // a_i = a_{parent} + \dot{S} * \dot{q_i} [+ S * \ddot{q_i}, which is 0]
                // Note: we are using the parent velocity here (for hinge itself)
                float *S_dot = computePhiDot(ctx, num_dofs, tmp_state,
                    parent_tmp_state.sVel);

                float *q_dot = velocity.qv;
                for (int j = 0; j < 6; ++j) {
                    // TODO: Probably should switch to row major - this isn't
                    // particularly cache-friendly.
                    tmp_state.sVel[j] += S[j + 6 * 0] * q_dot[0] +
                                         S[j + 6 * 1] * q_dot[1] +
                                         S[j + 6 * 2] * q_dot[2];

                    tmp_state.sAcc[j] += S_dot[j + 6 * 0] * q_dot[0] +
                                         S_dot[j + 6 * 1] * q_dot[1] +
                                         S_dot[j + 6 * 2] * q_dot[2];
                }
            } else {
                MADRONA_UNREACHABLE();
            }

            // f_i = I_i a_i + v_i [spatial star cross] I_i v_i
            tmp_state.sForce = tmp_state.spatialInertia.multiply(tmp_state.sAcc);
            tmp_state.sForce += tmp_state.sVel.crossStar(
                tmp_state.spatialInertia.multiply(tmp_state.sVel));
        }
    }

    // Backward pass to find bias forces
    float *tau = body_grp.getBias(ctx);

    warpSetZero(tau, total_dofs * sizeof(float));
    __syncwarp();

    uint32_t warp_id = threadIdx.x / 32;
    const int32_t num_smem_bytes_per_warp =
        mwGPU::SharedMemStorage::numBytesPerWarp();
    float *smem_buf = (float *)((uint8_t *)mwGPU::SharedMemStorage::buffer +
                    num_smem_bytes_per_warp * warp_id);

    if (lane_id == 0) {
        for (CountT body = body_grp.numBodies-1; body >= 0; --body) {
            auto &hier_desc = ctx.get<DofObjectHierarchyDesc>(body_grp.bodies(ctx)[body]);
            uint32_t num_dofs = ctx.get<DofObjectNumDofs>(body_grp.bodies(ctx)[body]).numDofs;
            auto &tmp_state = ctx.get<DofObjectTmpState>(body_grp.bodies(ctx)[body]);

            uint32_t dof_offset = tmp_state.dofOffset;
            const float *S = tmp_state.getPhiFull(ctx);

            float sForce[6] = {
                tmp_state.sForce[0],
                tmp_state.sForce[1],
                tmp_state.sForce[2],
                tmp_state.sForce[3],
                tmp_state.sForce[4],
                tmp_state.sForce[5]
            };

            for (uint32_t row = 0; row < num_dofs; ++row) {
                const float *S_col = S + 6 * row;

                #pragma unroll
                for (uint32_t k = 0; k < 6; ++k) {
                    tau[dof_offset + row] += S_col[k] * sForce[k];
                }
            }

            if (hier_desc.parent != Entity::none()) {
                auto &parent_tmp_state = ctx.get<DofObjectTmpState>(hier_desc.parent);
                parent_tmp_state.sForce += tmp_state.sForce;
            }
        }
    }
}
#else
// RNE: Compute bias forces and gravity
// May want to do a GPU specific version of this to extract some
// parallelism out of this
inline void recursiveNewtonEuler(Context &ctx,
                                BodyGroupHierarchy &body_grp) {
    PhysicsSystemState &physics_state = ctx.singleton<PhysicsSystemState>();
    uint32_t total_dofs = body_grp.numDofs;

    // Forward pass. Find in Plücker coordinates:
    //  1. velocities. v_i = v_{parent} + S * \dot{q_i}
    //  2. accelerations. a_i = a_{parent} + \dot{S} * \dot{q_i} + S * \ddot{q_i}
    //  3. forces. f_i = I_i a_i + v_i [spatial star cross] I_i v_i
    for (int i = 0; i < body_grp.numBodies; ++i) {
        Entity body = body_grp.bodies(ctx)[i];

        auto num_dofs = ctx.get<DofObjectNumDofs>(body);
        auto velocity = ctx.get<DofObjectVelocity>(body);

        auto &tmp_state = ctx.get<DofObjectTmpState>(body);
        auto &hier_desc = ctx.get<DofObjectHierarchyDesc>(body);

        float *S = tmp_state.getPhiFull(ctx);

        if (num_dofs.type == DofType::FreeBody) {
            // Free bodies must be root of their hierarchy
            SpatialVector v_body = {
                {velocity.qv[0], velocity.qv[1], velocity.qv[2]},
                Vector3::zero()
            };
            float *S_dot = computePhiDot(ctx, num_dofs, tmp_state, v_body);

            // v_0 = 0, a_0 = -g (fictitious upward acceleration)
            tmp_state.sAcc = {-physics_state.g, Vector3::zero()};
            tmp_state.sVel = {Vector3::zero(), Vector3::zero()};

            // S\dot{q_i} and \dot{S}\dot{q_i}
            for (int j = 0; j < 6; ++j) {
                for (int k = 0; k < num_dofs.numDofs; ++k) {
                    tmp_state.sVel[j] += S[j + 6 * k] * velocity.qv[k];
                    tmp_state.sAcc[j] += S_dot[j + 6 * k] * velocity.qv[k];
                }
            }
        }
        else if (num_dofs.type == DofType::FixedBody) {
            // Fixeds bodies must also be root of their hierarchy
            // tmp_state.sVel = {Vector3::zero(), Vector3::zero()};
            // tmp_state.sAcc = {-physics_state.g, Vector3::zero()};
            if (hier_desc.parent == Entity::none()) {
                tmp_state.sVel = {Vector3::zero(), Vector3::zero()};
                tmp_state.sAcc = {-physics_state.g, Vector3::zero()};
            } else {
                auto parent_tmp_state = ctx.get<DofObjectTmpState>(hier_desc.parent);
                tmp_state.sVel = parent_tmp_state.sVel;
                tmp_state.sAcc = parent_tmp_state.sAcc;
            }
        }
        else if (num_dofs.type == DofType::Slider) {
            assert(hier_desc.parent != Entity::none());
            auto parent_tmp_state = ctx.get<DofObjectTmpState>(hier_desc.parent);
            tmp_state.sVel = parent_tmp_state.sVel;
            tmp_state.sAcc = parent_tmp_state.sAcc;

            // v_i = v_{parent} + S * \dot{q_i}, compute S * \dot{q_i}
            // a_i = a_{parent} + \dot{S} * \dot{q_i} [+ S * \ddot{q_i}, which is 0]
            float *S_dot = computePhiDot(ctx, num_dofs, tmp_state,
                parent_tmp_state.sVel);

            float q_dot = velocity.qv[0];
            for (int j = 0; j < 6; ++j) {
                tmp_state.sVel[j] += S[j] * q_dot;
                tmp_state.sAcc[j] += S_dot[j] * q_dot;
            }
        }
        else if (num_dofs.type == DofType::Hinge) {
            assert(hier_desc.parent != Entity::none());
            auto parent_tmp_state = ctx.get<DofObjectTmpState>(hier_desc.parent);
            tmp_state.sVel = parent_tmp_state.sVel;
            tmp_state.sAcc = parent_tmp_state.sAcc;

            // v_i = v_{parent} + S * \dot{q_i}, compute S * \dot{q_i}
            // a_i = a_{parent} + \dot{S} * \dot{q_i} [+ S * \ddot{q_i}, which is 0]
            // Note: we are using the parent velocity here (for hinge itself)
            float *S_dot = computePhiDot(ctx, num_dofs, tmp_state,
                parent_tmp_state.sVel);

            float q_dot = velocity.qv[0];
            for (int j = 0; j < 6; ++j) {
                tmp_state.sVel[j] += S[j] * q_dot;
                tmp_state.sAcc[j] += S_dot[j] * q_dot;
            }
        }
        else if (num_dofs.type == DofType::Ball) {
            assert(hier_desc.parent != Entity::none());
            auto parent_tmp_state = ctx.get<DofObjectTmpState>(hier_desc.parent);
            tmp_state.sVel = parent_tmp_state.sVel;
            tmp_state.sAcc = parent_tmp_state.sAcc;

            // v_i = v_{parent} + S * \dot{q_i}, compute S * \dot{q_i}
            // a_i = a_{parent} + \dot{S} * \dot{q_i} [+ S * \ddot{q_i}, which is 0]
            // Note: we are using the parent velocity here (for hinge itself)
            float *S_dot = computePhiDot(ctx, num_dofs, tmp_state,
                parent_tmp_state.sVel);

            float *q_dot = velocity.qv;
            for (int j = 0; j < 6; ++j) {
                // TODO: Probably should switch to row major - this isn't
                // particularly cache-friendly.
                tmp_state.sVel[j] += S[j + 6 * 0] * q_dot[0] +
                                     S[j + 6 * 1] * q_dot[1] +
                                     S[j + 6 * 2] * q_dot[2];

                tmp_state.sAcc[j] += S_dot[j + 6 * 0] * q_dot[0] +
                                     S_dot[j + 6 * 1] * q_dot[1] +
                                     S_dot[j + 6 * 2] * q_dot[2];
            }
        } else {
            MADRONA_UNREACHABLE();
        }

        // f_i = I_i a_i + v_i [spatial star cross] I_i v_i
        tmp_state.sForce = tmp_state.spatialInertia.multiply(tmp_state.sAcc);
        tmp_state.sForce += tmp_state.sVel.crossStar(
            tmp_state.spatialInertia.multiply(tmp_state.sVel));
    }

    // Backward pass to find bias forces
    float *tau = body_grp.getBias(ctx);

    memset(tau, 0, total_dofs * sizeof(float));

    CountT dof_index = total_dofs;
    for (CountT i = body_grp.numBodies-1; i >= 0; --i) {
        auto &hier_desc = ctx.get<DofObjectHierarchyDesc>(body_grp.bodies(ctx)[i]);
        auto num_dofs = ctx.get<DofObjectNumDofs>(body_grp.bodies(ctx)[i]);
        auto &tmp_state = ctx.get<DofObjectTmpState>(body_grp.bodies(ctx)[i]);

        // tau_i = S_i^T f_i
        dof_index -= num_dofs.numDofs;
        float *S = tmp_state.getPhiFull(ctx);
        for(CountT row = 0; row < num_dofs.numDofs; ++row) {
            float *S_col = S + 6 * row;
            for(CountT k = 0; k < 6; ++k) {
                tau[dof_index + row] += S_col[k] * tmp_state.sForce[k];
            }
        }

        // Add to parent's force
        if (hier_desc.parent != Entity::none()) {
            auto &parent_tmp_state = ctx.get<DofObjectTmpState>(hier_desc.parent);
            parent_tmp_state.sForce += tmp_state.sForce;
        }
    }
}
#endif

// Add applied/external forces, assuming the forces are applied in joint space
inline void addExternalForces(Context &ctx,
                              BodyGroupHierarchy &body_grp) {
    float *tau = body_grp.getBias(ctx);

    uint32_t dof_index = 0;
    for (CountT i = 0; i < body_grp.numBodies; ++i) {
        auto body_dofs = ctx.get<DofObjectNumDofs>(body_grp.bodies(ctx)[i]);
        auto &ext_force = ctx.get<DofObjectExtForce>(body_grp.bodies(ctx)[i]);
        for(CountT j = 0; j < body_dofs.numDofs; ++j) {
            tau[dof_index + j] += ext_force.force[j];
        }
        dof_index += body_dofs.numDofs;
    }
}


// Sum the diagonals of the mass matrix (used for solver scaling)
inline void sumInertiaDiagonals(Context &ctx,
                                BodyGroupHierarchy &body_grp) {
    float *M = body_grp.getMassMatrix(ctx);
    float inertiaSum = 0.f;
    for (CountT i = 0; i < body_grp.numDofs; ++i) {
        inertiaSum += M[i + i * (body_grp.numDofs)];
    }
    body_grp.inertiaSum = inertiaSum;
}

inline void processContacts(Context &ctx,
                            ContactConstraint &contact,
                            ContactTmpState &tmp_state)
{
    Entity ref = ctx.get<LinkParentDofObject>(contact.ref).parentDofObject;
    Entity alt = ctx.get<LinkParentDofObject>(contact.alt).parentDofObject;

    // If a parent collides with its direct child, unless the parent is a
    //  fixed body, we should ignore the contact.
    auto &refHier = ctx.get<DofObjectHierarchyDesc>(ref);
    auto &altHier = ctx.get<DofObjectHierarchyDesc>(alt);
    if (refHier.parent == alt || altHier.parent == ref) {
        auto &refNumDofs = ctx.get<DofObjectNumDofs>(ref);
        auto &altNumDofs = ctx.get<DofObjectNumDofs>(alt);
        if (refNumDofs.type != DofType::FixedBody
            && altNumDofs.type != DofType::FixedBody) {
            contact.numPoints = 0;
            return;
        }
    }

    DofObjectFriction friction_i = ctx.get<DofObjectFriction>(ref);
    DofObjectFriction friction_j = ctx.get<DofObjectFriction>(alt);

    // Create a coordinate system for the contact
    Vector3 n = contact.normal.normalize();
    Vector3 t{};

    Vector3 x_axis = {1.f, 0.f, 0.f};
    if(n.cross(x_axis).length() > 0.01f) {
        t = n.cross(x_axis).normalize();
    } else {
        t = n.cross({0.f, 1.f, 0.f}).normalize();
    }
    Vector3 s = n.cross(t).normalize();
    tmp_state.C[0] = n;
    tmp_state.C[1] = t;
    tmp_state.C[2] = s;

    // Get friction coefficient
    float mu = fminf(friction_i.muS, friction_j.muS);
    tmp_state.mu = mu;
}

// TODO: This will require a complete rewrite for GPU version.
// Will keep for now for the sake of having first iteration
// of the numerical solver written.
//
// Renaming to brobdingnag for the time beingbecause it does a
// ton of crap - will need to separate.
//
// (https://en.wikipedia.org/wiki/Brobdingnag)
inline void brobdingnag(Context &ctx,
                        CVSolveData &cv_sing)
{
#ifdef MADRONA_GPU_MODE
#define GPU_SINGLE_THREAD if (threadIdx.x % 32 == 0)
#else
#define GPU_SINGLE_THREAD
#endif

    uint32_t world_id = ctx.worldID().idx;

    StateManager *state_mgr = getStateManager(ctx);
    PhysicsSystemState &physics_state = ctx.singleton<PhysicsSystemState>();

    // Recover necessary pointers.
    BodyGroupHierarchy *hiers = state_mgr->getWorldComponents<
        BodyGroup, BodyGroupHierarchy>(world_id);
    CountT num_grps = state_mgr->numRows<BodyGroup>(world_id);

    // Create complete mass matrix
    uint32_t total_num_dofs = 0;
    for (CountT i = 0; i < num_grps; ++i) {
        total_num_dofs += hiers[i].numDofs;
    }

#ifndef MADRONA_GPU_MODE
    CountT num_mass_mat_bytes = sizeof(float) *
        total_num_dofs * total_num_dofs;
    CountT num_full_dofs_bytes = sizeof(float) * total_num_dofs;

    // Row-major
    float *total_mass_mat = (float *)ctx.tmpAlloc(
            num_mass_mat_bytes);
    memset(total_mass_mat, 0, num_mass_mat_bytes);

    float *full_free_acc = (float *)ctx.tmpAlloc(
            total_num_dofs * sizeof(float));
    memset(full_free_acc, 0, num_full_dofs_bytes);

    SparseBlkDiag mass_sparse;
    mass_sparse.fullDim = total_num_dofs;
    mass_sparse.numBlks = num_grps;
    mass_sparse.blks = (SparseBlkDiag::Blk *)ctx.tmpAlloc(
            sizeof(SparseBlkDiag::Blk) * num_grps);

    float *full_vel = (float *)ctx.tmpAlloc(
            num_full_dofs_bytes);
    memset(full_vel, 0, num_full_dofs_bytes);

    uint32_t processed_dofs = 0;

    for (CountT i = 0; i < num_grps; ++i) {
        float *local_mass = hiers[i].getMassMatrix(ctx);

        for (CountT row = 0; row < hiers[i].numDofs; ++row) {
            float *freeAcceleration = hiers[i].getBias(ctx);
            full_free_acc[row + processed_dofs] = freeAcceleration[row];

            for (CountT col = 0; col < hiers[i].numDofs; ++col) {
                uint32_t mi = row + processed_dofs;
                uint32_t mj = col + processed_dofs;

                // The total mass matrix is row major but the
                // local mass matrix is column major.
                total_mass_mat[mj + mi * total_num_dofs] =
                    local_mass[row + hiers[i].numDofs * col];

            }
        }

        processed_dofs += hiers[i].numDofs;
    }

    // Full velocity
    processed_dofs = 0;
    for (CountT i = 0; i < num_grps; ++i) {
        for (CountT j = 0; j < hiers[i].numBodies; ++j) {

            DofObjectVelocity vel = ctx.get<DofObjectVelocity>(
                    hiers[i].bodies(ctx)[j]);
            DofObjectNumDofs num_dofs = ctx.get<DofObjectNumDofs>(
                    hiers[i].bodies(ctx)[j]);

            for (CountT k = 0; k < num_dofs.numDofs; ++k) {
                full_vel[processed_dofs] = vel.qv[k];
                processed_dofs++;
            }

        }
    }
#endif

    // Create the contact Jacobian
    ContactConstraint *contacts = state_mgr->getWorldComponents<
        Contact, ContactConstraint>(world_id);
    ContactTmpState *contacts_tmp_state = state_mgr->getWorldComponents<
        Contact, ContactTmpState>(world_id);

    // Count contacts
#if defined(MADRONA_GPU_MODE)
    CountT num_contacts = state_mgr->numRows<Contact>(world_id);
    CountT total_contact_pts = 0;

    gpu_utils::warpLoopSync(
        num_contacts,
        [&](uint32_t iter) {
            uint32_t v = (iter == 0xFFFF'FFFF) ? 0 : contacts[iter].numPoints;
            total_contact_pts += gpu_utils::warpReduceSum(v);
        });
#else
    CountT num_contacts = state_mgr->numRows<Contact>(world_id);
    CountT total_contact_pts = 0;
    for (int i = 0; i < num_contacts; ++i) {
        total_contact_pts += contacts[i].numPoints;
    }
#endif

    GPU_SINGLE_THREAD {
        // Process mu and penetrations for each point
#ifndef MADRONA_GPU_MODE
        CountT num_full_contact_bytes = sizeof(float) * total_contact_pts;
        float *full_mu = (float *)ctx.tmpAlloc(
                num_full_contact_bytes);
        float *full_penetration = (float *)ctx.tmpAlloc(
                num_full_contact_bytes);
        CountT processed_pts = 0;
        for (int i = 0; i < num_contacts; ++i) {
            ContactTmpState &tmp_state = contacts_tmp_state[i];
            for (int j = 0; j < contacts[i].numPoints; ++j) {
                full_mu[processed_pts] = tmp_state.mu;
                full_penetration[processed_pts] = contacts[i].points[j].w;
                processed_pts++;
            }
        }
#endif

        // Jacobian is size 3n_c x n_dofs, column-major
        uint32_t J_rows = 3 * total_contact_pts;
        uint32_t J_cols = total_num_dofs;

        CountT jac_row = 0;

#if !defined(MADRONA_GPU_MODE)
        uint32_t max_dofs = 0;

        // Prefix sum for each of the body groups
        uint32_t *block_start = (uint32_t *)ctx.tmpAlloc(
                num_grps * sizeof(uint32_t));
        uint32_t block_offset = 0;

        for (CountT i = 0; i < num_grps; ++i) {
            block_start[i] = block_offset;
            block_offset += hiers[i].numDofs;
            hiers[i].tmpIdx0 = i;

            max_dofs = std::max(max_dofs, hiers[i].numDofs);
        }

        float *J_c = (float *) ctx.tmpAlloc(
            J_rows * J_cols * sizeof(float));
        float *diagApprox_c = (float *) ctx.tmpAlloc(
            J_rows * sizeof(float));

        memset(J_c, 0, J_rows * J_cols * sizeof(float));

        float *J_c_body_scratch = (float *)ctx.tmpAlloc(
                3 * max_dofs * sizeof(float));

        for (CountT ct_idx = 0; ct_idx < num_contacts; ++ct_idx) {
            ContactConstraint contact = contacts[ct_idx];
            ContactTmpState &tmp_state = contacts_tmp_state[ct_idx];

            Entity ref = ctx.get<LinkParentDofObject>(contact.ref).parentDofObject;
            Entity alt = ctx.get<LinkParentDofObject>(contact.alt).parentDofObject;

            auto &ref_num_dofs = ctx.get<DofObjectNumDofs>(ref);
            auto &alt_num_dofs = ctx.get<DofObjectNumDofs>(alt);
            uint32_t ref_parent_idx = ctx.get<DofObjectHierarchyDesc>(ref).parentIndex;
            uint32_t alt_parent_idx = ctx.get<DofObjectHierarchyDesc>(alt).parentIndex;

            bool ref_fixed = (ref_num_dofs.type == DofType::FixedBody && ref_parent_idx == -1);
            bool alt_fixed = (alt_num_dofs.type == DofType::FixedBody && alt_parent_idx == -1);

            // Diagonal approximation
            auto &ref_inertial = ctx.get<DofObjectInertial>(ref);
            auto &alt_inertial = ctx.get<DofObjectInertial>(alt);
            float inv_weight = 1.f / (ref_inertial.approxInvMassTrans +
                                      alt_inertial.approxInvMassTrans);

            // Each of the contact points
            for(CountT pt_idx = 0; pt_idx < contact.numPoints; pt_idx++) {
                Vector3 contact_pt = contact.points[pt_idx].xyz();

                // Compute the Jacobians for each body at the contact point
                // if(!ref_fixed) {
                    DofObjectHierarchyDesc &ref_hier_desc = ctx.get<DofObjectHierarchyDesc>(
                            ref);
                    BodyGroupHierarchy &ref_grp = ctx.get<BodyGroupHierarchy>(
                            ref_hier_desc.bodyGroup);

                    float *J_ref = computeContactJacobian(ctx, ref_grp,
                        ref_hier_desc, tmp_state.C, contact_pt, J_c,
                        block_start[ref_grp.tmpIdx0], jac_row, J_rows, -1.f,
                        (ct_idx == 0 && pt_idx == 0));
                // }
                // if(!alt_fixed) {
                    auto &alt_hier_desc = ctx.get<DofObjectHierarchyDesc>(
                            alt);
                    auto &alt_grp = ctx.get<BodyGroupHierarchy>(
                            alt_hier_desc.bodyGroup);

                    float *J_alt = computeContactJacobian(ctx, alt_grp,
                        alt_hier_desc, tmp_state.C, contact_pt, J_c,
                        block_start[alt_grp.tmpIdx0], jac_row, J_rows, 1.f,
                        (ct_idx == 0 && pt_idx == 0));
                // }
                // Compute the diagonal approximation
                diagApprox_c[jac_row] = diagApprox_c[jac_row + 1] =
                    diagApprox_c[jac_row + 2] = inv_weight;

                jac_row += 3;
            }
        }

        cv_sing.J_c = J_c;
        cv_sing.diagApprox_c = diagApprox_c;
#endif

#if !defined(MADRONA_GPU_MODE)
        {
            // This gives us the start in the global array of generalized
            // velocities.
            uint32_t *block_start = (uint32_t *)ctx.tmpAlloc(
                    num_grps * sizeof(uint32_t));
            uint32_t block_offset = 0;

            for (CountT i = 0; i < num_grps; ++i) {
                block_start[i] = block_offset;
                block_offset += hiers[i].numDofs;
                hiers[i].tmpIdx0 = i;

                max_dofs = std::max(max_dofs, hiers[i].numDofs);
            }

            // Starting row in the equality jacobian for each body group
            uint32_t *row_start = (uint32_t *)ctx.tmpAlloc(
                    num_grps * sizeof(uint32_t));
            uint32_t row_offset = 0;
            uint32_t total_num_rows = 0;

            for (uint32_t i = 0; i < num_grps; ++i) {
                row_start[i] = row_offset;
                row_offset += hiers[i].numEqualityRows;

                total_num_rows += hiers[i].numEqualityRows;
            }

            float *J_e = (float *)ctx.tmpAlloc(
                    total_num_rows * total_num_dofs * sizeof(float));
            memset(J_e, 0, total_num_rows * total_num_dofs * sizeof(float));

            float *diagApprox_e = (float *)ctx.tmpAlloc(
                    total_num_rows * sizeof(float));

            float *residuals = (float *)ctx.tmpAlloc(
                    total_num_rows * sizeof(float));
            memset(residuals, 0, total_num_rows * sizeof(float));

            // This is much easier to do with parallel execution
            for (uint32_t grp_idx = 0; grp_idx < num_grps; ++grp_idx) {
                BodyGroupHierarchy &hier = hiers[grp_idx];
                Entity *bodies = hier.bodies(ctx);

                for (uint32_t body_idx = 0; body_idx < hier.numBodies; ++body_idx) {
                    Entity body = bodies[body_idx];
                    DofObjectLimit limit = ctx.get<DofObjectLimit>(body);
                    DofObjectPosition &pos = ctx.get<DofObjectPosition>(body);
                    DofObjectTmpState &tmp_state = ctx.get<DofObjectTmpState>(
                            body);
                    DofObjectInertial &inertial = ctx.get<DofObjectInertial>(
                            body);

                    if (limit.type == DofObjectLimit::Type::None) {
                        continue;
                    }

                    uint32_t glob_row_offset = row_start[grp_idx] +
                                               limit.rowOffset;
                    uint32_t glob_col_offset = block_start[grp_idx] +
                                               tmp_state.dofOffset;

                    switch (limit.type) {
                    case DofObjectLimit::Type::Hinge: {
                        float *to_change = J_e +
                            total_num_rows * glob_col_offset +
                            glob_row_offset;

                        to_change[0] =
                            limit.hinge.dConstraintViolation(pos.q[0]);
                        residuals[glob_row_offset] = limit.hinge.constraintViolation(pos.q[0]);
                        diagApprox_e[glob_row_offset] = 1.f / inertial.approxInvMassDof[0];

                        printf("dviolation = %f; violation = %f\n",
                                to_change[0], residuals[glob_row_offset]);
                    } break;

                    case DofObjectLimit::Type::Slider: {
                        float *to_change = J_e +
                            total_num_rows * glob_col_offset +
                            glob_row_offset;

                        to_change[0] =
                            limit.slider.dConstraintViolation(pos.q[0]);

                        residuals[glob_row_offset] = limit.slider.constraintViolation(pos.q[0]);
                        diagApprox_e[glob_row_offset] = 1.f / inertial.approxInvMassDof[0];

                        printf("dviolation = %f; violation = %f\n",
                                to_change[0], residuals[glob_row_offset]);
                    } break;

                    default: {
                        MADRONA_UNREACHABLE();
                    } break;
                    }
                }
            }

            cv_sing.J_e = J_e;
            cv_sing.numRowsJe = total_num_rows;
            cv_sing.numColsJe = total_num_dofs;
            cv_sing.eqResiduals = residuals;
            cv_sing.diagApprox_e = diagApprox_e;
        }
#endif

        cv_sing.totalNumDofs = total_num_dofs;
        cv_sing.numContactPts = total_contact_pts;
        cv_sing.h = physics_state.h;

#ifndef MADRONA_GPU_MODE
        cv_sing.mass = total_mass_mat;
        cv_sing.freeAcc = full_free_acc;
        cv_sing.vel = full_vel;
        cv_sing.mu = full_mu;
        cv_sing.penetrations = full_penetration;
        cv_sing.dofOffsets = block_start;
#endif

        cv_sing.numBodyGroups = num_grps;

        cv_sing.massDim = total_num_dofs;
        cv_sing.freeAccDim = total_num_dofs;
        cv_sing.velDim = total_num_dofs;
        cv_sing.numRowsJc = J_rows;
        cv_sing.numColsJc = J_cols;
        cv_sing.muDim = total_contact_pts;
        cv_sing.penetrationsDim = total_contact_pts;
    }
}

inline void integrationStep(Context &ctx,
                            DofObjectPosition &position,
                            DofObjectVelocity &velocity,
                            DofObjectAcceleration &acceleration,
                            DofObjectNumDofs &numDofs)
{
    PhysicsSystemState &physics_state = ctx.singleton<PhysicsSystemState>();
    float h = physics_state.h;

    if (numDofs.type == DofType::FreeBody) {
        // Symplectic Euler
        for (int i = 0; i < 6; ++i) {
            velocity.qv[i] += h * acceleration.dqv[i];
        }
        for (int i = 0; i < 3; ++i) {
            position.q[i] += h * velocity.qv[i];
        }

        // From angular velocity to quaternion [Q_w, Q_x, Q_y, Q_z]
        Vector3 omega = { velocity.qv[3], velocity.qv[4], velocity.qv[5] };
        Quat rot_quat = { position.q[3], position.q[4], position.q[5], position.q[6], };
        Quat new_rot = {rot_quat.w, rot_quat.x, rot_quat.y, rot_quat.z};
        new_rot.w += 0.5f * h * (-rot_quat.x * omega.x - rot_quat.y * omega.y - rot_quat.z * omega.z);
        new_rot.x += 0.5f * h * (rot_quat.w * omega.x + rot_quat.z * omega.y - rot_quat.y * omega.z);
        new_rot.y += 0.5f * h * (-rot_quat.z * omega.x + rot_quat.w * omega.y + rot_quat.x * omega.z);
        new_rot.z += 0.5f * h * (rot_quat.y * omega.x - rot_quat.x * omega.y + rot_quat.w * omega.z);
        new_rot = new_rot.normalize();
        position.q[3] = new_rot.w;
        position.q[4] = new_rot.x;
        position.q[5] = new_rot.y;
        position.q[6] = new_rot.z;
    }
    else if (numDofs.type == DofType::Slider) {
        velocity.qv[0] += h * acceleration.dqv[0];
        position.q[0] += h * velocity.qv[0];
    }
    else if (numDofs.type == DofType::Hinge) {
        velocity.qv[0] += h * acceleration.dqv[0];
        position.q[0] += h * velocity.qv[0];
    }
    else if (numDofs.type == DofType::FixedBody) {
        // Do nothing
    }
    else if (numDofs.type == DofType::Ball) {
        // Symplectic Euler
        for (int i = 0; i < 3; ++i) {
            velocity.qv[i] += h * acceleration.dqv[i];
        }

        // From angular velocity to quaternion [Q_w, Q_x, Q_y, Q_z]
        Vector3 omega = { velocity.qv[0], velocity.qv[1], velocity.qv[2] };
        Quat rot_quat = { position.q[0], position.q[1], position.q[2], position.q[3] };
        Quat new_rot = {rot_quat.w, rot_quat.x, rot_quat.y, rot_quat.z};
        new_rot.w += 0.5f * h * (-rot_quat.x * omega.x - rot_quat.y * omega.y - rot_quat.z * omega.z);
        new_rot.x += 0.5f * h * (rot_quat.w * omega.x + rot_quat.z * omega.y - rot_quat.y * omega.z);
        new_rot.y += 0.5f * h * (-rot_quat.z * omega.x + rot_quat.w * omega.y + rot_quat.x * omega.z);
        new_rot.z += 0.5f * h * (rot_quat.y * omega.x - rot_quat.x * omega.y + rot_quat.w * omega.z);
        new_rot = new_rot.normalize();
        position.q[0] = new_rot.w;
        position.q[1] = new_rot.x;
        position.q[2] = new_rot.y;
        position.q[3] = new_rot.z;
    }
    else {
        MADRONA_UNREACHABLE();
    }
}

// Convert all the generalized coordinates here.
inline void convertPostSolve(
        Context &ctx,
        Position &position,
        Rotation &rotation,
        Scale &scale,
        LinkParentDofObject &link)
{
    // TODO: use some forward kinematics results here
    Entity physical_entity = link.parentDofObject;

    DofObjectNumDofs &num_dofs = ctx.get<DofObjectNumDofs>(physical_entity);
    DofObjectTmpState &tmp_state = ctx.get<DofObjectTmpState>(physical_entity);
    DofObjectHierarchyDesc &hier_desc = ctx.get<DofObjectHierarchyDesc>(physical_entity);
    BodyGroupHierarchy &grp_info = ctx.get<BodyGroupHierarchy>(hier_desc.bodyGroup);

    BodyObjectData body_obj_data =
        ((BodyObjectData *)ctx.memoryRangePointer<MRElement128b>(
                grp_info.mrCollisionVisual))[link.mrOffset];

    scale = body_obj_data.scale * hier_desc.globalScale;

    if (num_dofs.type == DofType::FreeBody) {
        position = tmp_state.comPos +
                   hier_desc.globalScale * 
                        tmp_state.composedRot.rotateVec(body_obj_data.offset);
        rotation = tmp_state.composedRot *
                   body_obj_data.rotation;
    }
    else if (num_dofs.type == DofType::Hinge) {
        position = tmp_state.comPos +
                   hier_desc.globalScale * 
                        tmp_state.composedRot.rotateVec(body_obj_data.offset);
        rotation = tmp_state.composedRot *
                   body_obj_data.rotation;
    }
    else if (num_dofs.type == DofType::Slider) {
        position = tmp_state.comPos +
                   hier_desc.globalScale * 
                        tmp_state.composedRot.rotateVec(body_obj_data.offset);
        rotation = tmp_state.composedRot *
                   body_obj_data.rotation;
    }
    else if (num_dofs.type == DofType::FixedBody) {
        // For this, we need to look at the first parent who isn't
        // fixed body and apply its transform.

        position = tmp_state.comPos +
                   hier_desc.globalScale * 
                        tmp_state.composedRot.rotateVec(body_obj_data.offset);
        rotation = tmp_state.composedRot *
                   body_obj_data.rotation;

#if 0
        printf("Fixed body info:\n");
        printf("\t- comPos = %f %f %f\n",
                tmp_state.comPos.x, tmp_state.comPos.y, tmp_state.comPos.z);
        printf("\t- position = %f %f %f\n",
                position.x, position.y, position.z);
#endif

        // Do nothing
    }
    else if (num_dofs.type == DofType::Ball) {
        position = tmp_state.comPos +
                   hier_desc.globalScale * 
                        tmp_state.composedRot.rotateVec(body_obj_data.offset);
        rotation = tmp_state.composedRot *
                   body_obj_data.rotation;
    }
    else {
        MADRONA_UNREACHABLE();
    }
}



}

void registerTypes(ECSRegistry &registry)
{
    registry.registerComponent<DummyComponent>();

    registry.registerSingleton<CVSolveData>();

    registry.registerComponent<DofObjectLimit>();
    registry.registerComponent<DofObjectPosition>();
    registry.registerComponent<DofObjectVelocity>();
    registry.registerComponent<DofObjectAcceleration>();
    registry.registerComponent<DofObjectExtForce>();
    registry.registerComponent<DofObjectNumDofs>();
    registry.registerComponent<DofObjectTmpState>();
    registry.registerComponent<DofObjectInertial>();
    registry.registerComponent<DofObjectFriction>();
    registry.registerComponent<DofObjectHierarchyDesc>();
    registry.registerComponent<ContactTmpState>();
    registry.registerComponent<LinkParentDofObject>();

    registry.registerArchetype<DofObjectArchetype>();
    registry.registerArchetype<Contact>();
    registry.registerArchetype<Joint>();
    registry.registerArchetype<LinkCollider>();
    registry.registerArchetype<LinkVisual>();

    registry.registerComponent<BodyGroupHierarchy>();
    registry.registerArchetype<BodyGroup>();

    registry.registerMemoryRangeElement<MRElement128b>();
    registry.registerMemoryRangeElement<SolverScratch256b>();
}

TaskGraphNodeID setupCVInitTasks(
        TaskGraphBuilder &builder,
        Span<const TaskGraphNodeID> deps)
{
    // Initialize memory and run forward kinematics
    auto node = builder.addToGraph<ParallelForNode<Context,
         tasks::initHierarchies,
         BodyGroupHierarchy
     >>(deps);

    // Initialization for initial position (e.g., inverse weights)
    node = builder.addToGraph<ParallelForNode<Context,
         tasks::computeBodyCOM,
            DofObjectTmpState,
            DofObjectInertial,
            DofObjectNumDofs
        >>({node});

    node = builder.addToGraph<ParallelForNode<Context,
         tasks::computeTotalCOM,
            BodyGroupHierarchy
        >>({node});

    node = builder.addToGraph<ParallelForNode<Context,
         tasks::computeSpatialInertias,
            DofObjectTmpState,
            DofObjectHierarchyDesc,
            DofObjectInertial,
            DofObjectNumDofs
        >>({node});

    node = builder.addToGraph<ParallelForNode<Context,
         tasks::computePhiHierarchy,
            DofObjectTmpState,
            DofObjectNumDofs,
            DofObjectHierarchyDesc
        >>({node});

    node = builder.addToGraph<ParallelForNode<Context,
         tasks::combineSpatialInertias,
            BodyGroupHierarchy
        >>({node});

    node = builder.addToGraph<ParallelForNode<Context,
         tasks::compositeRigidBody,
            BodyGroupHierarchy
        >>({node});

    node = builder.addToGraph<ParallelForNode<Context,
         tasks::factorizeMassMatrix,
            BodyGroupHierarchy
        >>({node});

    node = builder.addToGraph<ParallelForNode<Context,
         tasks::computeInvMass,
         BodyGroupHierarchy
     >>({node});

    node =
        builder.addToGraph<ParallelForNode<Context, tasks::convertPostSolve,
            Position,
            Rotation,
            Scale,
            LinkParentDofObject
        >>({node});

    return node;
}

TaskGraphNodeID setupCVSolverTasks(TaskGraphBuilder &builder,
                                   TaskGraphNodeID broadphase,
                                   CountT num_substeps)
{
    auto cur_node = broadphase;

#ifdef MADRONA_GPU_MODE
    auto sort_sys = builder.addToGraph<
        SortArchetypeNode<BodyGroup, WorldID>>({cur_node});
    sort_sys = builder.addToGraph<
        SortArchetypeNode<DofObjectArchetype, WorldID>>({sort_sys});
    cur_node = builder.addToGraph<ResetTmpAllocNode>({sort_sys});
#endif

    for (CountT i = 0; i < num_substeps; ++i) {
        auto run_narrowphase = narrowphase::setupTasks(builder, {cur_node});

#ifdef MADRONA_GPU_MODE
        // We need to sort the contacts by world.
        run_narrowphase = builder.addToGraph<
            SortArchetypeNode<Contact, WorldID>>(
                {run_narrowphase});
        run_narrowphase = builder.addToGraph<ResetTmpAllocNode>(
                {run_narrowphase});
#endif

        auto compute_body_coms = builder.addToGraph<ParallelForNode<Context,
             tasks::computeBodyCOM,
                DofObjectTmpState,
                DofObjectInertial,
                DofObjectNumDofs
            >>({run_narrowphase});

        auto compute_total_com = builder.addToGraph<ParallelForNode<Context,
             tasks::computeTotalCOM,
                BodyGroupHierarchy
            >>({compute_body_coms});

        auto compute_spatial_inertia = builder.addToGraph<ParallelForNode<Context,
             tasks::computeSpatialInertias,
                DofObjectTmpState,
                DofObjectHierarchyDesc,
                DofObjectInertial,
                DofObjectNumDofs
            >>({compute_total_com});

        auto compute_phi = builder.addToGraph<ParallelForNode<Context,
             tasks::computePhiHierarchy,
                DofObjectTmpState,
                DofObjectNumDofs,
                DofObjectHierarchyDesc
            >>({compute_spatial_inertia});

#ifdef MADRONA_GPU_MODE
        auto recursive_newton_euler = builder.addToGraph<CustomParallelForNode<Context,
             tasks::recursiveNewtonEuler, 32, 1,
                BodyGroupHierarchy
            >>({compute_phi});
#else
        auto recursive_newton_euler = builder.addToGraph<ParallelForNode<Context,
             tasks::recursiveNewtonEuler,
                BodyGroupHierarchy
            >>({compute_phi});
#endif

        auto add_external_forces = builder.addToGraph<ParallelForNode<Context,
             tasks::addExternalForces,
                BodyGroupHierarchy
            >>({recursive_newton_euler});

        auto combine_spatial_inertia = builder.addToGraph<ParallelForNode<Context,
             tasks::combineSpatialInertias,
                BodyGroupHierarchy
            >>({add_external_forces});

        auto composite_rigid_body = builder.addToGraph<ParallelForNode<Context,
             tasks::compositeRigidBody,
                BodyGroupHierarchy
            >>({combine_spatial_inertia});

        auto sum_inertia_diagonals = builder.addToGraph<ParallelForNode<Context,
             tasks::sumInertiaDiagonals,
                BodyGroupHierarchy
            >>({composite_rigid_body});

        auto factorize_mass_matrix = builder.addToGraph<ParallelForNode<Context,
             tasks::factorizeMassMatrix,
                BodyGroupHierarchy
            >>({sum_inertia_diagonals});

        auto compute_free_acc = builder.addToGraph<ParallelForNode<Context,
             tasks::computeFreeAcceleration,
                BodyGroupHierarchy
            >>({factorize_mass_matrix});

        auto contact_node = builder.addToGraph<ParallelForNode<Context,
             tasks::processContacts,
                ContactConstraint,
                ContactTmpState
            >>({compute_free_acc});

#ifdef MADRONA_GPU_MODE
        auto thing = builder.addToGraph<CustomParallelForNode<Context,
             tasks::brobdingnag, 32, 1,
                CVSolveData
            >>({contact_node});
#else
        auto thing = builder.addToGraph<ParallelForNode<Context,
             tasks::brobdingnag,
                CVSolveData
            >>({contact_node});
#endif

#ifdef MADRONA_GPU_MODE
        auto solve = builder.addToGraph<tasks::GaussMinimizationNode>(
                {thing});

#if 0
        solve = builder.addToGraph<
            SortMemoryRangeNode<SolverScratch256b, false>>({solve});
#endif
        solve = builder.addToGraph<ResetTmpAllocNode>({solve});
#else
        auto solve = builder.addToGraph<ParallelForNode<Context,
             tasks::solveCPU,
                CVSolveData
            >>({thing});
#endif

        auto int_node = builder.addToGraph<ParallelForNode<Context,
             tasks::integrationStep,
                 DofObjectPosition,
                 DofObjectVelocity,
                 DofObjectAcceleration,
                 DofObjectNumDofs
            >>({solve});

        auto post_forward_kinematics = builder.addToGraph<ParallelForNode<Context,
             tasks::forwardKinematics,
                BodyGroupHierarchy
            >>({int_node});

        cur_node =
            builder.addToGraph<ParallelForNode<Context, tasks::convertPostSolve,
                Position,
                Rotation,
                Scale,
                LinkParentDofObject
            >>({post_forward_kinematics});

        cur_node = builder.addToGraph<
            ClearTmpNode<Contact>>({cur_node});

        cur_node = builder.addToGraph<ResetTmpAllocNode>({cur_node});
    }

    cur_node = builder.addToGraph<
        ClearTmpNode<CandidateTemporary>>({cur_node});

#ifdef MADRONA_GPU_MODE
    cur_node = builder.addToGraph<
        SortMemoryRangeNode<MRElement128b>>({cur_node});
    cur_node = builder.addToGraph<ResetTmpAllocNode>({cur_node});
#endif

    return cur_node;
}



void getSolverArchetypeIDs(uint32_t *contact_archetype_id,
                           uint32_t *joint_archetype_id)
{
    *contact_archetype_id = TypeTracker::typeID<Contact>();
    *joint_archetype_id = TypeTracker::typeID<Joint>();
}

void init(Context &ctx, CVXSolve *cvx_solve)
{
    ctx.singleton<CVSolveData>().cvxSolve = cvx_solve;
    // ctx.singleton<CVSolveData>().solverScratchMemory = MemoryRange::none();
    // ctx.singleton<CVSolveData>().accRefMemory = MemoryRange::none();
    ctx.singleton<CVSolveData>().scratchAllocatedBytes = 0;
    ctx.singleton<CVSolveData>().accRefAllocatedBytes = 0;
}

Entity makeBodyGroup(Context &ctx, uint32_t num_bodies, float global_scale)
{
    Entity e = ctx.makeEntity<BodyGroup>();

    auto &hier = ctx.get<BodyGroupHierarchy>(e);
    hier.numBodies = num_bodies;
    hier.bodyCounter = 0;
    hier.collisionObjsCounter = 0;
    hier.visualObjsCounter = 0;
    hier.globalScale = global_scale;

    uint64_t mr_num_bytes = num_bodies * sizeof(Entity);
    uint32_t num_elems = (mr_num_bytes + sizeof(MRElement128b) - 1) /
        sizeof(MRElement128b);
    hier.mrBodies = ctx.allocMemoryRange<MRElement128b>(num_elems);

    hier.numEqualityRows = 0;

    return e;
}

Entity makeBody(Context &ctx, Entity body_grp, BodyDesc desc)
{
    auto &grp_info = ctx.get<BodyGroupHierarchy>(body_grp);

    Entity physical_entity = ctx.makeEntity<DofObjectArchetype>();

    auto &pos = ctx.get<DofObjectPosition>(physical_entity);
    auto &vel = ctx.get<DofObjectVelocity>(physical_entity);
    auto &acc = ctx.get<DofObjectAcceleration>(physical_entity);
    auto &ext_force = ctx.get<DofObjectExtForce>(physical_entity);
    auto &tmp_state = ctx.get<DofObjectTmpState>(physical_entity);
    auto &inertial = ctx.get<DofObjectInertial>(physical_entity);
    auto &friction = ctx.get<DofObjectFriction>(physical_entity);
    auto &limit = ctx.get<DofObjectLimit>(physical_entity);
    auto &hier_desc = ctx.get<DofObjectHierarchyDesc>(physical_entity);

    inertial.mass = desc.mass;
    inertial.inertia = desc.inertia;
    friction.muS = desc.muS;

    tmp_state.responseType = desc.responseType;

    tmp_state.numCollisionObjs = desc.numCollisionObjs;
    tmp_state.collisionObjOffset = grp_info.collisionObjsCounter;

    tmp_state.numVisualObjs = desc.numVisualObjs;
    tmp_state.visualObjOffset = grp_info.visualObjsCounter;

    limit.type = DofObjectLimit::Type::None;

    grp_info.collisionObjsCounter += desc.numCollisionObjs;;
    grp_info.visualObjsCounter += desc.numVisualObjs;

    switch ((DofType)desc.type) {
    case DofType::FreeBody: {
        pos.q[0] = desc.initialPos.x;
        pos.q[1] = desc.initialPos.y;
        pos.q[2] = desc.initialPos.z;

        pos.q[3] = desc.initialRot.w;
        pos.q[4] = desc.initialRot.x;
        pos.q[5] = desc.initialRot.y;
        pos.q[6] = desc.initialRot.z;

        for(int i = 0; i < 6; i++) {
            vel.qv[i] = 0.f;
            acc.dqv[i] = 0.f;
            ext_force.force[i] = 0.f;
        }
    } break;

    case DofType::Slider: {
        pos.q[0] = 0.f;
        vel.qv[0] = 0.f;
        acc.dqv[0] = 0.f;
    } break;

    case DofType::Hinge: {
        pos.q[0] = 0.0f;
        vel.qv[0] = 0.f;
        acc.dqv[0] = 0.f;
    } break;

    case DofType::Ball: {
        pos.q[0] = desc.initialRot.w;
        pos.q[1] = desc.initialRot.x;
        pos.q[2] = desc.initialRot.y;
        pos.q[3] = desc.initialRot.z;

        for(int i = 0; i < 3; i++) {
            vel.qv[i] = 0.f;
            acc.dqv[i] = 0.f;
            ext_force.force[i] = 0.f;
        }
    } break;

    case DofType::FixedBody: {
        // Keep these around for forward kinematics
        pos.q[0] = desc.initialPos.x;
        pos.q[1] = desc.initialPos.y;
        pos.q[2] = desc.initialPos.z;

        pos.q[3] = desc.initialRot.w;
        pos.q[4] = desc.initialRot.x;
        pos.q[5] = desc.initialRot.y;
        pos.q[6] = desc.initialRot.z;

        for (int i = 0; i < 6; i++) {
            vel.qv[i] = 0.f;
            acc.dqv[i] = 0.f;
            ext_force.force[i] = 0.f;
        }
    } break;
    }

    ctx.get<DofObjectNumDofs>(physical_entity) = {
        desc.type,
        getNumDofs(desc.type),
    };

    hier_desc.index = grp_info.bodyCounter;
    hier_desc.globalScale = grp_info.globalScale;

    grp_info.bodyCounter++;

    if (grp_info.bodyCounter == grp_info.numBodies) {
        // Now, do allocation for collision / visual info
        uint64_t mr_num_bytes =
            grp_info.collisionObjsCounter * sizeof(BodyObjectData) +
            grp_info.visualObjsCounter * sizeof(BodyObjectData);
        uint32_t num_elems = (mr_num_bytes + sizeof(MRElement128b) - 1) /
            sizeof(MRElement128b);
        grp_info.mrCollisionVisual =
            ctx.allocMemoryRange<MRElement128b>(num_elems);

        grp_info.bodyCounter = 0;
    }

    return physical_entity;
}

void attachCollision(
        Context &ctx,
        Entity body_grp,
        Entity body,
        uint32_t idx,
        CollisionDesc desc)
{
    BodyGroupHierarchy &grp_info = ctx.get<BodyGroupHierarchy>(body_grp);
    DofObjectTmpState &tmp_state = ctx.get<DofObjectTmpState>(body);

    BodyObjectData *col_data = grp_info.getCollisionData(ctx);

    Entity col_obj = ctx.makeEntity<LinkCollider>();

    ctx.get<broadphase::LeafID>(col_obj) =
        PhysicsSystem::registerEntity(ctx, col_obj, { (int32_t)desc.objID });
    ctx.get<ResponseType>(col_obj) = tmp_state.responseType;
    ctx.get<ObjectID>(col_obj) = { (int32_t)desc.objID };

    ctx.get<Velocity>(col_obj) = {
        Vector3::zero(),
        Vector3::zero(),
    };

    ctx.get<ExternalForce>(col_obj) = Vector3::zero();
    ctx.get<ExternalTorque>(col_obj) = Vector3::zero();

    ctx.get<LinkParentDofObject>(col_obj) = {
        .parentDofObject = body,
        .mrOffset = tmp_state.collisionObjOffset + idx,
    };

    col_data[tmp_state.collisionObjOffset + idx] = {
        col_obj,
        desc.offset,
        desc.rotation,
        desc.scale,
    };
}

void attachVisual(
        Context &ctx,
        Entity body_grp,
        Entity body,
        uint32_t idx,
        VisualDesc desc)
{
    BodyGroupHierarchy &grp_info = ctx.get<BodyGroupHierarchy>(body_grp);
    DofObjectTmpState &tmp_state = ctx.get<DofObjectTmpState>(body);

    BodyObjectData *viz_data = grp_info.getVisualData(ctx);

    Entity viz_obj = ctx.makeEntity<LinkVisual>();

    ctx.get<ObjectID>(viz_obj) = { (int32_t)desc.objID };

    // Make this entity renderable
    render::RenderingSystem::makeEntityRenderable(ctx, viz_obj);

    ctx.get<LinkParentDofObject>(viz_obj) = {
        .parentDofObject = body,
        .mrOffset = grp_info.collisionObjsCounter +
                    tmp_state.visualObjOffset + idx,
    };

    viz_data[tmp_state.visualObjOffset + idx] = {
        viz_obj,
        desc.offset,
        desc.rotation,
        desc.scale,
    };
}

void attachLimit(Context &ctx,
                 Entity grp,
                 Entity body,
                 HingeLimit hinge_limit)
{
    BodyGroupHierarchy &hier = ctx.get<BodyGroupHierarchy>(grp);
    DofObjectLimit &limit = ctx.get<DofObjectLimit>(body);

    limit.type = DofObjectLimit::Type::Hinge;
    limit.hinge = hinge_limit;
    limit.rowOffset = hier.numEqualityRows;

    hier.numEqualityRows += 1;
}

void attachLimit(Context &ctx,
                 Entity grp,
                 Entity body,
                 SliderLimit slider_limit)
{
    BodyGroupHierarchy &hier = ctx.get<BodyGroupHierarchy>(grp);
    DofObjectLimit &limit = ctx.get<DofObjectLimit>(body);

    limit.type = DofObjectLimit::Type::Slider;
    limit.slider = slider_limit;
    limit.rowOffset = hier.numEqualityRows;

    hier.numEqualityRows += 1;
}

void setRoot(Context &ctx,
             Entity body_group,
             Entity body)
{
    auto &hierarchy = ctx.get<DofObjectHierarchyDesc>(body);
    hierarchy.leaf = true;
    hierarchy.index = 0;
    hierarchy.parentIndex = -1;
    hierarchy.parent = Entity::none();
    hierarchy.bodyGroup = body_group;

    auto &body_grp_hier = ctx.get<BodyGroupHierarchy>(body_group);

    body_grp_hier.bodyCounter = 1;
    body_grp_hier.bodies(ctx)[0] = body;
    body_grp_hier.numDofs = ctx.get<DofObjectNumDofs>(body).numDofs;
}

static inline void joinBodiesGeneral(
        Context &ctx,
        Entity body_grp,
        Entity parent_physics_entity,
        Entity child_physics_entity,
        Vector3 rel_position_parent,
        Vector3 rel_position_child,
        Quat rel_parent_rotation,
        Vector3 axis = Vector3 { 0.f, 0.f, 0.f })
{
    BodyGroupHierarchy &grp = ctx.get<BodyGroupHierarchy>(body_grp);

    auto &hier_desc = ctx.get<DofObjectHierarchyDesc>(child_physics_entity);
    auto &parent_hier_desc =
        ctx.get<DofObjectHierarchyDesc>(parent_physics_entity);

    hier_desc.parent = parent_physics_entity;
    hier_desc.relPositionParent = rel_position_parent;
    hier_desc.relPositionLocal = rel_position_child;
    hier_desc.parentToChildRot = rel_parent_rotation;

    hier_desc.axis = axis;

    hier_desc.leaf = true;
    hier_desc.bodyGroup = body_grp;

    // hier_desc.index = grp.bodyCounter;
    hier_desc.parentIndex = parent_hier_desc.index;

    grp.bodies(ctx)[hier_desc.index] = child_physics_entity;

    // Make the parent no longer a leaf
    ctx.get<DofObjectHierarchyDesc>(parent_physics_entity).leaf = false;

    grp.numDofs += ctx.get<DofObjectNumDofs>(child_physics_entity).numDofs;
}

void joinBodies(Context &ctx,
                Entity body_grp,
                Entity parent_physics_entity,
                Entity child_physics_entity,
                JointHinge hinge_info)
{
    joinBodiesGeneral(ctx, 
                      body_grp,
                      parent_physics_entity,
                      child_physics_entity,
                      hinge_info.relPositionParent,
                      hinge_info.relPositionChild,
                      hinge_info.relParentRotation,
                      hinge_info.hingeAxis);
}

void joinBodies(Context &ctx,
                Entity body_grp,
                Entity parent_physics_entity,
                Entity child_physics_entity,
                JointBall ball_info)
{
    joinBodiesGeneral(ctx, 
                      body_grp,
                      parent_physics_entity,
                      child_physics_entity,
                      ball_info.relPositionParent,
                      ball_info.relPositionChild,
                      ball_info.relParentRotation);
}

void joinBodies(
        Context &ctx,
        Entity body_grp,
        Entity parent_physics_entity,
        Entity child_physics_entity,
        JointSlider slider_info)
{
    joinBodiesGeneral(ctx, 
                      body_grp,
                      parent_physics_entity,
                      child_physics_entity,
                      slider_info.relPositionParent,
                      slider_info.relPositionChild,
                      slider_info.relParentRotation,
                      slider_info.slideVector);
}

void joinBodies(Context &ctx,
                 Entity body_grp,
                 Entity parent_physics_entity,
                 Entity child_physics_entity,
                 JointFixed fixed_info)
{
    joinBodiesGeneral(ctx, 
                      body_grp,
                      parent_physics_entity,
                      child_physics_entity,
                      fixed_info.relPositionParent,
                      fixed_info.relPositionChild,
                      fixed_info.relParentRotation);
}

Entity loadModel(Context &ctx,
                 ModelConfig cfg,
                 ModelData model_data,
                 Vector3 initial_pos,
                 Quat initial_rot,
                 float global_scale)
{
    Entity grp = makeBodyGroup(ctx, cfg.numBodies, global_scale);
    Entity *bodies_tmp =
        (Entity *)ctx.tmpAlloc(sizeof(Entity) * cfg.numBodies);

    { // Make the root
        BodyDesc desc = model_data.bodies[cfg.bodiesOffset];

        desc.initialPos = initial_pos;
        desc.initialRot = initial_rot;

        bodies_tmp[0] = makeBody(
                ctx,
                grp,
                desc);
    }

    // Create the bodies (links)
    for (uint32_t i = 1; i < cfg.numBodies; ++i) {
        bodies_tmp[i] = makeBody(
                ctx,
                grp,
                model_data.bodies[cfg.bodiesOffset + i]);
    }

    // Attach the colliders
    for (uint32_t i = 0; i < cfg.numColliders; ++i) {
        CollisionDesc desc = model_data.colliders[cfg.collidersOffset + i];

        attachCollision(
                ctx,
                grp,
                bodies_tmp[desc.linkIdx],
                desc.subIndex,
                desc);
    }

    // Attach the visuals
    for (uint32_t i = 0; i < cfg.numVisuals; ++i) {
        VisualDesc desc = model_data.visuals[cfg.visualsOffset + i];

        attachVisual(
                ctx,
                grp,
                bodies_tmp[desc.linkIdx],
                desc.subIndex,
                desc);
    }

    { // Create the hierarchy
        setRoot(ctx, grp, bodies_tmp[0]);

        for (uint32_t i = 0; i < cfg.numConnections; ++i) {
            JointConnection conn =
                model_data.connections[cfg.connectionsOffset + i];

            switch (conn.type) {
            case DofType::Hinge: {
                joinBodies(
                        ctx,
                        grp,
                        bodies_tmp[conn.parentIdx],
                        bodies_tmp[conn.childIdx],
                        conn.hinge);
            } break;

            case DofType::Ball: {
                joinBodies(
                        ctx,
                        grp,
                        bodies_tmp[conn.parentIdx],
                        bodies_tmp[conn.childIdx],
                        conn.ball);
            } break;

            case DofType::Slider: {
                joinBodies(
                        ctx,
                        grp,
                        bodies_tmp[conn.parentIdx],
                        bodies_tmp[conn.childIdx],
                        conn.slider);
            } break;

            case DofType::FixedBody: {
                joinBodies(
                        ctx,
                        grp,
                        bodies_tmp[conn.parentIdx],
                        bodies_tmp[conn.childIdx],
                        conn.fixed);
            } break;

            default: {
                // Not supported yet
                assert(false);
            } break;

            }
        }
    }

    return grp;
}

void addHingeExternalForce(Context &ctx, Entity hinge_joint, float newtons)
{
    ctx.get<DofObjectExtForce>(hinge_joint).force[0] = newtons;
}

}
