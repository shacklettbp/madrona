#ifdef MADRONA_GPU_MODE
namespace madrona::phys::cv {

namespace gpu_utils {
    
// For debugging purposes
template <bool transposed, bool host_print>
void printMatrix(float *mat,
                 uint32_t num_rows,
                 uint32_t num_cols,
                 const char *name)
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

static void warpSetZero(void *dst, uint32_t num_bytes)
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

static void warpCopy(void *dst, void *src, uint32_t num_bytes, bool dbg)
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
          bool reset_res>
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

template <typename DataT, typename ReadFn, typename WriteFn>
void warpExclusivePrefixSumPred(
        ReadFn &&read_fn,
        WriteFn &&write_fn,
        uint32_t num_elems)
{
    uint32_t processed = 0;

    warpLoopSync(
        num_elems,
        [&](uint32_t iter) {
            DataT val = (iter == 0xFFFF'FFFF) ? 0 : read_fn(iter);
            DataT ipf = warpInclusivePrefixSum(val);

            if (iter != 0xFFFF'FFFF) {
                DataT epf = processed + ipf - val;
                write_fn(iter, epf);
            }

            processed += __shfl_sync(0xFFFF'FFFF, ipf, 31);
        });
    __syncwarp();
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
}
#endif
