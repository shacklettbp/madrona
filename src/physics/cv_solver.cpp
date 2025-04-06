#include "cv.hpp"
#include "cv_gpu.hpp"
#include "cv_cpu.hpp"
#include "physics_impl.hpp"

#define dbg_warp_printf(...) if (dbg) { warp_printf(__VA_ARGS__); }
#define dbg_matrix_printf(...) if (dbg) { printMatrix(__VA_ARGS__); }

using namespace madrona::math;

namespace madrona::phys::cv {

namespace tasks {
#ifdef MADRONA_GPU_MODE

struct ConeSpace {
    float n;
    float t1;
    float t2;
    float t;
    float mu;
    float midWeight;
};

inline float square(float x)
{
    return x * x;
}

inline float cube(float x)
{
    return x * x * x;
}
    
struct GaussMinimizationNode : NodeBase {
    GaussMinimizationNode(StateManager *state_mgr);

    inline ConeSpace mapToConeSpace(float *jar, float *mus, uint32_t idx);

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
        float *jaccref_cont,
        float *jaccref_eq,
        float *Mxmin,
        float *acc_ref_cont,
        float *acc_ref_eq,
        bool dbg = false);

    // Call this after calling dobjWarp
    float objWarp(
        float *x,
        CVSolveData *sd,
        float *jaccref_cont,
        float *jaccref_eq,
        float *Mxmin);

    float exactLineSearch(
            CVSolveData *sd,
            float *jaccref_cont,
            float *jaccref_eq,
            float *Mxmin,
            float *p,
            float *x,
            float tol,
            float *scratch,
            float *d_c,
            float *d_e,
            bool dbg = false);

    void computeContactJacobian(BodyGroupProperties &prop,
                                BodyGroupMemory &mem,
                                uint32_t body_idx,
                                Mat3x3 &C,
                                Vector3 &origin,
                                float *J,
                                uint32_t body_dof_offset,
                                uint32_t jac_row,
                                uint32_t j_num_rows,
                                float coeff,
                                bool dbg);

    template <typename Fn>
    void computeAccRef(
            float *acc_ref_cont,
            uint32_t acc_ref_dim,
            float *j_mat,
            uint32_t num_rows_j,
            uint32_t num_cols_j,
            float *vel,
            uint32_t vel_dim,
            float h,
            Fn &&residual_fn,
            float *diag_approx,
            float *r_vec,
            bool dbg = false);

    void calculateSolverDims(uint32_t world_id, CVSolveData *sd);
    void prepareRegInfos(CVSolveData *sd);



    // Nodes in the taskgraph:
    void allocateScratch(int32_t invocation_idx);
    // Prepares mass matrix and contact jacobian
    void prepareSolver(int32_t invocation_idx);
    void computeContactAccRef(int32_t invocation_idx);
    void computeEqualityAccRef(int32_t invocation_idx);
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
    float *jaccref_cont,
    float *jaccref_eq,
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

    // We don't actually need to pass in the state manager
    float *dc = sd->getContactR(nullptr);
    float *de = sd->getEqualityR(nullptr);

    // Calculate s(...) now:
    warpLoopSync(sd->numRowsJc / 3, [&](uint32_t iter) {
        float curr_val = 0.f;
        if (iter != 0xFFFF'FFFF) {
#if 0
            float n = jaccref_cont[iter * 3];
            float t1 = jaccref_cont[iter * 3 + 1];
            float t2 = jaccref_cont[iter * 3 + 2];

            float t = sqrtf(t1 * t1 + t2 * t2);
            float mu = sd->mu[iter];
            float mw = 1.f / (1.f + mu * mu);
#endif

            auto [n, t1, t2, t, mu, mw] = mapToConeSpace(
                    jaccref_cont, sd->mu, iter);

            float dn = dc[3 * iter + 0];
            float d1 = dc[3 * iter + 1];
            float d2 = dc[3 * iter + 2];

            if (n >= mu * t || (t <= 0.f && 0.f <= n)) {
                // Do nothing (top zone)
            } else if (mu * n + t <= 0.f || (t <= 0.f && n < 0.f)) {
                // Bottom zone
#if 0
                curr_val = 0.5f * (dn * n * n + 
                                   d1 * t1 * t1 +
                                   d2 * t2 * t2);
#endif
                curr_val = 
                    0.5f * (dn * jaccref_cont[iter * 3 + 0] * jaccref_cont[iter * 3 + 0] +
                            d1 * jaccref_cont[iter * 3 + 1] * jaccref_cont[iter * 3 + 1] + 
                            d2 * jaccref_cont[iter * 3 + 2] * jaccref_cont[iter * 3 + 2]);
            } else {
                // Middle zone
                curr_val = 0.5f * dn * mw * (n - mu * t) * (n - mu * t);
            }
        }

        res += warpReduceSum(curr_val);
    });

    warpLoopSync(sd->numRowsJe, [&](uint32_t iter) {
        float curr_val = 0.f;
        if (iter != 0xFFFF'FFFF) {
            curr_val = 0.5f * de[iter] * jaccref_eq[iter] * jaccref_eq[iter];
        }

        res += warpReduceSum(curr_val);
    });

    return res;
}

inline ConeSpace GaussMinimizationNode::mapToConeSpace(
        float *jar, float *mus, uint32_t idx)
{
    float jar_n = jar[3 * idx + 0];
    float jar_t1 = jar[3 * idx + 1];
    float jar_t2 = jar[3 * idx + 2];

    float mu = mus[3 * idx + 0];
    float mu1 = mus[3 * idx + 1];
    float mu2 = mus[3 * idx + 2];

    float n = jar_n * mu;
    float t1 = jar_t1 * mu1;
    float t2 = jar_t2 * mu2;

    float t = sqrtf(t1 * t1 + t2 * t2);
    float mid_weight = 1.f / (mu * mu * (1.f + mu * mu));

    return {
        n, t1, t2, t, mu, mid_weight
    };
}

template <bool calc_package>
void GaussMinimizationNode::dobjWarp(
        float *res,
        float *x,
        CVSolveData *sd,
        float *scratch,
        float *jaccref_cont,
        float *jaccref_eq,
        float *Mxmin,
        float *acc_ref_cont,
        float *acc_ref_eq,
        bool dbg)
{
    using namespace gpu_utils;

#if 0
    if (threadIdx.x % 32 == 0) {
        LOG("(world_id = {}) before warpCopy (freeAccDim = {}): scratch ({}), x ({}), freeAcc ({})\n",
                sd->worldID,
                sd->freeAccDim,
                scratch,
                x,
                sd->freeAcc);
    }
#endif

    // x - acc_free
    warpLoop(sd->freeAccDim, [&](uint32_t iter) {
        scratch[iter] = x[iter] - sd->freeAcc[iter];
    });

    __syncwarp();

#if 0
    if (threadIdx.x % 32 == 0) {
        LOG("after warpCopy\n");
    }

    if (threadIdx.x % 32 == 0) {
        LOG("before sparseBlkDiagSmallReg\n");
    }
#endif

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
    }

    gmmaWarpSmallReg<float, 4, true, false, true>(
            scratch,
            sd->J_c,
            x,
            sd->numRowsJc,
            sd->numColsJc,
            sd->freeAccDim,
            1);
    __syncwarp();

    warpLoop(sd->numRowsJc, [&](uint32_t iter) {
        scratch[iter] -= acc_ref_cont[iter];
    });
    __syncwarp();

    if constexpr (calc_package) {
        warpCopy(jaccref_cont, scratch, sd->numRowsJc * sizeof(float), dbg);
        __syncwarp();
    }

    float *dc = sd->getContactR(nullptr);
    float *de = sd->getEqualityR(nullptr);

    // ds
    warpLoop(sd->numRowsJc / 3, [&](uint32_t iter) {
#if 0
        float n = scratch[iter * 3];
        float t1 = scratch[iter * 3 + 1];
        float t2 = scratch[iter * 3 + 2];

        float t = sqrtf(t1 * t1 + t2 * t2);
        float mu = sd->mu[iter];
        float mw = 1.f / (1.f + mu * mu);
#endif

        auto [n, t1, t2, t, mu, mw] = mapToConeSpace(
                scratch, sd->mu, iter);

        float dn = dc[3 * iter + 0];
        float d1 = dc[3 * iter + 1];
        float d2 = dc[3 * iter + 2];

        if (n >= mu * t || (t <= 0.f && 0 <= n)) {
            scratch[3 * iter] = 0.f;
            scratch[3 * iter + 1] = 0.f;
            scratch[3 * iter + 2] = 0.f;
        } else if (mu * n + t <= 0.f || (t <= 0.f && n < 0.f)) {
            scratch[3 * iter] = dn * scratch[3 * iter];
            scratch[3 * iter + 1] = d1 * scratch[3 * iter + 1];
            scratch[3 * iter + 2] = d2 * scratch[3 * iter + 2];
        } else {
            float tmp = dn * mw * (n - mu * t) * mu;
            scratch[3 * iter] = tmp;

            if (t == 0.f) {
                printf("nan going to happen dobj!\n");
            }

            scratch[3 * iter + 1] = -(tmp / t) * t1 * sd->mu[3 * iter + 1];
            scratch[3 * iter + 2] = -(tmp / t) * t2 * sd->mu[3 * iter + 2];
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



    // Get equality part of the gradient
    gmmaWarpSmallReg<float, 4, true, false, true>(
            scratch,
            sd->J_e,
            x,
            sd->numRowsJe,
            sd->numColsJe,
            sd->freeAccDim,
            1);
    __syncwarp();

    warpLoop(sd->numRowsJe, [&](uint32_t iter) {
        scratch[iter] -= acc_ref_eq[iter];
    });
    __syncwarp();

    if constexpr (calc_package) {
        warpCopy(jaccref_eq, scratch, sd->numRowsJe * sizeof(float), dbg);
        __syncwarp();
    }

    warpLoop(sd->numRowsJe, [&](uint32_t iter) {
        scratch[iter] = de[iter] * scratch[iter];
    });

    gmmaWarpSmallReg<float, 4, false, false, false>(
            res,
            sd->J_e,
            scratch,
            sd->numColsJe,
            sd->numRowsJe,
            sd->numRowsJe,
            1);
    __syncwarp();
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
        float *jaccref_cont,
        float *jaccref_eq,
        float *Mxmin,
        float *p,
        float *x,
        float tol,
        float *scratch,
        float *d_c,
        float *d_e,
        bool dbg)
{
    using namespace gpu_utils;

    float pMx_free = dotVectors(p, Mxmin, sd->freeAccDim);

    dbg_warp_printf("@@@@@@@@@@@@@ ENTERING FAUlTY LINE SEARCH\n");

    dbg_matrix_printf(p, 1, sd->freeAccDim, "p");
    dbg_matrix_printf(Mxmin, 1, sd->freeAccDim, "Mx_free");
    dbg_matrix_printf(jaccref_cont, 1, sd->numRowsJc, "jaccref");

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

    float *Jep = scratch;
    { // Calculate Je @ p
        // printMatrix(p, 1, sd->freeAccDim, "p");
        // printMatrix<true>(sd->J_e, sd->numRowsJe, sd->numColsJe, "J_e");
        gmmaWarpSmallReg<float, 4, true, false, true>(
            Jep, sd->J_e, p,
            sd->numRowsJe, sd->numColsJe,
            sd->numColsJe, 1);
    }
    __syncwarp();

    // printMatrix(jaccref_eq, 1, sd->numRowsJe, "Jaccref_eq");
    // printMatrix(Jep, 1, sd->numRowsJe, "Jep");

    // function, first deriv and second deriv evals
    struct Evals {
        float fun;
        float grad;
        float hess;
    };

    float *mus = sd->mu;

   auto fdh_phi = [&](float alpha, bool print = false) -> Evals {
        float fun = 0.5f * alpha * alpha * pMp +
                    alpha * pMx_free + 0.5f * xmin_M_xmin;
        float grad = alpha * pMp + pMx_free;
        float hess = pMp;

        if (print) {
            warp_printf("fun0 = %f; grad0 = %f; hess0 = %f\n",
                    fun, grad, hess);
        }

        struct Diff {
            float dfun;
            float dgrad;
            float dhess;
        };

        warpLoopSync(sd->numRowsJc / 3, [&](uint32_t iter) {
            auto d = [&]() -> Diff {
                if (iter == 0xFFFF'FFFF) {
                    return {
                        0.f, 0.f, 0.f
                    };
                } else {
                    float jx_n = jaccref_cont[iter * 3];
                    float jx_t1 = jaccref_cont[iter * 3 + 1];
                    float jx_t2 = jaccref_cont[iter * 3 + 2];

                    float mu = mus[iter * 3];
                    float mu1 = mus[iter * 3 + 1];
                    float mu2 = mus[iter * 3 + 2];

                    float dn = d_c[iter * 3];
                    float d1 = d_c[iter * 3 + 1];
                    float d2 = d_c[iter * 3 + 2];

                    // float mu = sd->mu[iter];
                    float mw = 1.f / (mu * mu * (1.f + mu * mu));

                    float jp_n = Jp[iter * 3];
                    float jp_t1 = Jp[iter * 3 + 1];
                    float jp_t2 = Jp[iter * 3 + 2];

                    float jacc_n = jx_n + alpha * jp_n;
                    float jacc_t1 = jx_t1 + alpha * jp_t1;
                    float jacc_t2 = jx_t2 + alpha * jp_t2;

                    float n_search = jacc_n * mu;
                    float t1_search = jacc_t1 * mu1;
                    float t2_search = jacc_t2 * mu2;

                    float t_search = sqrtf(t1_search * t1_search +
                                           t2_search * t2_search);

                    if (n_search >= mu * t_search || (t_search <= 0.f && 0.f <= n_search)) {
                        // Don't add anything up
                        return {0.f, 0.f, 0.f};

                        if (print)
                            printf("iter = %d; first return (0)\n", iter);
                    } else if (mu * n_search + t_search <= 0.f || 
                            (t_search <= 0.f && n_search < 0.f)) {
#if 0
                        float p_sq = p0 * p0 + p1 * p1 + p2 * p2;
#endif

                        float p_sq = dn * (jp_n * jp_n) + 
                                     d1 * (jp_t1 * jp_t1) +
                                     d2 * (jp_t2 * jp_t2);

                        Diff diff = {
                            // Fun
                            0.5f * (dn * (jacc_n * jacc_n) +
                                    d1 * (jacc_t1 * jacc_t1) +
                                    d2 * (jacc_t2 * jacc_t2)),
                            // Grad
                            (dn * jp_n * jx_n + 
                             d1 * jp_t1 * jx_t1 +
                             d2 * jp_t2 * jx_t2 +
                             alpha * p_sq),
                            // Hess
                            p_sq
                        };

                        if (print)
                            printf("iter = %d; second return; f=%f; g=%f; h=%f\n",
                                    iter, diff.dfun, diff.dgrad, diff.dhess);

                        return diff;
                    } else {
                        float dn_da = jp_n * mu;
                        float dt_da = (square(mu1) * jp_t1 * jacc_t1 +
                                       square(mu2) * jp_t2 * jacc_t2) / t_search;
                        float d2tp_da2 = ((square(mu1) * square(mu2) *
                                           square(jp_t2 * jx_t1 - jp_t1 * jx_t2)) / cube(t_search));
                        float tmp = n_search - mu * t_search;
                        float d_tmp = dn_da - mu * dt_da;

                        Diff diff = {
                            0.5f * dn * mw * square(n_search - mu * t_search),
                            dn * mw * tmp * d_tmp,
                            dn * mw * (square(d_tmp) - tmp * mu * d2tp_da2),
                        };

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

        warpLoopSync(sd->numRowsJe, [&](uint32_t iter) {
            auto d = [&]() -> Diff {
                if (iter == 0xFFFF'FFFF) {
                    return {
                        0.f, 0.f, 0.f
                    };
                } else {
                    float orig = jaccref_eq[iter];
                    float dj = Jep[iter];

                    float de = d_e[iter];
                    float n_search = orig + alpha * dj;

                    return {
                        0.5f * de * square(n_search),
                        de * (orig * dj + alpha * square(dj)),
                        de * square(dj)
                    };
                }
            } ();

            float dfun = warpReduceSum(d.dfun);
            float dgrad = warpReduceSum(d.dgrad);
            float dhess = warpReduceSum(d.dhess);

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

void GaussMinimizationNode::calculateSolverDims(
        uint32_t world_id,
        CVSolveData *sd)
{
    using namespace gpu_utils;

    uint32_t lane_id = threadIdx.x % 32;

    StateManager *state_mgr = getStateManager();

    BodyGroupProperties *all_properties = state_mgr->getWorldComponents<
        BodyGroupArchetype, BodyGroupProperties>(world_id);
    BodyGroupMemory *all_memories = state_mgr->getWorldComponents<
        BodyGroupArchetype, BodyGroupMemory>(world_id);
    CountT num_grps = state_mgr->numRows<BodyGroupArchetype>(world_id);

    if (lane_id == 0) { // Get num groups
        sd->numBodyGroups = num_grps;
    }

    __syncwarp();

    { // Get total num dofs
        uint32_t total_num_dofs = warpSumPred<uint32_t>(
            [&](uint32_t iter) {
                return all_properties[iter].qvDim;
            }, sd->numBodyGroups);

        if (lane_id == 0) {
            sd->totalNumDofs = total_num_dofs;
            sd->freeAccDim = total_num_dofs;
            sd->velDim = total_num_dofs;
        }
    }

    { // Get num contact points
        ContactConstraint *contacts = state_mgr->getWorldComponents<
            Contact, ContactConstraint>(world_id);
        CountT num_contacts = state_mgr->numRows<Contact>(world_id);

        uint32_t total_num_contact_pts = warpSumPred<uint32_t>(
            [&](uint32_t iter) {
                return contacts[iter].numPoints;
            }, num_contacts);

        if (lane_id == 0) {
            sd->numContactPts = total_num_contact_pts;
            sd->numRowsJc = 3 * total_num_contact_pts;
            sd->numColsJc = sd->totalNumDofs;
            sd->muDim = total_num_contact_pts;
            sd->penetrationsDim = total_num_contact_pts;
        }
    }

    if (lane_id == 0) { // Get physics step
        sd->h = state_mgr->getSingleton<PhysicsSystemState>({(int32_t)world_id}).h;
    }

    __syncwarp();

    { // Get equality jacobian dims
        uint32_t total_num_rows = warpSumPred<uint32_t>(
            [&](uint32_t iter) {
                return all_properties[iter].numEq;
            }, num_grps);

        if (lane_id == 0) {
            sd->numRowsJe = total_num_rows;
            sd->numColsJe = sd->totalNumDofs;
        }
    }
}

void GaussMinimizationNode::allocateScratch(int32_t invocation_idx)
{
    CV_PROF_START(t0, allocScratch)

    { // Solver dims
        uint32_t world_id = invocation_idx;
        CVSolveData *curr_sd = &solveDatas[world_id];

        calculateSolverDims(world_id, curr_sd);

        if (curr_sd->freeAccDim == 0) {
            return;
        }
    }

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
        // scratch6
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


        { // AccRef allocation
            uint32_t cont_acc_ref_bytes = sizeof(float) * curr_sd->numRowsJc;
            uint32_t eq_acc_ref_bytes = sizeof(float) * curr_sd->numRowsJe;

            curr_sd->accRefMem = (uint8_t *)mwGPU::TmpAllocator::get().alloc(
                    cont_acc_ref_bytes + eq_acc_ref_bytes);
            curr_sd->accRefAllocatedBytes = cont_acc_ref_bytes + eq_acc_ref_bytes;
        }


        // TODO: Compare performance of tmp alloc vs the memory range
        // allocator.
        //
        // Also, separate out the memory between frame dependent and
        // non-frame dependent allocations.
        //
        // Any tensor which depends on the number of contacts should
        // be in separate allocations from those who don't depend.

        { // Mass matrix allocation
            CountT num_grps = state_mgr->numRows<BodyGroupArchetype>(world_id);

            uint32_t total_num_dofs = curr_sd->totalNumDofs;
            uint32_t total_contact_pts = curr_sd->numContactPts;

            uint32_t prep_bytes =
                // Mass matrix
                sizeof(SparseBlkDiag::Blk) * num_grps +
                // full vel
                sizeof(float) * total_num_dofs +
                // free acc
                sizeof(float) * total_num_dofs +
                // mu
                sizeof(float) * total_contact_pts * 3 +
                // penetrations
                sizeof(float) * total_contact_pts * 3 +
                // J_c    TODO: Make this sparse
                sizeof(float) * 3 * total_contact_pts * total_num_dofs +
                // diag approx of J_c
                sizeof(float) * 3 * total_contact_pts +
                // J_e    TODO: Make this sparse
                sizeof(float) * curr_sd->numRowsJe * curr_sd->numColsJe +
                // diag approx of J_e
                sizeof(float) * curr_sd->numRowsJe +
                // Equality residuals
                sizeof(float) * curr_sd->numRowsJe +
                // R_c
                sizeof(float) * curr_sd->numRowsJc +
                // R_e
                sizeof(float) * curr_sd->numRowsJe;

            curr_sd->prepMem = (uint8_t *)mwGPU::TmpAllocator::get().alloc(
                    prep_bytes);
            curr_sd->prepAllocatedBytes = prep_bytes;
        }
    }
}

void GaussMinimizationNode::prepareSolver(int32_t invocation_idx)
{
    CV_PROF_START(t0, prepSolver);

    using namespace gpu_utils;

    uint32_t lane_id = threadIdx.x % 32;

    uint32_t world_id = invocation_idx;

    StateManager *state_mgr = getStateManager();

    CVSolveData *curr_sd = &solveDatas[world_id];

    if (curr_sd->freeAccDim == 0) {
        return;
    }

    if (lane_id == 0) {
        curr_sd->worldID = world_id;
    }

    BodyGroupProperties *all_properties = state_mgr->getWorldComponents<
        BodyGroupArchetype, BodyGroupProperties>(world_id);
    BodyGroupMemory *all_memories = state_mgr->getWorldComponents<
        BodyGroupArchetype, BodyGroupMemory>(world_id);
    CountT num_grps = state_mgr->numRows<BodyGroupArchetype>(world_id);

    { // Get the total mass in the world
        float total_mass = 0.f;

        total_mass = warpSumPred<float>(
            [&](uint32_t iter) -> float {
                return all_properties[iter].inertiaSum;
            }, num_grps);

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
                    BodyGroupMemory m = all_memories[i];
                    BodyGroupProperties p = all_properties[i];

                    mass_sparse.blks[i].dim = p.qvDim;
                    mass_sparse.blks[i].values = m.massMatrix(p);
                    mass_sparse.blks[i].ltdl = m.massLTDLMatrix(p);
                    mass_sparse.blks[i].expandedParent = m.expandedParent(p);
                });
        __syncwarp();

        if (lane_id == 0) {
            curr_sd->massSparse = mass_sparse;
        }
    }

    { // Prepare free acc
        float *free_acc = curr_sd->getFreeAcc(state_mgr);

        warpExclusivePrefixSumPred<uint32_t>(
                [&](uint32_t iter) {
                    return all_properties[iter].qvDim;
                },
                [&](uint32_t iter, uint32_t dof_offset) {
                    all_properties[iter].tmp.qvOffset = dof_offset;
                },
                curr_sd->numBodyGroups);

        for (uint32_t grp = 0; grp < curr_sd->numBodyGroups; ++grp) {
            BodyGroupMemory m = all_memories[grp];
            BodyGroupProperties p = all_properties[grp];

            __syncwarp();

            float *local_free_acc = m.biasVector(p);

            warpLoop(p.qvDim,
                [&](uint32_t i) {
                    free_acc[p.tmp.qvOffset + i] = local_free_acc[i];
                });
        }

        if (lane_id == 0) {
            curr_sd->freeAcc = free_acc;
        }
    }

    { // Prepare full velocity
        float *full_vel = curr_sd->getFullVel(state_mgr);

        for (uint32_t grp = 0; grp < curr_sd->numBodyGroups; ++grp) {
            BodyGroupMemory m = all_memories[grp];
            BodyGroupProperties p = all_properties[grp];
            BodyOffsets *grp_offsets = m.offsets(p);
            float *qv = m.qv(p);

            warpLoop(p.numBodies,
                [&](uint32_t iter) {
                    BodyOffsets o = grp_offsets[iter];

                    #pragma unroll
                    for (uint32_t i = 0; i < 6; ++i) {
                        if (i < o.numDofs)
                            full_vel[p.tmp.qvOffset + o.velOffset + i] = qv[o.velOffset + i];
                    }
                });

            __syncwarp();
        }

        if (lane_id == 0) {
            curr_sd->vel = full_vel;
        }
    }

    uint32_t num_contacts = state_mgr->numRows<Contact>(world_id);

#if 0
    if (lane_id == 0) {
        printf("num_contacts = %d\n", num_contacts);
    }
#endif

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
                        full_mu[3 * (offset + i) + 0] = tmp_state.mu;
                        full_mu[3 * (offset + i) + 1] = tmp_state.mu;
                        full_mu[3 * (offset + i) + 2] = tmp_state.mu;

                        full_penetrations[offset + i] = contact.points[i].w;
                    }
                }

                processed_pts += __shfl_sync(0xFFFF'FFFF, num_pts_ipf, 31);
            });

        curr_sd->mu = full_mu;
        curr_sd->penetrations = full_penetrations;
    }

    { // Prepare prefix sum of body group DOFs in current world
        warpExclusivePrefixSumPred<uint32_t>(
                [&](uint32_t iter) {
                    return all_properties[iter].numEq;
                },
                [&](uint32_t iter, uint32_t eq_offset) {
                    all_properties[iter].tmp.eqOffset = eq_offset;

                    // Might as well store the grpIndex
                    all_properties[iter].tmp.grpIndex = iter;
                },
                curr_sd->numBodyGroups);
    }

#if 1
    { // Prepare the contact jacobian
        float *j_c = curr_sd->getContactJacobian(state_mgr);
        curr_sd->J_c = j_c;

        float *diag_approx = curr_sd->getContactDiagApprox(state_mgr);

        warpSetZero(j_c, sizeof(float) *
                         3 * curr_sd->numContactPts *
                         curr_sd->totalNumDofs);

        warpSetZero(diag_approx, sizeof(float) *
                        3 * curr_sd->numContactPts);

        // TODO: Maybe experiment with different axes of parallelism
        // Right now, trying to parallelize over contacts
        struct ContactInfo {
            ContactConstraint contact;
            ContactTmpState tmpState;

            BodyGroupProperties pRef;
            BodyGroupMemory mRef;
            uint32_t refBodyIndex;

            BodyGroupProperties pAlt;
            BodyGroupMemory mAlt;
            uint32_t altBodyIndex;

            float invWeight;
            uint32_t numPoints;
        };

        auto get_contact_info = [&](uint32_t ct_idx) -> ContactInfo {
            auto contact = contacts[ct_idx];

            auto &ref_link = state_mgr->getUnsafe<LinkParentDofObject>(
                    contact.ref);
            auto &alt_link = state_mgr->getUnsafe<LinkParentDofObject>(
                    contact.alt);

            BodyGroupProperties p_ref = state_mgr->getUnsafe<BodyGroupProperties>(ref_link.bodyGroup);
            BodyGroupMemory m_ref = state_mgr->getUnsafe<BodyGroupMemory>(ref_link.bodyGroup);
            uint32_t body_idx_ref = ref_link.bodyIdx;

            BodyGroupProperties p_alt = state_mgr->getUnsafe<BodyGroupProperties>(alt_link.bodyGroup);
            BodyGroupMemory m_alt = state_mgr->getUnsafe<BodyGroupMemory>(alt_link.bodyGroup);
            uint32_t body_idx_alt = alt_link.bodyIdx;

            auto &ref_inertial = m_ref.inertials(p_ref)[body_idx_ref];
            auto &alt_inertial = m_alt.inertials(p_alt)[body_idx_alt];

            float inv_weight = (ref_inertial.approxInvMassTrans + 
                                alt_inertial.approxInvMassTrans);

            return {
                contact,
                contacts_tmp_state[ct_idx],

                p_ref,
                m_ref,
                body_idx_ref,

                p_alt,
                m_alt,
                body_idx_alt,

                inv_weight,
                (uint32_t)contacts[ct_idx].numPoints,
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

                        { // Compute jacobian for ref
                            computeContactJacobian(
                                    c_info.pRef,
                                    c_info.mRef,
                                    c_info.refBodyIndex,
                                    c_info.tmpState.C,
                                    contact_pt,
                                    j_c,
                                    c_info.pRef.tmp.qvOffset,
                                    //world_block_start[grp->tmpIdx0],
                                    curr_jacc_row,
                                    curr_sd->numRowsJc,
                                    -1.f,
                                    false);//(ct_idx == 0 && pt_idx == 0));
                        }

                        { // Compute jacobian for alt
                            computeContactJacobian(
                                    c_info.pAlt,
                                    c_info.mAlt,
                                    c_info.altBodyIndex,
                                    c_info.tmpState.C,
                                    contact_pt,
                                    j_c,
                                    c_info.pAlt.tmp.qvOffset,
                                    //world_block_start[grp->tmpIdx0],
                                    curr_jacc_row,
                                    curr_sd->numRowsJc,
                                    1.f,
                                    false);//(ct_idx == 0 && pt_idx == 0));
                        }

                        { // Compute the diagonal approximation
                            float inv_weight_trans = c_info.invWeight;
                            diag_approx[curr_jacc_row + 0] = inv_weight_trans;
                            diag_approx[curr_jacc_row + 1] = inv_weight_trans;
                            diag_approx[curr_jacc_row + 2] = inv_weight_trans;
                        }
                    }
                }
            });
    }

    { // Prepare equality jacobian
        float *j_e = curr_sd->getEqualityJacobian(state_mgr);
        curr_sd->J_e = j_e;

        float *diag_approx = curr_sd->getEqualityDiagApprox(state_mgr);
        float *residuals = curr_sd->getEqualityResiduals(state_mgr);

        curr_sd->diagApprox_e = diag_approx;
        curr_sd->eqResiduals = residuals;

        warpSetZero(j_e, 
                    sizeof(float) *
                    curr_sd->numRowsJe *
                    curr_sd->numColsJe);

        warpSetZero(diag_approx, sizeof(float) * curr_sd->numRowsJe);
        warpSetZero(residuals, sizeof(float) * curr_sd->numRowsJe);

        for (uint32_t grp = 0; grp < num_grps; ++grp) {
            __syncwarp();

            BodyGroupMemory m = all_memories[grp];
            BodyGroupProperties p = all_properties[grp];

            warpLoop(
                p.numBodies,
                [&](uint32_t iter) {
                    BodyOffsets o = m.offsets(p)[iter];

                    if (o.numEqs == 0) {
                        return;
                    }

                    BodyLimitConstraint limit = m.limits(p)[o.eqOffset];

                    if (limit.type == BodyLimitConstraint::Type::None) {
                        return;
                    }

                    uint32_t glob_row_offset = p.tmp.eqOffset +
                                               o.eqOffset;
                    uint32_t glob_col_offset = p.tmp.qvOffset +
                                               o.velOffset;

                    float *q = m.q(p) + o.posOffset;
                    BodyInertial &inertial = m.inertials(p)[iter];

                    switch (limit.type) {
                    case BodyLimitConstraint::Type::Hinge: {
                        if (limit.hinge.isActive(q[0])) {
                            float *to_change = j_e +
                                curr_sd->numRowsJe * glob_col_offset +
                                glob_row_offset;

                            to_change[0] =
                                limit.hinge.dConstraintViolation(q[0]);
                            residuals[glob_row_offset] = limit.hinge.constraintViolation(q[0]);
                            diag_approx[glob_row_offset] = inertial.approxInvMassDof[0];
                        } else {
                            float *to_change = j_e +
                                curr_sd->numRowsJe * glob_col_offset +
                                glob_row_offset;

                            to_change[0] = 0.f;
                            residuals[0] = 0.f;
                            diag_approx[glob_row_offset] = 0.f;
                        }
                    } break;

                    case BodyLimitConstraint::Type::Slider: {
                        if (limit.hinge.isActive(q[0])) {
                            float *to_change = j_e +
                                curr_sd->numRowsJe * glob_col_offset +
                                glob_row_offset;

                            to_change[0] =
                                limit.slider.dConstraintViolation(q[0]);
                            residuals[glob_row_offset] = limit.slider.constraintViolation(q[0]);
                            diag_approx[glob_row_offset] = inertial.approxInvMassDof[0];
                        } else {
                            float *to_change = j_e +
                                curr_sd->numRowsJe * glob_col_offset +
                                glob_row_offset;

                            to_change[0] = 0.f;
                            residuals[0] = 0.f;
                            diag_approx[glob_row_offset] = 0.f;
                        }
                    } break;

                    default: {
                        MADRONA_UNREACHABLE();
                    } break;
                    }
                });
        }
    }
#endif
}

void GaussMinimizationNode::computeContactJacobian(
        BodyGroupProperties &prop,
        BodyGroupMemory &mem,
        uint32_t body_idx,
        Mat3x3 &C,
        Vector3 &origin,
        float *J,
        uint32_t body_dof_offset,
        uint32_t jac_row,
        uint32_t j_num_rows,
        float coeff,
        bool dbg)
{
    (void)dbg;

    // Compute prefix sum to determine the start of the block for each body
    BodyOffsets *all_offsets = mem.offsets(prop);
    BodyPhi *all_phis = mem.bodyPhi(prop);

    // Populate J_C by traversing up the hierarchy
    uint8_t curr_idx = body_idx;
    while (curr_idx != 0xFF) {
        BodyOffsets offsets = all_offsets[curr_idx];

        // Populate columns of J_C
        float S[18] = {};
        computePhiTrans(offsets.dofType, all_phis[curr_idx], origin, S);

        // Only use translational part of S
        for(CountT i = 0; i < BodyOffsets::getDofTypeDim(offsets.dofType); ++i) {
            float *J_col = J +
                    j_num_rows * (body_dof_offset + (uint32_t)offsets.velOffset + i) +
                    jac_row;
            float *S_col = S + 3 * i;

            #pragma unroll
            for(CountT j = 0; j < 3; ++j) {
                J_col[j] = S_col[j];
            }
        }

        curr_idx = offsets.parent;
    }

    // Multiply by C^T to project into contact space
    for (CountT i = 0; i < prop.qvDim; ++i) {
        float *J_col = J +
                j_num_rows * (body_dof_offset + i) +
                jac_row;

        Vector3 J_col_vec = { J_col[0], J_col[1], J_col[2] };
        J_col_vec = C.transpose() * J_col_vec;
        J_col[0] = coeff * J_col_vec.x;
        J_col[1] = coeff * J_col_vec.y;
        J_col[2] = coeff * J_col_vec.z;
    }

#if 0
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
    }
#endif
}

template <typename Fn>
void GaussMinimizationNode::computeAccRef(
        float *acc_ref,
        uint32_t acc_ref_dim,
        float *j_mat,
        uint32_t num_rows_j,
        uint32_t num_cols_j,
        float *vel,
        uint32_t vel_dim,
        float h,
        Fn &&residual_fn,
        float *diag_approx,
        float *r_vec,
        bool dbg)
{
    using namespace gpu_utils;

    float time_const = 2.f * h;
    constexpr float damp_ratio = 1.f;
    constexpr float d_min = 0.9f,
                    d_max = 0.95f,
                    width = 0.001f,
                    mid = 0.5f,
                    power = 2.f;

    if (dbg) {
        printMatrix<true>(
                j_mat,
                num_rows_j,
                num_cols_j,
                "J_e");
    }

    // First store J @ v
    gmmaWarpSmallReg<float, 4, true, false, true>(
            acc_ref,
            j_mat,
            vel,
            num_rows_j,
            num_cols_j,
            vel_dim,
            1);

    warpLoop(num_rows_j, [&](uint32_t iter) {
        float r = residual_fn(iter);

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

        r_vec[iter] = ((1.f - imp) / imp) * diag_approx[iter];
    });
}

// Might be overkill to allocate a warp per world but we can obviously
// experiment.
void GaussMinimizationNode::computeContactAccRef(int32_t invocation_idx)
{
    CV_PROF_START(t0, contAccRef);

    using namespace gpu_utils;

    uint32_t total_num_worlds = mwGPU::GPUImplConsts::get().numWorlds;

    // This is the global warp ID
    // uint32_t warp_id = (blockDim.x * blockIdx.x + threadIdx.x) / 32;
    uint32_t warp_id = threadIdx.x / 32;

    assert(blockDim.x == consts::numMegakernelThreads);

    StateManager *state_mgr = getStateManager();

    uint32_t world_id = invocation_idx;

    { // Do the actual computation
        CVSolveData *curr_sd = &solveDatas[world_id];

        if (curr_sd->freeAccDim == 0) {
            return;
        }

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
                    curr_sd->getContactAccRef(state_mgr),
                    false
                };
            }
        } ();

        if (curr_sd->numRowsJc > 0) {
            computeAccRef(
                    acc_ref,
                    curr_sd->numRowsJc,
                    curr_sd->J_c,
                    curr_sd->numRowsJc,
                    curr_sd->numColsJc,
                    curr_sd->vel,
                    curr_sd->velDim,
                    curr_sd->h,
                    [&](uint32_t iter) {
                        return (iter % 3 == 0) ?
                            -curr_sd->penetrations[iter / 3] : 0.f;
                    },
                    curr_sd->getContactDiagApprox(state_mgr),
                    curr_sd->getContactR(state_mgr));

            float *r_vec = curr_sd->getContactR(state_mgr);
            float *mus = curr_sd->getMu(state_mgr);

            static constexpr float kImpRatio = 1.f;
            static constexpr uint32_t kConeDim = 3;

            warpLoop(
                curr_sd->numRowsJc / 3,
                [&](uint32_t iter) {
                    uint32_t full_iter = iter * 3;

                    r_vec[full_iter + 1] = r_vec[full_iter] / kImpRatio;

                    mus[full_iter] = mus[full_iter + 1] * sqrtf(
                            r_vec[full_iter + 1] / r_vec[full_iter]);

                    for (uint32_t i = 2; i < kConeDim; ++i) {
                        r_vec[full_iter + i] =
                            r_vec[full_iter + 1] * mus[full_iter + 1] * mus[full_iter + 1] / 
                            (mus[full_iter + i] * mus[full_iter + i]);
                    }
                });
            __syncwarp();

            // The R becomes D now
            warpLoop(
                curr_sd->numRowsJc,
                [&](uint32_t iter) {
                    r_vec[iter] = 1.f / r_vec[iter];
                });

            if (acc_ref && in_smem) {
                float * acc_ref_glob =
#if 0
                    (float *)state_mgr->memoryRangePointer<SolverScratch256b>(
                            curr_sd->accRefMemory);
#endif
                    curr_sd->getContactAccRef(state_mgr);

                warpCopy(acc_ref_glob, acc_ref,
                         sizeof(float) * curr_sd->numRowsJc);
                acc_ref = (float *)acc_ref_glob;
            }
        }

        __syncwarp();
    }
}

void GaussMinimizationNode::computeEqualityAccRef(int32_t invocation_idx)
{
    CV_PROF_START(t0, eqAccRef);

    using namespace gpu_utils;

    uint32_t total_num_worlds = mwGPU::GPUImplConsts::get().numWorlds;

    // This is the global warp ID
    // uint32_t warp_id = (blockDim.x * blockIdx.x + threadIdx.x) / 32;
    uint32_t warp_id = threadIdx.x / 32;

    assert(blockDim.x == consts::numMegakernelThreads);

    StateManager *state_mgr = getStateManager();

    uint32_t world_id = invocation_idx;

    { // Do the actual computation
        CVSolveData *curr_sd = &solveDatas[world_id];

        if (curr_sd->freeAccDim == 0) {
            return;
        }

        float *residuals = curr_sd->eqResiduals;

        // We want acc_ref to have priority in shared memory
        auto [acc_ref, in_smem] = [&]() -> std::pair<float *, bool> {
            const int32_t num_smem_bytes_per_warp =
                mwGPU::SharedMemStorage::numBytesPerWarp();

            uint32_t acc_ref_bytes = sizeof(float) * curr_sd->numRowsJe;
            auto smem_buf = (uint8_t *)mwGPU::SharedMemStorage::buffer +
                            num_smem_bytes_per_warp * warp_id;

            if (acc_ref_bytes == 0) {
                return { (float *)nullptr, false };
            } else if (acc_ref_bytes < num_smem_bytes_per_warp) {
                return { (float *)smem_buf, true };
            } else {
                return {
                    (float *)curr_sd->getEqualityAccRef(state_mgr),
                    false
                };
            }
        } ();

        if (curr_sd->numRowsJe > 0) {
            computeAccRef(
                    acc_ref,
                    curr_sd->numRowsJe,
                    curr_sd->J_e,
                    curr_sd->numRowsJe,
                    curr_sd->numColsJe,
                    curr_sd->vel,
                    curr_sd->velDim,
                    curr_sd->h,
                    [&](uint32_t iter) {
                        return residuals[iter];
                    },
                    curr_sd->getEqualityDiagApprox(state_mgr),
                    curr_sd->getEqualityR(state_mgr));

            float *r_vec = curr_sd->getEqualityR(state_mgr);

            // The R becomes D now
            warpLoop(
                curr_sd->numRowsJe,
                [&](uint32_t iter) {
                    if (r_vec[iter] == 0.f) {
                        r_vec[iter] = 0.f;
                    } else {
                        r_vec[iter] = 1.f / r_vec[iter];
                    }
                });

            if (acc_ref && in_smem) {
                float * acc_ref_glob =
                    (float *)curr_sd->getEqualityAccRef(state_mgr);

                warpCopy(acc_ref_glob, acc_ref,
                         sizeof(float) * curr_sd->numRowsJe);
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
    CV_PROF_START(t0, cg);

#define iter_warp_printf(...) if (iter < 4) { warp_printf(__VA_ARGS__); }
#define iter_matrix_printf(...) if (iter < 4) { printMatrix(__VA_ARGS__); }

    using namespace gpu_utils;

    constexpr float kTolerance = 1e-8f;
    constexpr float lsTolerance = 0.01f;
    // constexpr float MINVAL = 1e-12f;

    uint32_t total_resident_warps = (blockDim.x * gridDim.x) / 32;

    uint32_t total_num_worlds = mwGPU::GPUImplConsts::get().numWorlds;

    // Global warp ID
    assert(blockDim.x == consts::numMegakernelThreads);

    StateManager *state_mgr = getStateManager();

    uint32_t world_id = invocation_idx;

    { // Do the computation
        CVSolveData *curr_sd = &solveDatas[world_id];

        if (curr_sd->freeAccDim == 0) {
            return;
        }

        float tol_scale = 1.f / curr_sd->totalMass;

        prepareRegInfos(curr_sd);

        float *x = (float *)curr_sd->regInfos[0].ptr;
        float *m_grad = (float *)curr_sd->regInfos[1].ptr;
        float *p = (float *)curr_sd->regInfos[2].ptr;
        float *scratch1 = (float *)curr_sd->regInfos[3].ptr;
        float *scratch2 = (float *)curr_sd->regInfos[4].ptr;
        float *scratch3 = (float *)curr_sd->regInfos[5].ptr;
        float *jaccref_cont = (float *)curr_sd->regInfos[6].ptr;
        float *mxmin = (float *)curr_sd->regInfos[7].ptr;
        float *jaccref_eq = (float *)curr_sd->regInfos[8].ptr;

        float *acc_ref_cont = [&]() -> float * {
            if (curr_sd->numRowsJc == 0) {
                return (float *)nullptr;
            } else {
                return curr_sd->getContactAccRef(state_mgr);
            }
        } ();

        float *acc_ref_eq = [&]() -> float * {
            if (curr_sd->numRowsJe == 0) {
                return (float *)nullptr;
            } else {
                return curr_sd->getEqualityAccRef(state_mgr);
            }
        } ();

        // We are using freeAcc as initial guess
        warpCopy(x, curr_sd->freeAcc, sizeof(float) * curr_sd->freeAccDim);
        __syncwarp();


        // Print all matrices
#if 0
        printMatrix<true>(curr_sd->J_c, curr_sd->numRowsJc, curr_sd->numColsJc, "J_c");
        printMatrix<true>(curr_sd->J_e, curr_sd->numRowsJe, curr_sd->numColsJe, "J_e");
        printMatrix(curr_sd->massSparse.blks[0].values,
                curr_sd->massSparse.blks[0].dim,
                curr_sd->massSparse.blks[0].dim,
                "M");
#endif

        __syncwarp();

        dobjWarp<true>(m_grad, x, curr_sd, scratch1,
                       jaccref_cont, jaccref_eq, 
                       mxmin, 
                       acc_ref_cont, acc_ref_eq,
                       false);
        __syncwarp();

        float curr_fun = objWarp(x, curr_sd, jaccref_cont, jaccref_eq, mxmin);
        __syncwarp();

        // Keep track of the norm2 of g (m_grad currently has g)
        float g_norm = sqrtf(norm2Warp(m_grad, curr_sd->freeAccDim));

        float g_dot_m_grad = sparseBlkDiagSolve<float, true>(
                m_grad, &curr_sd->massSparse, scratch1);

        // By now, m_grad actually has m_grad
        __syncwarp();

        warpLoop(curr_sd->freeAccDim, [&](uint32_t iter) {
            p[iter] = -m_grad[iter];
        });
        __syncwarp();

        uint32_t max_iters = 100;
        uint32_t iter = 0;

        for (; iter < max_iters; ++iter) {
            if (tol_scale * g_norm < kTolerance)
                break;

            float p_norm = sqrtf(norm2Warp(p, curr_sd->freeAccDim));

            if (p_norm < MINVAL)
                break;

           float lsTol = lsTolerance;
           float alpha = exactLineSearch(
                curr_sd, jaccref_cont, jaccref_eq, 
                mxmin, p, x, lsTol, scratch1, 
                curr_sd->getContactR(nullptr),
                curr_sd->getEqualityR(nullptr),
                false);
            __syncwarp();

            // No improvement
            if (alpha == 0.f)
                break;

            // Update x to the new value after alpha was found
            warpLoop(curr_sd->freeAccDim,
                [&](uint32_t iter) {
                    x[iter] += alpha * p[iter];
                });

            __syncwarp();

            float *g_new = m_grad;
            float new_fun = 0.f;
            { // Get the new gradient
                warpCopy(scratch2, m_grad, curr_sd->freeAccDim * sizeof(float));

                dobjWarp<true>(g_new, x, curr_sd, scratch1,
                               jaccref_cont, jaccref_eq, mxmin, 
                               acc_ref_cont, acc_ref_eq, false);

                __syncwarp();

                new_fun = objWarp(x, curr_sd, jaccref_cont, jaccref_eq, mxmin);
                __syncwarp();

                g_norm = sqrtf(norm2Warp(g_new, curr_sd->freeAccDim));
            }

            if (tol_scale * (curr_fun - new_fun) < kTolerance) {
                break;
            }

            {
                // Now we have scratch1 and scratch2 to play with
                // We need have access to these three at the same time:
                // g_new, Mgrad_new M_grad
                warpCopy(scratch3, g_new, curr_sd->freeAccDim * sizeof(float));

                float g_dot_m_grad_new = sparseBlkDiagSolve<float, true>(
                        m_grad, &curr_sd->massSparse, scratch1);

                // By now, m_grad actually has m_grad_new,
                //         scratch2 has m_grad
                //         scratch3 has g_new
                __syncwarp();

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

                warpLoop(
                    curr_sd->freeAccDim,
                    [&](uint32_t iter) {
                        p[iter] = -m_grad[iter] + beta * p[iter];
                    });
            }

            curr_fun = new_fun;
        }

        { // Now, we need to copy x into the right components
            BodyGroupProperties *all_properties = state_mgr->getWorldComponents<
                BodyGroupArchetype, BodyGroupProperties>(world_id);
            BodyGroupMemory *all_memories = state_mgr->getWorldComponents<
                BodyGroupArchetype, BodyGroupMemory>(world_id);
            CountT num_grps = state_mgr->numRows<BodyGroupArchetype>(world_id);

            for (uint32_t grp = 0; grp < num_grps; ++grp) {
                BodyGroupProperties p = all_properties[grp];
                BodyGroupMemory m = all_memories[grp];

                BodyOffsets *grp_offsets = m.offsets(p);
                float *dqv = m.dqv(p);

                warpLoop(
                    p.numBodies,
                    [&](uint32_t iter) {
                        BodyOffsets o = grp_offsets[iter];

                        for (uint32_t i = 0; i < o.numDofs; ++i) {
                            dqv[o.velOffset + i] = x[p.tmp.qvOffset + o.velOffset + i];
                        }
                    });
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
        &GaussMinimizationNode::computeContactAccRef>(data_id, {cur_node},
                Optional<TaskGraph::NodeID>::none(),
                num_invocations,
                32);

    cur_node = builder.addNodeFn<
        &GaussMinimizationNode::computeEqualityAccRef>(data_id, {cur_node},
                Optional<TaskGraph::NodeID>::none(),
                num_invocations,
                32);

    cur_node = builder.addNodeFn<
        &GaussMinimizationNode::nonlinearCG>(data_id, {cur_node},
                Optional<TaskGraph::NodeID>::none(),
                num_invocations,
                32);

    return cur_node;
}

#endif

#ifndef MADRONA_GPU_MODE
void copyResult(Context &ctx,
             const float *res)
{
    uint32_t world_id = ctx.worldID().idx;
    StateManager *state_mgr = getStateManager(ctx);
    BodyGroupMemory *all_memories = state_mgr->getWorldComponents<
        BodyGroupArchetype, BodyGroupMemory>(world_id);
    BodyGroupProperties *all_properties = state_mgr->getWorldComponents<
        BodyGroupArchetype, BodyGroupProperties>(world_id);

    CountT num_grps = state_mgr->numRows<BodyGroupArchetype>(world_id);

    // Update the body accelerations
    uint32_t processed_dofs = 0;
    for (CountT i = 0; i < num_grps; ++i) {
        BodyGroupMemory &m = all_memories[i];
        BodyGroupProperties &p = all_properties[i];

        for (CountT j = 0; j < all_properties[i].numBodies; j++) {
            BodyOffsets offsets = m.offsets(p)[j];
            float *dqv = m.dqv(p) + offsets.velOffset;

            for (CountT k = 0; k < BodyOffsets::getDofTypeDim(offsets.dofType); k++) {
                dqv[k] = res[processed_dofs];
                processed_dofs++;
            }
        }
    }
}

void solveCPU(Context &ctx,
              CVSolveData &cv_sing)
{

    using namespace cpu_solver;
    uint32_t dim_c = cv_sing.numRowsJc;
    uint32_t dim_e = cv_sing.numRowsJe;
    uint32_t num_constraints = dim_c + dim_e;

    // No constraints, done
    if (num_constraints == 0) { copyResult(ctx, cv_sing.freeAcc); }

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
            copyResult(ctx, res);
        }
        return;
    }

    if (!ctx.singleton<CVSolveData>().enablePhysics) {
        return;
    }

    // Memory for reference acceleration, impedance, and residuals
    //  todo: make this contiguous
    float *acc_ref_c = (float *)ctx.tmpAlloc(sizeof(float) * dim_c);
    float *acc_ref_e = (float *)ctx.tmpAlloc(sizeof(float) * dim_e);
    float *R_c = (float *)ctx.tmpAlloc(sizeof(float) * dim_c);
    float *R_e = (float *)ctx.tmpAlloc(sizeof(float) * dim_e);
    float *D_c = (float *)ctx.tmpAlloc(sizeof(float) * dim_c);
    float *D_e = (float *)ctx.tmpAlloc(sizeof(float) * dim_e);
    float *r_con = (float *)ctx.tmpAlloc(sizeof(float) * dim_c);
    float *a_solve = (float *)ctx.tmpAlloc(sizeof(float) * cv_sing.totalNumDofs);
    float *x0 = (float *)ctx.tmpAlloc(sizeof(float) * cv_sing.totalNumDofs);

    // Set residuals for contact constraints
    memset(r_con, 0, sizeof(float) * dim_c);
    for (uint32_t i = 0; i < cv_sing.numContactPts; i++) {
        r_con[i * 3] = -cv_sing.penetrations[i];
    }

    // Compute reference accelerations
    computeAccRef(acc_ref_c, R_c, cv_sing.vel, cv_sing.J_c,
                  cv_sing.numRowsJc, cv_sing.numColsJc, r_con,
                  cv_sing.diagApprox_c, cv_sing.h);
    computeAccRef(acc_ref_e, R_e, cv_sing.vel, cv_sing.J_e,
                  cv_sing.numRowsJe, cv_sing.numColsJe, cv_sing.eqResiduals,
                  cv_sing.diagApprox_e, cv_sing.h);

    adjustContactRegularization(R_c, cv_sing.mu, cv_sing.muDim);

    // Constraint mass
    for (uint32_t i = 0; i < dim_c; i++) {
        D_c[i] = 1 / R_c[i];
    }
    for (uint32_t i = 0; i < dim_e; i++) {
        D_e[i] = 1 / R_e[i];
        // Todo: filter out unused constraints
        if (R_e[i] < MINVAL) { D_e[i] = 0.f; }
    }

    constexpr float tol = 1e-8f;
    constexpr float ls_tol = 0.01f;
    uint32_t max_iter = 100;
    uint32_t ls_iters = 50;

    // Set initial guess to be previous acceleration
    memcpy(x0, cv_sing.currAcc, sizeof(float) * cv_sing.totalNumDofs);
    nonlinearCG(ctx, a_solve, x0, acc_ref_c, acc_ref_e, D_c, D_e,
        tol, ls_tol, max_iter, ls_iters, cv_sing);
    copyResult(ctx, a_solve);
}
#endif
}

TaskGraphNodeID setupSolveTasks(TaskGraphBuilder &builder,
                                TaskGraphNodeID prev)
{
#ifdef MADRONA_GPU_MODE
    auto cur_node = builder.addToGraph<
        tasks::GaussMinimizationNode>({prev});
#else
    auto cur_node = builder.addToGraph<ParallelForNode<Context,
         tasks::solveCPU,
            CVSolveData
        >>({prev});
#endif

    return cur_node;
}

}
