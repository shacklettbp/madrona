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
    // Make sure scratch has shape max(sd->freeAccDim, sd->nc)
    //
    // This is also going to return a couple other things to reduce computation
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
        float *jaccref_c,
        float *jaccref_l,
        float *Mxmin);

    float exactLineSearch(
        CVSolveData* sd,
        float* jaccref_cont,
        float* jaccref_eq,
        float* Mxmin,
        float* p,
        float* x,
        float ls_tol,
        float* scratch,
        float* d_c,
        float* d_e);

    void computeContactJacobian(BodyGroupProperties &prop,
                                BodyGroupMemory &mem,
                                uint32_t body_idx,
                                Vector3 &origin,
                                float *J,
                                uint32_t body_dof_offset,
                                uint32_t jac_row,
                                uint32_t j_num_rows,
                                float coeff,
                                bool dbg);

    template <typename Fn>
    void computeAccRef(
        float* acc_ref_cont,
        float* j_mat,
        uint32_t num_rows_j,
        uint32_t num_cols_j,
        float* vel,
        uint32_t vel_dim,
        float h,
        Fn&& residual_fn,
        float* diag_approx,
        float* R_vec);

    void calculateSolverDims(uint32_t world_id, CVSolveData *sd);
    void prepareRegInfos(CVSolveData *sd);



    // Nodes in the taskgraph:
    void allocateScratch(int32_t invocation_idx);
    // Prepares mass matrix and contact jacobian
    void prepareSolver(int32_t invocation_idx);
    void computeContactAccRef(int32_t invocation_idx);
    void computeLimitAccRef(int32_t invocation_idx);
    void nonlinearCG(int32_t invocation_idx);

    // Testing of helper functions
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
    float *jaccref_c,
    float *jaccref_l,
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
            sd->nv);

    // We don't actually need to pass in the state manager
    float *dc = sd->getContactR(nullptr);
    float *de = sd->getEqualityR(nullptr);

    // Calculate s(...) now:
    warpLoopSync(sd->nc / 3, [&](uint32_t iter) {
        float curr_val = 0.f;
        if (iter != 0xFFFF'FFFF) {
            auto [n, t1, t2, t, mu, mw] = mapToConeSpace(
                    jaccref_c, sd->mu, iter);

            float dn = dc[3 * iter + 0];
            float d1 = dc[3 * iter + 1];
            float d2 = dc[3 * iter + 2];

            if (n >= mu * t || (t <= 0.f && 0.f <= n)) {
                // Do nothing (top zone)
            } else if (mu * n + t <= 0.f || (t <= 0.f && n < 0.f)) {
                // Bottom zone
                curr_val =
                    0.5f * (dn * jaccref_c[iter * 3 + 0] * jaccref_c[iter * 3 + 0] +
                            d1 * jaccref_c[iter * 3 + 1] * jaccref_c[iter * 3 + 1] +
                            d2 * jaccref_c[iter * 3 + 2] * jaccref_c[iter * 3 + 2]);
            } else {
                // Middle zone
                curr_val = 0.5f * dn * mw * (n - mu * t) * (n - mu * t);
            }
        }

        res += warpReduceSum(curr_val);
    });

    warpLoopSync(sd->nl, [&](uint32_t iter) {
        float curr_val = 0.f;
        if (iter != 0xFFFF'FFFF) {
            // Constraint is satisfied
            if (jaccref_l[iter] >= 0.f) {
                curr_val = 0.f;
            } else {
                curr_val = 0.5f * de[iter] * jaccref_l[iter] * jaccref_l[iter];
            }
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

    // x - acc_free
    warpLoop(sd->nv, [&](uint32_t iter) {
        scratch[iter] = x[iter] - sd->freeAcc[iter];
    });

    __syncwarp();

    sparseBlkDiagSmallReg<float, 4, false, false, true>(
            res,
            &sd->massSparse,
            scratch,
            sd->nv, 1);
    __syncwarp();

    if (dbg) {
        printMatrix(res, 1, sd->nv, "mxmin new");
    }

    // By now, res has M @ x_min_acc_free

    if constexpr (calc_package) {
        warpCopy(Mxmin, res, sd->nv * sizeof(float));
        __syncwarp();
    }

    gmmaWarpSmallReg<float, 4, true, false, true>(
            scratch,
            sd->J_c,
            x,
            sd->nc,
            sd->nv,
            sd->nv,
            1);
    __syncwarp();

    warpLoop(sd->nc, [&](uint32_t iter) {
        scratch[iter] -= acc_ref_cont[iter];
    });
    __syncwarp();

    if constexpr (calc_package) {
        warpCopy(jaccref_cont, scratch, sd->nc * sizeof(float));
        __syncwarp();
    }

    float *dc = sd->getContactR(nullptr);
    float *de = sd->getEqualityR(nullptr);

    // ds
    warpLoop(sd->nc / 3, [&](uint32_t iter) {
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
            sd->nv,
            sd->nc,
            sd->nc,
            1);
    __syncwarp();



    // Get equality part of the gradient
    gmmaWarpSmallReg<float, 4, true, false, true>(
            scratch,
            sd->J_l,
            x,
            sd->nl,
            sd->nv,
            sd->nv,
            1);
    __syncwarp();

    warpLoop(sd->nl, [&](uint32_t iter) {
        scratch[iter] -= acc_ref_eq[iter];
    });
    __syncwarp();

    if constexpr (calc_package) {
        warpCopy(jaccref_eq, scratch, sd->nl * sizeof(float));
        __syncwarp();
    }

    warpLoop(sd->nl, [&](uint32_t iter) {
        // Constraint is satisfied
        if (scratch[iter] >= 0.f) {
            scratch[iter] = 0.f;
        } else {
            scratch[iter] = de[iter] * scratch[iter];
        }
    });

    gmmaWarpSmallReg<float, 4, false, false, false>(
            res,
            sd->J_l,
            scratch,
            sd->nv,
            sd->nl,
            sd->nl,
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
    CVSolveData* sd,
    float* jaccref_cont,
    float* jaccref_eq,
    float* Mxmin,
    float* p,
    float* x,
    float ls_tol,
    float* scratch,
    float* d_c,
    float* d_e)
{
    CV_PROF_START(t0, lineSearch);

    using namespace gpu_utils;
    uint32_t iter = 0;
    uint32_t ls_iters = 50;

    // function, first deriv and second deriv evals
    struct Evals {
        float alpha; // useful to save
        float fun;
        float grad;
        float hess;
    };

    // Precompute solve values
    // 1. Gauss objective {p@M(x-a_free), (x-a_free)@M(x-a_free), p@M@p}
    float pMx_free = dotVectors(p, Mxmin, sd->nv);
    float xmin_M_xmin = dotVectorsPred<float>(
        [&](uint32_t iter) {
            return x[iter] - sd->freeAcc[iter];
        },
        [&](uint32_t iter) {
            return Mxmin[iter];
        },
        sd->nv);
    __syncwarp();
    float pMp = 0.f;
    { // Ok, now we don't need to store Mxmin anymore
        float *Mp = Mxmin;
        sparseBlkDiagSmallReg<float, 4, false, false, true>(
                Mp, &sd->massSparse, p,
                sd->nv, 1);
        __syncwarp();

        pMp = dotVectors(p, Mp, sd->nv);
    }
    __syncwarp();

    // Store J_c @ p in the scratch space we used for Mp
    float *Jp_c = Mxmin;
    { // Calculate J_c @ p
        gmmaWarpSmallReg<float, 4, true, false, true>(
            Jp_c, sd->J_c, p,
            sd->nc, sd->nv,
            sd->nv, 1);
    }
    __syncwarp();

    float *Jp_l = scratch;
    { // Calculate J_l @ p
        gmmaWarpSmallReg<float, 4, true, false, true>(
            Jp_l, sd->J_l, p,
            sd->nl, sd->nv,
            sd->nv, 1);
    }
    __syncwarp();


    float *mus = sd->mu;

    // Line search function: updates evals with phi(a), d_phi(a), d2_phi(a)
    auto phi = [&](Evals *evals) {
        float a = evals->alpha;
        // Process Gauss first
        float fun = 0.5f * a * a * pMp + a * pMx_free + 0.5f * xmin_M_xmin;
        float grad = a * pMp + pMx_free;
        float hess = pMp;

        struct Diff {
            float dfun;
            float dgrad;
            float dhess;
        };

        // Process contacts
        warpLoopSync(sd->nc / 3, [&](uint32_t iter) {
            auto d = [&]() -> Diff {
                if (iter == 0xFFFF'FFFF) {
                    return {
                        0.f, 0.f, 0.f
                    };
                } else {
                    // Components of J @ x - a_ref, J @ p
                    float Jx_n = jaccref_cont[iter * 3];
                    float Jx_t1 = jaccref_cont[iter * 3 + 1];
                    float Jx_t2 = jaccref_cont[iter * 3 + 2];
                    float Jp_n = Jp_c[iter * 3];
                    float Jp_t1 = Jp_c[iter * 3 + 1];
                    float Jp_t2 = Jp_c[iter * 3 + 2];
                    // Friction
                    float mu = mus[iter * 3];
                    float mu1 = mus[iter * 3 + 1];
                    float mu2 = mus[iter * 3 + 2];
                    // Weights
                    float Dn = d_c[iter * 3];
                    float D1 = d_c[iter * 3 + 1];
                    float D2 = d_c[iter * 3 + 2];
                    float Dm = Dn / (mu * mu * (1.f + mu * mu));

                    // Quadratic (bottom zone)
                    float quad0 = 0.5f * (Dn * Jx_n * Jx_n +
                                          D1 * Jx_t1 * Jx_t1 +
                                          D2 * Jx_t2 * Jx_t2);
                    float quad1 = (Dn * Jx_n * Jp_n +
                                   D1 * Jx_t1 * Jp_t1 +
                                   D2 * Jx_t2 * Jp_t2);
                    float quad2 = 0.5f * (Dn * Jp_n * Jp_n +
                                          D1 * Jp_t1 * Jp_t1 +
                                          D2 * Jp_t2 * Jp_t2);

                    // Map to dual cone space
                    Jx_n = Jx_n * mu;
                    Jx_t1 = Jx_t1 * mu1;
                    Jx_t2 = Jx_t2 * mu2;
                    Jp_n = Jp_n * mu;
                    Jp_t1 = Jp_t1 * mu1;
                    Jp_t2 = Jp_t2 * mu2;

                    // Temporary
                    float U0 = Jx_n;
                    float V0 = Jp_n;
                    float UU = Jx_t1 * Jx_t1 + Jx_t2 * Jx_t2;
                    float UV = Jx_t1 * Jp_t1 + Jx_t2 * Jp_t2;
                    float VV = Jp_t1 * Jp_t1 + Jp_t2 * Jp_t2;

                    float N = U0 + a * V0;
                    float T_sqr = UU + a * (2 * UV + a * VV);
                    // No tangent force
                    if (T_sqr <= 0) {
                        // Bottom zone
                        if (N < 0) {
                            return {
                                // Fun
                                quad0 + a * quad1 + a * a * quad2,
                                // Grad
                                quad1 + 2.f * a * quad2,
                                // Hess
                                2.f * quad2
                            };
                        }
                        // Top zone
                        return { 0.f, 0.f, 0.f };
                    }

                    float T = sqrtf(T_sqr);
                    if (N >= mu * T) {
                        // Don't add anything up
                        return {0.f, 0.f, 0.f};
                    }
                    // Bottom zone
                    else if (mu * N + T <= 0.f) {
                        return {
                            // Fun
                            quad0 + a * quad1 + a * a * quad2,
                            // Grad
                            quad1 + 2.f * a * quad2,
                            // Hess
                            2.f * quad2
                        };
                    } else {
                        float N1 = V0;
                        float T1 = (UV + a * VV) / T;
                        float T2 = VV / T - (UV + a * VV) * T1 / (T * T);
                        return {
                            0.5f*Dm*(N-mu*T)*(N-mu*T),
                            Dm*(N-mu*T)*(N1-mu*T1),
                            Dm*((N1-mu*T1)*(N1-mu*T1) + (N-mu*T)*(-mu*T2)),
                        };
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

        warpLoopSync(sd->nl, [&](uint32_t iter) {
            auto d = [&]() -> Diff {
                if (iter == 0xFFFF'FFFF) {
                    return { 0.f, 0.f, 0.f };
                }

                float jx = jaccref_eq[iter];
                float jp = Jp_l[iter];
                float N_search = jx + a * jp;
                // Active limit
                if (N_search < 0.f)
                {
                    float de = d_e[iter];
                    float quad0 = 0.5f * de * jx * jx;
                    float quad1 = de * jx * jp;
                    float quad2 = 0.5f * de * jp * jp;
                    return {
                        quad0 + a * quad1 + a * a * quad2,
                        quad1 + 2.f * a * quad2,
                        2.f * quad2
                    };
                }
                return {0.f, 0.f, 0.f};
            } ();

            float dfun = warpReduceSum(d.dfun);
            float dgrad = warpReduceSum(d.dgrad);
            float dhess = warpReduceSum(d.dhess);

            fun += dfun;
            grad += dgrad;
            hess += dhess;
        });

        evals->fun = fun;
        evals->grad = grad;
        evals->hess = hess;
        iter++;
    };

    auto updateBracket = [&](Evals *p, Evals candidates[3], Evals *pnext) {
        int flag = 0;
        for (int i = 0; i < 3; i++) {
            if (p->grad < 0 && candidates[i].grad < 0
                && p->grad < candidates[i].grad) {
                *p = candidates[i];
                flag = 1;
            }
            else if (p->grad > 0 && candidates[i].grad > 0 &&
                     p->grad > candidates[i].grad) {
                *p = candidates[i];
                flag = 2;
             }
        }

        if (flag) {
            pnext->alpha = p->alpha - p->grad / p->hess;
            phi(pnext);
        }
        return flag;
    };

    Evals p0, p1, p2, pmid, p1next, p2next;
    p0.alpha = 0.f;
    phi(&p0);

    p1.alpha = p0.alpha - p0.grad / p0.hess;
    phi(&p1);
    if (p0.fun < p1.fun) {
        p1 = p0;
    }

    // Initial convergence
    if (fabsf(p1.grad) < ls_tol) {
        return p1.alpha;
    }

    // Opposing direction of gradient at alpha1
    float dir = p1.grad < 0.f ? 1.f : -1.f;

    // One-sided search
    while (p1.grad * dir <= -ls_tol && iter < ls_iters) {
        p2 = p1;

        // Newton step
        p1.alpha -= p1.grad / p1.hess;
        phi(&p1);

        // Check for convergence
        if (fabs(p1.grad) < ls_tol) {
            return p1.alpha;
        }
    }

    // Failed to bracket
    if (iter >= ls_iters) {
        return p1.alpha;
    }

    // Bracketing to find where d_phi equals zero
    p2next = p1;
    p1next.alpha = p1.alpha - p1.grad / p1.hess;
    phi(&p1next);

    // Bracketed search
    while (iter < ls_iters) {
        // midpoint evaluation
        pmid.alpha = 0.5f * (p1.alpha + p2.alpha);
        phi(&pmid);

        Evals candidates[3] = {p1next, p2next, pmid};
        // check candidates for convergence
        float best_cost = 0;
        int best_ind = -1;
        for (int i = 0; i < 3; i++) {
            if (fabsf(candidates[i].grad) < ls_tol &&
                (best_ind == -1 || candidates[i].fun < best_cost)) {
                best_cost = candidates[i].fun;
                best_ind = i;
                }
        }
        if (best_ind >= 0) {
            return candidates[best_ind].alpha;
        }

        // Update brackets
        int b1 = updateBracket(&p1, candidates, &p1next);
        int b2 = updateBracket(&p2, candidates, &p2next);
        // use midpoint if can't update
        if (!b1 && !b2) {
            return pmid.alpha;
        }
    }


    // choose bracket with best cost
    if (p1.fun <= p2.fun && p1.fun < p0.fun) {
        return p1.alpha;
    }
    if (p2.fun <= p1.fun && p2.fun < p0.fun) {
        return p2.alpha;
    }
    return 0;

    MADRONA_UNREACHABLE();
}

void GaussMinimizationNode::calculateSolverDims(
        uint32_t world_id,
        CVSolveData *sd)
{
    using namespace gpu_utils;

    StateManager *state_mgr = getStateManager();
    BodyGroupProperties *all_properties = state_mgr->getWorldComponents<
        BodyGroupArchetype, BodyGroupProperties>(world_id);
    CountT num_grps = state_mgr->numRows<BodyGroupArchetype>(world_id);

    // Number of body groups and physics step
    MADRONA_GPU_SINGLE_THREAD {
        sd->numBodyGroups = num_grps;
        sd->h = state_mgr->getSingleton<PhysicsSystemState>({(int32_t)world_id}).h;
    }

    __syncwarp();

    { // Get total num dofs
        uint32_t total_num_dofs = warpSumPred<uint32_t>(
            [&](uint32_t iter) {
                return all_properties[iter].qvDim;
            }, sd->numBodyGroups);

        MADRONA_GPU_SINGLE_THREAD {
            sd->nv = total_num_dofs;
            sd->nv = total_num_dofs;
            sd->nv = total_num_dofs;
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

        MADRONA_GPU_SINGLE_THREAD {
            sd->nc = 3 * total_num_contact_pts;
            sd->nv = sd->nv;
            sd->muDim = total_num_contact_pts;
            sd->penetrationsDim = total_num_contact_pts;
        }
    }

    __syncwarp();

    { // Get limit jacobian dims
        uint32_t total_num_rows = warpSumPred<uint32_t>(
            [&](uint32_t iter) {
                return all_properties[iter].numLimits;
            }, num_grps);

        MADRONA_GPU_SINGLE_THREAD {
            sd->nl = total_num_rows;
            sd->nv = sd->nv;
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

        if (curr_sd->nv == 0) { return; }
    }

    MADRONA_GPU_SINGLE_THREAD {
        const int32_t num_smem_bytes_per_warp =
            mwGPU::SharedMemStorage::numBytesPerWarp();

        // We want to fit as much data as possible into shared memory
        uint32_t world_id = invocation_idx;
        CVSolveData *curr_sd = &solveDatas[world_id];

        uint32_t max_num_comps = max(curr_sd->nv, curr_sd->nc);

        CVSolveData::RegInfo *sizes = curr_sd->regInfos;
        uint32_t num_sizes = 0;
        // x
        sizes[num_sizes++] = { sizeof(float) * curr_sd->nv, false };
        // m_grad
        sizes[num_sizes++] = { sizeof(float) * curr_sd->nv, false };
        // p
        sizes[num_sizes++] = { sizeof(float) * curr_sd->nv, false };
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
            uint32_t cont_acc_ref_bytes = sizeof(float) * curr_sd->nc;
            uint32_t lim_acc_ref_bytes = sizeof(float) * curr_sd->nl;

            curr_sd->accRefMem = (uint8_t *)mwGPU::TmpAllocator::get().alloc(
                    cont_acc_ref_bytes + lim_acc_ref_bytes);
            curr_sd->accRefAllocatedBytes = cont_acc_ref_bytes + lim_acc_ref_bytes;
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
            uint32_t prep_bytes =
                // Mass matrix
                sizeof(SparseBlkDiag::Blk) * num_grps +
                // full vel
                sizeof(float) * curr_sd->nv +
                // free acc
                sizeof(float) * curr_sd->nv +
                // mu
                sizeof(float) * curr_sd->nc +
                // penetrations
                sizeof(float) * curr_sd->nc +
                // J_c    TODO: Make this sparse
                sizeof(float) * curr_sd->nc * curr_sd->nv +
                // diag approx of J_c
                sizeof(float) * curr_sd->nc +
                // J_e    TODO: Make this sparse
                sizeof(float) * curr_sd->nl * curr_sd->nv +
                // diag approx of J_e
                sizeof(float) * curr_sd->nl +
                // Equality residuals
                sizeof(float) * curr_sd->nl +
                // R_c
                sizeof(float) * curr_sd->nc +
                // R_e
                sizeof(float) * curr_sd->nl;

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

    uint32_t world_id = invocation_idx;
    StateManager *state_mgr = getStateManager();
    CVSolveData *curr_sd = &solveDatas[world_id];
    MADRONA_GPU_SINGLE_THREAD { curr_sd->worldID = world_id; }

    if (curr_sd->nv == 0) { return; } // Zero dofs

    BodyGroupProperties *all_properties = state_mgr->getWorldComponents<
        BodyGroupArchetype, BodyGroupProperties>(world_id);
    BodyGroupMemory *all_memories = state_mgr->getWorldComponents<
        BodyGroupArchetype, BodyGroupMemory>(world_id);
    CountT num_grps = state_mgr->numRows<BodyGroupArchetype>(world_id);

    { // Get the total mass in the world
        float mass_diag_sum = 0.f;

        mass_diag_sum = warpSumPred<float>(
            [&](uint32_t iter) -> float {
                return all_properties[iter].inertiaSum;
            }, num_grps);

        MADRONA_GPU_SINGLE_THREAD { curr_sd->massDiagSum = mass_diag_sum; }
    }

    { // Prepare the mass matrix
        SparseBlkDiag mass_sparse;
        mass_sparse.fullDim = curr_sd->nv;
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

        MADRONA_GPU_SINGLE_THREAD { curr_sd->massSparse = mass_sparse; }
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

        MADRONA_GPU_SINGLE_THREAD { curr_sd->freeAcc = free_acc; }
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

        MADRONA_GPU_SINGLE_THREAD { curr_sd->vel = full_vel; }
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

        MADRONA_GPU_SINGLE_THREAD {
            for (uint32_t i = 0; i < curr_sd->nc; ++i) {
                assert(curr_sd->mu[i] != 0.f);
            }
        }
    }

    { // Prepare prefix sum of body group DOFs in current world
        warpExclusivePrefixSumPred<uint32_t>(
                [&](uint32_t iter) {
                    return all_properties[iter].numLimits;
                },
                [&](uint32_t iter, uint32_t eq_offset) {
                    all_properties[iter].tmp.limOffset = eq_offset;

                    // Might as well store the grpIndex
                    all_properties[iter].tmp.grpIndex = iter;
                },
                curr_sd->numBodyGroups);
    }

    { // Prepare the contact jacobian and diag approx
        float *j_c = curr_sd->getContactJacobian(state_mgr);
        float *diag_approx = curr_sd->getContactDiagApprox(state_mgr);
        curr_sd->J_c = j_c;

        warpSetZero(j_c, sizeof(float) * curr_sd->nc * curr_sd->nv);
        warpSetZero(diag_approx, sizeof(float) * curr_sd->nc);

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
                                    contact_pt,
                                    j_c,
                                    c_info.pRef.tmp.qvOffset,
                                    //world_block_start[grp->tmpIdx0],
                                    curr_jacc_row,
                                    curr_sd->nc,
                                    -1.f,
                                    false);//(ct_idx == 0 && pt_idx == 0));
                        }

                        { // Compute jacobian for alt
                            computeContactJacobian(
                                    c_info.pAlt,
                                    c_info.mAlt,
                                    c_info.altBodyIndex,
                                    contact_pt,
                                    j_c,
                                    c_info.pAlt.tmp.qvOffset,
                                    //world_block_start[grp->tmpIdx0],
                                    curr_jacc_row,
                                    curr_sd->nc,
                                    1.f,
                                    false);//(ct_idx == 0 && pt_idx == 0));
                        }

                        { // Project into contact space
                            Mat3x3 C = c_info.tmpState.C;
                            // Multiply by C^T to project into contact space
                            for (CountT i = 0; i < curr_sd->nv; ++i) {
                                float *J_col = j_c +
                                        curr_sd->nc * i + curr_jacc_row;
                                Vector3 J_col_vec = { J_col[0], J_col[1], J_col[2] };
                                J_col_vec = C.transpose() * J_col_vec;
                                J_col[0] = J_col_vec.x;
                                J_col[1] = J_col_vec.y;
                                J_col[2] = J_col_vec.z;
                            }

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

    { // Prepare limit jacobian
        float *j_l = curr_sd->getLimitJacobian(state_mgr);
        float *diag_approx_l = curr_sd->getEqualityDiagApprox(state_mgr);
        float *res_l = curr_sd->getEqualityResiduals(state_mgr);

        curr_sd->J_l = j_l;
        curr_sd->diagApprox_l = diag_approx_l;
        curr_sd->limitResiduals = res_l;

        warpSetZero(j_l, sizeof(float) *
                    curr_sd->nl *
                    curr_sd->nv);

        warpSetZero(diag_approx_l, sizeof(float) * curr_sd->nl);
        warpSetZero(res_l, sizeof(float) * curr_sd->nl);

        for (uint32_t grp = 0; grp < num_grps; ++grp) {
            __syncwarp();

            BodyGroupMemory m = all_memories[grp];
            BodyGroupProperties p = all_properties[grp];

            warpLoop(
                p.numBodies,
                [&](uint32_t iter) {
                    BodyOffsets o = m.offsets(p)[iter];
                    if (o.dofType == DofType::FixedBody || o.numLimits == 0) {
                        return;
                    }

                    BodyLimitConstraint limit = m.limits(p)[o.eqOffset];

                    if (limit.type == BodyLimitConstraint::Type::None) {
                        return;
                    }

                    uint32_t glob_row_offset = p.tmp.limOffset +
                                               o.eqOffset;
                    uint32_t glob_col_offset = p.tmp.qvOffset +
                                               o.velOffset;

                    float *q = m.q(p) + o.posOffset;
                    BodyInertial &inertial = m.inertials(p)[iter];

                    switch (limit.type) {
                    case BodyLimitConstraint::Type::Hinge: {
                        if (limit.hinge.isActive(q[0])) {
                            float *to_change = j_l +
                                curr_sd->nl * glob_col_offset +
                                glob_row_offset;

                            to_change[0] =
                                limit.hinge.dConstraintViolation(q[0]);
                            res_l[glob_row_offset] = limit.hinge.constraintViolation(q[0]);
                            diag_approx_l[glob_row_offset] = inertial.approxInvMassDof[0];
                        } else {
                            float *to_change = j_l +
                                curr_sd->nl * glob_col_offset +
                                glob_row_offset;

                            to_change[0] = 0.f;
                            res_l[0] = 0.f;
                            diag_approx_l[glob_row_offset] = 0.f;
                        }
                    } break;

                    case BodyLimitConstraint::Type::Slider: {
                        if (limit.hinge.isActive(q[0])) {
                            float *to_change = j_l +
                                curr_sd->nl * glob_col_offset +
                                glob_row_offset;

                            to_change[0] =
                                limit.slider.dConstraintViolation(q[0]);
                            res_l[glob_row_offset] = limit.slider.constraintViolation(q[0]);
                            diag_approx_l[glob_row_offset] = inertial.approxInvMassDof[0];
                        } else {
                            float *to_change = j_l +
                                curr_sd->nl * glob_col_offset +
                                glob_row_offset;

                            to_change[0] = 0.f;
                            res_l[0] = 0.f;
                            diag_approx_l[glob_row_offset] = 0.f;
                        }
                    } break;

                    default: {
                        MADRONA_UNREACHABLE();
                    } break;
                    }
                });
        }
    }
}

void GaussMinimizationNode::computeContactJacobian(
        BodyGroupProperties &prop,
        BodyGroupMemory &mem,
        uint32_t body_idx,
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
                J_col[j] += coeff * S_col[j];
            }
        }

        curr_idx = offsets.parent;
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
    float* acc_ref,
    float* j_mat,
    uint32_t num_rows_j,
    uint32_t num_cols_j,
    float* vel,
    uint32_t vel_dim,
    float h,
    Fn&& residual_fn,
    float* diag_approx,
    float* R_vec)
{
    using namespace gpu_utils;

    float time_const = 2.f * h;
    constexpr float damp_ratio = 1.f;
    constexpr float d_min = 0.9f,
                    d_max = 0.95f,
                    width = 0.001f,
                    mid = 0.5f,
                    power = 2.f;

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
        R_vec[iter] = ((1.f - imp) / imp) * diag_approx[iter];
    });
}

// Might be overkill to allocate a warp per world but we can obviously
// experiment.
void GaussMinimizationNode::computeContactAccRef(int32_t invocation_idx)
{
    CV_PROF_START(t0, contAccRef);

    using namespace gpu_utils;

    // This is the global warp ID
    // uint32_t warp_id = (blockDim.x * blockIdx.x + threadIdx.x) / 32;
    uint32_t warp_id = threadIdx.x / 32;

    assert(blockDim.x == consts::numMegakernelThreads);

    StateManager *state_mgr = getStateManager();
    uint32_t world_id = invocation_idx;

    { // Do the actual computation
        CVSolveData *curr_sd = &solveDatas[world_id];

        if (curr_sd->nv == 0) { return; }

        // We want acc_ref to have priority in shared memory
        auto [acc_ref, in_smem] = [&]() -> std::pair<float *, bool> {
            const int32_t num_smem_bytes_per_warp =
                mwGPU::SharedMemStorage::numBytesPerWarp();

            uint32_t acc_ref_bytes = sizeof(float) * curr_sd->nc;
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

        if (curr_sd->nc > 0) {
            computeAccRef(
                acc_ref,
                curr_sd->J_c,
                curr_sd->nc,
                curr_sd->nv,
                curr_sd->vel,
                curr_sd->nv,
                curr_sd->h,
                // Penetrations only relevant in normal direction
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
                curr_sd->nc / 3,
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
                curr_sd->nc,
                [&](uint32_t iter) {
                    r_vec[iter] = 1.f / r_vec[iter];
                });

            if (acc_ref && in_smem) {
                float * acc_ref_glob = curr_sd->getContactAccRef(state_mgr);
                warpCopy(acc_ref_glob, acc_ref, sizeof(float) * curr_sd->nc);
                acc_ref = acc_ref_glob;
            }
        }

        __syncwarp();
    }
}

void GaussMinimizationNode::computeLimitAccRef(int32_t invocation_idx)
{
    CV_PROF_START(t0, eqAccRef);

    using namespace gpu_utils;

    // This is the global warp ID
    // uint32_t warp_id = (blockDim.x * blockIdx.x + threadIdx.x) / 32;
    uint32_t warp_id = threadIdx.x / 32;

    assert(blockDim.x == consts::numMegakernelThreads);

    StateManager *state_mgr = getStateManager();
    uint32_t world_id = invocation_idx;

    { // Do the actual computation
        CVSolveData *curr_sd = &solveDatas[world_id];
        if (curr_sd->nv == 0) { return; }

        float *residuals = curr_sd->limitResiduals;
        // We want acc_ref to have priority in shared memory
        auto [acc_ref, in_smem] = [&]() -> std::pair<float *, bool> {
            const int32_t num_smem_bytes_per_warp =
                mwGPU::SharedMemStorage::numBytesPerWarp();

            uint32_t acc_ref_bytes = sizeof(float) * curr_sd->nl;
            auto smem_buf = (uint8_t *)mwGPU::SharedMemStorage::buffer +
                            num_smem_bytes_per_warp * warp_id;

            if (acc_ref_bytes == 0) {
                return { (float *)nullptr, false };
            } else if (acc_ref_bytes < num_smem_bytes_per_warp) {
                return { (float *)smem_buf, true };
            } else {
                return {
                    (float *)curr_sd->getLimitAccRef(state_mgr),
                    false
                };
            }
        } ();

        if (curr_sd->nl > 0) {
            computeAccRef(
                acc_ref,
                curr_sd->J_l,
                curr_sd->nl,
                curr_sd->nv,
                curr_sd->vel,
                curr_sd->nv,
                curr_sd->h,
                [&](uint32_t iter) {
                    return residuals[iter];
                },
                curr_sd->getEqualityDiagApprox(state_mgr),
                curr_sd->getEqualityR(state_mgr));

            float *r_vec = curr_sd->getEqualityR(state_mgr);

            // The R becomes D now
            warpLoop(
                curr_sd->nl,
                [&](uint32_t iter) {
                    if (r_vec[iter] == 0.f) {
                        r_vec[iter] = 0.f;
                    } else {
                        //if (r_vec[ite])
                        r_vec[iter] = 1.f / r_vec[iter];
                    }
                });

            if (acc_ref && in_smem) {
                float * acc_ref_glob = curr_sd->getLimitAccRef(state_mgr);
                warpCopy(acc_ref_glob, acc_ref, sizeof(float) * curr_sd->nl);
                acc_ref = acc_ref_glob;
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

    uint32_t total_resident_warps = (blockDim.x * gridDim.x) / 32;

    // Global warp ID
    assert(blockDim.x == consts::numMegakernelThreads);

    StateManager *state_mgr = getStateManager();
    uint32_t world_id = invocation_idx;

    { // Do the computation
        CVSolveData *curr_sd = &solveDatas[world_id];

        if (curr_sd->nv == 0) { return; }

        float scale = 1.f / curr_sd->massDiagSum;

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
            if (curr_sd->nc == 0) {
                return nullptr;
            }
            return curr_sd->getContactAccRef(state_mgr);
        } ();

        float *acc_ref_eq = [&]() -> float * {
            if (curr_sd->nl == 0) {
                return nullptr;
            }
            return curr_sd->getLimitAccRef(state_mgr);
        } ();

        // We are using freeAcc as initial guess
        warpCopy(x, curr_sd->freeAcc, sizeof(float) * curr_sd->nv);
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
        float g_norm = sqrtf(norm2Warp(m_grad, curr_sd->nv));

        float g_dot_m_grad = sparseBlkDiagSolve<float, true>(
                m_grad, &curr_sd->massSparse, scratch1);

        // By now, m_grad actually has m_grad
        __syncwarp();

        warpLoop(curr_sd->nv, [&](uint32_t iter) {
            p[iter] = -m_grad[iter];
        });
        __syncwarp();

        uint32_t max_iters = 100;
        uint32_t iter = 0;

        for (; iter < max_iters; ++iter) {
            float p_norm = sqrtf(norm2Warp(p, curr_sd->nv));
            float ada_ls_tol = kTolerance * lsTolerance * p_norm / scale;

            if (p_norm < MINVAL) { break; } // search vector too small
            float alpha = exactLineSearch(
                curr_sd, jaccref_cont, jaccref_eq, 
                mxmin, p, x, ada_ls_tol, scratch1,
                curr_sd->getContactR(nullptr),
                curr_sd->getEqualityR(nullptr));
            __syncwarp();

            // No improvement
            if (alpha == 0.f) break;

            // x_{k+1} = x_k + alpha * p_k
            warpLoop(curr_sd->nv,
                [&](uint32_t iter) {
                    x[iter] += alpha * p[iter];
                });

            __syncwarp();

            float *g_new = m_grad;
            float new_fun = 0.f;
            { // Get the new gradient
                warpCopy(scratch2, m_grad, curr_sd->nv * sizeof(float));

                dobjWarp<true>(g_new, x, curr_sd, scratch1,
                               jaccref_cont, jaccref_eq, mxmin, 
                               acc_ref_cont, acc_ref_eq, false);

                __syncwarp();

                new_fun = objWarp(x, curr_sd, jaccref_cont, jaccref_eq, mxmin);
                __syncwarp();

                g_norm = sqrtf(norm2Warp(g_new, curr_sd->nv));
            }

            // Convergence checks
            if (scale * (curr_fun - new_fun) < kTolerance) { break; }
            if (scale * g_norm < kTolerance) { break; }

            {
                // Now we have scratch1 and scratch2 to play with
                // We need have access to these three at the same time:
                // g_new, Mgrad_new M_grad
                warpCopy(scratch3, g_new, curr_sd->nv * sizeof(float));

                float g_dot_m_grad_new = sparseBlkDiagSolve<float, true>(
                        m_grad, &curr_sd->massSparse, scratch1);

                // By now, m_grad actually has m_grad_new,
                //         scratch2 has m_grad
                //         scratch3 has g_new
                __syncwarp();

                // Polak-Ribiere
                float g_new_dot_mgradmin =
                    dotVectorsPred<float>(
                        [&](uint32_t iter) {
                            return scratch3[iter];
                        },
                        [&](uint32_t iter) {
                            return m_grad[iter] - scratch2[iter];
                        },
                        curr_sd->nv);
                float beta = g_new_dot_mgradmin / fmax(g_dot_m_grad, MINVAL);
                g_dot_m_grad = g_dot_m_grad_new;
                beta = fmax(0.f, beta);

                // p_{k+1} = beta * p_{k} - Mgrad
                warpLoop(
                    curr_sd->nv,
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
        &GaussMinimizationNode::allocateScratch>(data_id, {deps},
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
        &GaussMinimizationNode::computeLimitAccRef>(data_id, {cur_node},
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
    if (!ctx.singleton<CVSolveData>().enablePhysics) { return; }

    using namespace cpu_solver;
    uint32_t nc = cv_sing.nc;
    uint32_t nl = cv_sing.nl;
    uint32_t nf = cv_sing.nf;
    uint32_t nv = cv_sing.nv;
    uint32_t num_constraints = nc + nl;

    // No constraints, done
    if (num_constraints == 0) { copyResult(ctx, cv_sing.freeAcc); }

    //Allocate memory for reference acceleration, impedance, and residuals
    uint32_t num_bytes = SolverMemory::numReqBytes(nv, nc, nl, nf);
    uint32_t num_elems = (num_bytes + sizeof(SolverScratch256b) - 1) /
        sizeof(SolverScratch256b);
    MemoryRange mem = ctx.allocMemoryRange<SolverScratch256b>(num_elems);
    void *memPtr = ctx.memoryRangePointer<SolverScratch256b>(mem);
    SolverMemory sm = SolverMemory { .nv = nv, .nc = nc, .nl = nl, .nf = nf,
        .mem = mem, .memPtr = memPtr };

    float *acc_ref_c = sm.acc_ref_c();
    float *R_c = sm.DR_c();
    float *res_c = sm.res_c();
    // Limit
    float *acc_ref_e = sm.acc_ref_l();
    float *R_e = sm.DR_l();
    float *res_l = sm.res_l();
    // Friction
    float *acc_ref_f = sm.acc_ref_f();
    float *R_f = sm.R_f();
    float *res_f = sm.res_f();

    // Set residuals for contact constraints
    memset(res_c, 0, sizeof(float) * nc);
    for (uint32_t i = 0; i < cv_sing.nc / 3; i++) {
        res_c[i * 3] = -cv_sing.penetrations[i];
    }
    // Set residuals for equality constraints
    memcpy(res_l, cv_sing.limitResiduals, sizeof(float) * nl);

    // For friction loss, residuals is just zero
    memset(res_f, 0, sizeof(float) * nf);

    // Compute reference accelerations
    computeAccRef(acc_ref_c, R_c, cv_sing.vel, cv_sing.J_c,
                  nc, nv, res_c,
                  cv_sing.diagApprox_c, cv_sing.h);
    computeAccRef(acc_ref_e, R_e, cv_sing.vel, cv_sing.J_l,
                  nl, nv, res_l,
                  cv_sing.diagApprox_l, cv_sing.h);
    computeAccRef(acc_ref_f, R_f, cv_sing.vel, cv_sing.J_f,
                  nf, nv, res_f,
                  cv_sing.diagApprox_f, cv_sing.h);

    adjustContactRegularization(R_c, cv_sing.mu, cv_sing.muDim);

    // Constraint mass
    for (uint32_t i = 0; i < nc; i++) {
        R_c[i] = 1 / R_c[i];
    }
    for (uint32_t i = 0; i < nl; i++) {
        R_e[i] = 1 / R_e[i];
    }

    constexpr float tol = 1e-8f;
    constexpr float ls_tol = 0.01f;
    bool adaptive_ls = true;
    uint32_t max_iter = 100;
    uint32_t ls_iters = 50;

    // Set initial guess to be previous acceleration
    float *a_solve = sm.x();
    memcpy(a_solve, cv_sing.currAcc, sizeof(float) * nv);
    nonlinearCG(ctx, sm, tol, ls_tol, adaptive_ls, max_iter, ls_iters, cv_sing);
    copyResult(ctx, a_solve);
    ctx.freeMemoryRange(sm.mem);
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
