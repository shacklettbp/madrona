#include "cv.hpp"

namespace madrona::phys::cv {

using namespace math;
using namespace base;

namespace tasks {
#if 0
inline void computeExpandedParent(Context &ctx,
                                  BodyGroupMemory m,
                                  BodyGroupProperties p)
{
    int32_t *expandedParent = m.expandedParent(p);
    uint8_t *max_ptr = (uint8_t *)m.qVectorsPtr +
                       BodyGroupMemory::qVectorsNumBytes(p);

    // Initialize n-N_B elements
    expandedParent[0] = -1;
    for(int32_t i = 1; i < (int32_t)p.qvDim; ++i) {
        expandedParent[i] = i - 1;
    }

    // Create a mapping from body index to start of block
    int32_t *map = (int32_t *)ctx.tmpAlloc(sizeof(int32_t) * p.numBodies);

    map[0] = -1;
    
    BodyOffsets *offsets = m.offsets(p);

    for(int32_t i = 1; i < p.numBodies; ++i) {
        uint32_t n_i = offsets[i].numDofs;
        map[i] = map[i - 1] + (int32_t) n_i;
    }
    // Finish expanded parent array
    for(int32_t i = 1; i < p.numBodies; ++i) {
        int32_t parent_idx = (int32_t)(offsets[i].parent == 0xFF ? -1 :
                                       offsets[i].parent);
        expandedParent[map[i - 1] + 1] = map[parent_idx];
        ASSERT_PTR_ACCESS(expandedParent, (map[i - 1] + 1), max_ptr);
    }
}

inline void initHierarchies(Context &ctx,
                            BodyGroupMemory m,
                            BodyGroupProperties p)
{
    computeExpandedParent(ctx, m, p);
    forwardKinematics(ctx, m, p);
}

inline void combineSpatialInertias(Context &ctx,
                                   BodyGroupMemory m,
                                   BodyGroupProperties p)
{
    BodySpatialVectors* spatialVectors = m.spatialVectors(p);
    BodyOffsets* offsets = m.offsets(p);

    // Backward pass from children to parent
    for (CountT i = p.numBodies - 1; i > 0; --i) {
        InertiaTensor &spatial_inertia = spatialVectors[i].spatialInertia;
        uint32_t parent_idx = (uint32_t)offsets[i].parent;

        assert(parent_idx < 0xFF);

        InertiaTensor& spatial_inertia_parent = 
            spatialVectors[parent_idx].spatialInertia;
        spatial_inertia_parent += spatial_inertia;
    }
}

inline float * computeBodyJacobian(BodyGroupMemory &m,
                                   BodyGroupProperties &p,
                                   uint8_t cur_body_idx,
                                   Vector3 &origin,
                                   float *J,
                                   uint32_t body_dof_offset,
                                   uint32_t jac_row,
                                   uint32_t j_num_rows)
{
    // Compute prefix sum to determine the start of the block for each body
    BodyOffsets *offsets = m.offsets(p);
    BodyPhi *phis = m.bodyPhi(p);

    // Populate J_C by traversing up the hierarchy
    while(cur_body_idx != 0xFF) {
        BodyOffsets cur_offset = offsets[cur_body_idx];

        float *S = m.phiFull(p) + 2 * 6 * cur_offset.velOffset;

        // Populate columns of J_C
        computePhi(cur_offset.dofType, phis[cur_body_idx], S, origin);

        for(CountT i = 0; i < cur_offset.numDofs; ++i) {
            float *J_col = J +
                    j_num_rows * (body_dof_offset + cur_offset.velOffset + i) +
                    jac_row;

            float *S_col = S + 6 * i;
            for(CountT j = 0; j < 6; ++j) {
                J_col[j] = S_col[j];
            }
        }

        cur_body_idx = cur_offset.parent;
    }

    return J;
}

inline void computeInvMass(
        Context &ctx,
        BodyGroupMemory m,
        BodyGroupProperties p)
{
#ifdef MADRONA_GPU_MODE
    if (threadIdx.x % 32 != 0) {
        return;
    }

    uint32_t warp_id = threadIdx.x / 32;

    const int32_t num_smem_bytes_per_warp =
        mwGPU::SharedMemStorage::numBytesPerWarp();
    auto smem_buf = (uint8_t *)mwGPU::SharedMemStorage::buffer +
                    num_smem_bytes_per_warp * warp_id;

    float *A = (float *)smem_buf;
    float *J = A + 36;
    float *MinvJT = J + 6 * p.qvDim;

    assert(sizeof(float) * (36 + 6 * p.qvDim + p.qvDim * 6) <
            num_smem_bytes_per_warp);
#else
    // For each body, find translational and rotational inverse weight
    //  by computing A = J M^{-1} J^T
    float A[36] = {}; // 6x6
    float J[6 * p.qvDim]; // col-major (shape 6 x numDofs)
    float MinvJT[p.qvDim * 6]; // col-major (shape numDofs x 6)
#endif

    BodyOffsets *offsets = m.offsets(p);
    BodyTransform *transforms = m.bodyTransforms(p);
    BodyInertial *inertials = m.inertials(p);

    // Compute the inverse weight for each body
    for (CountT i_body = 0; i_body < p.numBodies; ++i_body) {
        BodyTransform transform = transforms[i_body];

        // Compute J
        memset(J, 0.f, 6 * p.qvDim * sizeof(float));
        computeBodyJacobian(
                m,
                p,
                (uint8_t)i_body,
                transform.com,
                J, 0, 0, 6);

        // Helper
        auto Jb = [&](int32_t row, int32_t col) -> float& {
            return J[row + 6 * col];
        };
        auto MinvJTb = [&](int32_t row, int32_t col) -> float& {
            return MinvJT[row + p.qvDim * col];
        };
        auto Ab = [&](int32_t row, int32_t col) -> float& {
            return A[row + 6 * col];
        };

        // Copy into J^T
        for (CountT i = 0; i < 6; ++i) {
            for (CountT j = 0; j < p.qvDim; ++j) {
                MinvJTb(j, i) = Jb(i, j);
            }
        }
        // M^{-1} J^T
        for (CountT i = 0; i < 6; ++i) {
            float *col = MinvJT + i * p.qvDim;
            solveM( p, m, col);
        }

        // A = J M^{-1} J^T
        memset(A, 0.f, 36 * sizeof(float));
        for (CountT i = 0; i < 6; ++i) {
            for (CountT j = 0; j < 6; ++j) {
                for (CountT k = 0; k < p.qvDim; ++k) {
                    Ab(i, j) += Jb(i, k) * MinvJTb(k, j);
                }
            }
        }

        // Compute the inverse weight
        inertials[i_body].approxInvMassTrans =
            (Ab(0, 0) + Ab(1, 1) + Ab(2, 2)) / 3.f;
        inertials[i_body].approxInvMassRot =
            (Ab(3, 3) + Ab(4, 4) + Ab(5, 5)) / 3.f;
    }

    // For each DOF, find the inverse weight
    uint32_t dof_offset = 0;
    for (CountT i_body = 0; i_body < p.numBodies; ++i_body) {
#if 0
        Entity body = grp.bodies(ctx)[i_body];
        auto body_dofs = ctx.get<DofObjectNumDofs>(body);
        auto &dof_inertial = ctx.get<DofObjectInertial>(body);
#endif
        BodyOffsets offset = offsets[i_body];
        auto &dof_inertial = inertials[i_body];

        // Jacobian size (body dofs x total dofs)
        auto Jd = [&](int32_t row, int32_t col) -> float& {
            return J[row + offset.numDofs * col];
        };
        // J^T and M^{-1}J^T (total dofs x body dofs)
        auto MinvJTd = [&](int32_t row, int32_t col) -> float& {
            return MinvJT[row + p.qvDim * col];
        };
        // A = JM^{-1}J^T. (body dofs x body dofs)
        auto Ad = [&](int32_t row, int32_t col) -> float& {
            return A[row + offset.numDofs * col];
        };

        // Fill in 1 for the corresponding body dofs
        memset(J, 0.f, 6 * p.qvDim * sizeof(float));
        for (CountT i = 0; i < offset.numDofs; ++i) {
            Jd(i, i + dof_offset) = 1.f;
        }

        // Copy into J^T
        for (CountT i = 0; i < offset.numDofs; ++i) {
            for (CountT j = 0; j < p.qvDim; ++j) {
                MinvJTd(j, i) = Jd(i, j);
            }
        }

        // M^{-1} J^T. (J^T is total dofs x body dofs)
        for (CountT i = 0; i < offset.numDofs; ++i) {
            float *col = MinvJT + i * p.qvDim;
            solveM(p, m, col);
        }

        // A = J M^{-1} J^T
        memset(A, 0.f, offset.numDofs * offset.numDofs * sizeof(float));
        for (CountT i = 0; i < offset.numDofs; ++i) {
            for (CountT j = 0; j < offset.numDofs; ++j) {
                for (CountT k = 0; k < p.qvDim; ++k) {
                    Ad(i, j) += Jd(i, k) * MinvJTd(k, j);
                }
            }
        }

        // Update the inverse mass of each DOF
        if (offset.numDofs == 6) {
           dof_inertial.approxInvMassDof[0] = 
               dof_inertial.approxInvMassDof[1] =
               dof_inertial.approxInvMassDof[2] =
               (Ad(0, 0) + Ad(1, 1) + Ad(2, 2)) / 3.f;
           dof_inertial.approxInvMassDof[3] =
               dof_inertial.approxInvMassDof[4] =
               dof_inertial.approxInvMassDof[5] =
                (Ad(3, 3) + Ad(4, 4) + Ad(5, 5)) / 3.f;
        } else if (offset.numDofs == 3) {
            dof_inertial.approxInvMassDof[0] =
                dof_inertial.approxInvMassDof[1] =
                dof_inertial.approxInvMassDof[2] =
                (Ad(0, 0) + Ad(1, 1) + Ad(2, 2)) / 3.f;
        } else {
            dof_inertial.approxInvMassDof[0] = Ad(0, 0);
        }

        dof_offset += offset.numDofs;
    }
}
#endif
}

#if 0
TaskGraphNodeID setupCVInitTasks(
        TaskGraphBuilder &builder,
        Span<const TaskGraphNodeID> deps)
{
    // Initialize memory and run forward kinematics
    auto node = builder.addToGraph<ParallelForNode<Context,
         tasks::initHierarchies,
             BodyGroupMemory,
             BodyGroupProperties
         >>(deps);

    // Initialization for initial position (e.g., inverse weights)
    node = builder.addToGraph<ParallelForNode<Context,
         tasks::computeGroupCOM,
            BodyGroupProperties,
            BodyGroupMemory
        >>({node});

    node = builder.addToGraph<ParallelForNode<Context,
         tasks::computeSpatialInertiasAndPhi,
            DofObjectGroup
        >>({node});

    node = builder.addToGraph<ParallelForNode<Context,
         tasks::combineSpatialInertias,
            BodyGroupMemory,
            BodyGroupProperties
        >>({node});

#ifdef MADRONA_GPU_MODE
    node = builder.addToGraph<CustomParallelForNode<Context,
         tasks::compositeRigidBody, 32, 1,
            BodyGroupProperties,
            BodyGroupMemory
        >>({node});
#else
    node = builder.addToGraph<ParallelForNode<Context,
         tasks::compositeRigidBody,
            BodyGroupProperties,
            BodyGroupMemory
        >>({node});
#endif

#ifdef MADRONA_GPU_MODE
    node = builder.addToGraph<CustomParallelForNode<Context,
         tasks::computeInvMass, 32, 1,
         BodyGroupMemory,
         BodyGroupProperties
     >>({node});
#else
    node = builder.addToGraph<ParallelForNode<Context,
         tasks::computeInvMass,
         BodyGroupMemory,
         BodyGroupProperties
     >>({node});
#endif

    node =
        builder.addToGraph<ParallelForNode<Context, tasks::convertPostSolve,
            Position,
            Rotation,
            Scale,
            LinkParentDofObject
        >>({node});

    return node;
}
#endif
    
}
