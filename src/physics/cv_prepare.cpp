#include <algorithm>
#include <madrona/state.hpp>
#include <madrona/utils.hpp>
#include <madrona/physics.hpp>
#include <madrona/context.hpp>
#include <madrona/cv_physics.hpp>
#include <madrona/taskgraph.hpp>

#include "physics_impl.hpp"

using namespace madrona::math;
using namespace madrona::base;

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

template <typename ArchetypeT, typename ComponentT>
ComponentT * getRows(StateManager *state_mgr, uint32_t world_id)
{
#ifdef MADRONA_GPU_MODE
    (void)world_id;
    return state_mgr->getArchetypeComponent<ArchetypeT, ComponentT>();
#else
    return state_mgr->getWorldComponents<ArchetypeT, ComponentT>(world_id);
#endif
}

struct MRElement128b {
    uint8_t d[128];
};

struct SolverScratch256b {
    uint8_t d[256];
};

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

namespace tasks {
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

// Computes the Phi matrix from generalized velocities to Plücker coordinates
inline float* computePhi(
    DofType dof_type,
    BodyPhi &body_phi,
    float* S,
    Vector3 origin)
{
    float* phi = body_phi.phi;

    if (dof_type == DofType::FreeBody) {
        // S = [1_3x3 r^x; 0 1_3x3], column-major
        memset(S, 0.f, 6 * 6 * sizeof(float));
        // Diagonal identity
        for(CountT i = 0; i < 6; ++i) {
            S[i * 6 + i] = 1.f;
        }
        // r^x Skew symmetric matrix
        Vector3 comPos = {phi[0], phi[1], phi[2]};
        comPos -= origin;
        S[0 + 6 * 4] = -comPos.z;
        S[0 + 6 * 5] = comPos.y;
        S[1 + 6 * 3] = comPos.z;
        S[1 + 6 * 5] = -comPos.x;
        S[2 + 6 * 3] = -comPos.y;
        S[2 + 6 * 4] = comPos.x;
    } else if (dof_type == DofType::Slider) {
        // This is just the axis of the slider.
        S[0] = phi[0];
        S[1] = phi[1];
        S[2] = phi[2];
        S[3] = 0.f;
        S[4] = 0.f;
        S[5] = 0.f;
    } else if (dof_type == DofType::Hinge) {
        // S = [r \times hinge; hinge]
        Vector3 hinge = {phi[0], phi[1], phi[2]};
        Vector3 anchorPos = {phi[3], phi[4], phi[5]};
        anchorPos -= origin;
        Vector3 r_cross_hinge = anchorPos.cross(hinge);
        S[0] = r_cross_hinge.x;
        S[1] = r_cross_hinge.y;
        S[2] = r_cross_hinge.z;
        S[3] = hinge.x;
        S[4] = hinge.y;
        S[5] = hinge.z;
    } else if (dof_type == DofType::Ball) {
        // This will just get right-multiplied by the angular velocity
        Vector3 anchor_pos = {phi[0], phi[1], phi[2]};
        anchor_pos -= origin;

        // We need to right multiply these by the parent composed rotation
        // matrix.
        Mat3x3 rx = skewSymmetricMatrix(anchor_pos);
        Quat parent_composed_rot = Quat{
            phi[3], phi[4], phi[5], phi[6]
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

// Same as the Phi matrix but only the first 3 rows (translational part)
inline float* computePhiTrans(
    DofType dof_type,
    BodyPhi &body_phi,
    float* S,
    Vector3 origin)
{
    float* phi = body_phi.phi;

    if (dof_type == DofType::FreeBody) {
        memset(S, 0.f, 6 * 3 * sizeof(float));
        for(CountT i = 0; i < 3; ++i) {
            S[i * 3 + i] = 1.f;
        }
        Vector3 comPos = {phi[0], phi[1], phi[2]};
        comPos -= origin;
        S[0 + 3 * 4] = -comPos.z;
        S[0 + 3 * 5] = comPos.y;
        S[1 + 3 * 3] = comPos.z;
        S[1 + 3 * 5] = -comPos.x;
        S[2 + 3 * 3] = -comPos.y;
        S[2 + 3 * 4] = comPos.x;
    } else if (dof_type == DofType::Slider) {
        S[0] = phi[0];
        S[1] = phi[1];
        S[2] = phi[2];
    } else if (dof_type == DofType::Hinge) {
        // S = [r \times hinge; hinge]
        Vector3 hinge = {phi[0], phi[1], phi[2]};
        Vector3 anchorPos = {phi[3], phi[4], phi[5]};
        anchorPos -= origin;
        Vector3 r_cross_hinge = anchorPos.cross(hinge);
        S[0] = r_cross_hinge.x;
        S[1] = r_cross_hinge.y;
        S[2] = r_cross_hinge.z;
    } else if (dof_type == DofType::Ball) {
        Vector3 anchor_pos = {phi[0], phi[1], phi[2]};
        anchor_pos -= origin;
        Mat3x3 rx = skewSymmetricMatrix(anchor_pos);
        Quat parent_composed_rot = Quat{
            phi[3], phi[4], phi[5], phi[6]
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
    return S;
}


inline void computeGroupCOM(Context &ctx,
                           BodyGroupProperties &prop,
                           BodyGroupMemory &mem)
{
    (void)ctx;
    Vector3 hierarchy_com = Vector3::zero();
    float total_mass = 0.f;

    uint32_t numBodies = prop.numBodies;
    BodyInertial *inertials = mem.inertials(prop);
    BodyTransform* transforms = mem.bodyTransforms(prop);
    for (int i = 0; i < numBodies; ++i) {
        BodyInertial body_inertia = inertials[i];
        BodyTransform body_transform = transforms[i];

        hierarchy_com += body_inertia.mass * body_transform.com;
        total_mass += body_inertia.mass;
    }

    prop.comPos = hierarchy_com / total_mass;
}

inline void computeSpatialInertiasAndPhi(Context &ctx,
                                         DofObjectGroup obj_grp)
{
    BodyGroupProperties &prop = ctx.get<BodyGroupProperties>(obj_grp.bodyGroup);
    BodyGroupMemory &mem = ctx.get<BodyGroupMemory>(obj_grp.bodyGroup);
    Vector3 body_grp_com_pos = prop.comPos;

    // --------- Compute spatial inertias -------------

    BodyInertial inertial = mem.inertials(prop)[obj_grp.idx];
    BodyTransform transform = mem.bodyTransforms(prop)[obj_grp.idx];
    Diag3x3 inertia = inertial.inertia * prop.globalScale * prop.globalScale;
    float mass = inertial.mass;

    // We need to find inertia tensor in world space orientation
    // I_world = R * I * R^T (since R^T transforms from world to local)
    Mat3x3 rot_mat = Mat3x3::fromQuat(transform.composedRot);
    Mat3x3 i_world_frame = rot_mat * inertia * rot_mat.transpose();

    // Compute the 3x3 skew-symmetric matrix (r^x)
    // (where r is from Plücker origin to COM)
    Vector3 adjustedCom = transform.com - body_grp_com_pos;
    Mat3x3 sym_mat = skewSymmetricMatrix(adjustedCom);
    // (I_world - m r^x r^x)
    Mat3x3 inertia_mat = i_world_frame - (mass * sym_mat * sym_mat);

    // Take only the upper triangular part (since it's symmetric)
    InertiaTensor& spatialInertia = mem.spatialVectors(prop)[obj_grp.idx].spatialInertia;

    spatialInertia.spatial_inertia[0] = inertia_mat[0][0];
    spatialInertia.spatial_inertia[1] = inertia_mat[1][1];
    spatialInertia.spatial_inertia[2] = inertia_mat[2][2];
    spatialInertia.spatial_inertia[3] = inertia_mat[1][0];
    spatialInertia.spatial_inertia[4] = inertia_mat[2][0];
    spatialInertia.spatial_inertia[5] = inertia_mat[2][1];

    // Rest of parameters
    spatialInertia.mass = mass;
    spatialInertia.mCom = mass * adjustedCom;

    // --------- Compute phi  -------------
    // Compute the full matrix/linear operator Phi with body group COM as origin

    BodyOffsets offset = mem.offsets(prop)[obj_grp.idx];

    DofType dof_type = offset.dofType;
    Vector3 com_pos = prop.comPos;
    BodyPhi* phis = mem.bodyPhi(prop);

    uint32_t velOffset = offset.velOffset;
    uint32_t S_offset = 2 * 6 * velOffset;
    float* S = mem.phiFull(prop) + S_offset;

    if (dof_type != DofType::None && dof_type != DofType::FixedBody) {
        computePhi(dof_type, phis[obj_grp.idx], S, com_pos);
    }

    // --------- Add external forces -------------
    uint32_t num_dofs = BodyOffsets::getDofTypeDim(dof_type);
    float *tau = mem.biasVector(prop);
    float *force = mem.f(prop);
    memcpy(tau + velOffset, force + velOffset, num_dofs * sizeof(float));
}

inline float* computePhiDot(DofType dof_type,
                            float *S,
                            float *S_dot,
                            SpatialVector &v_hat)
{
    if (dof_type == DofType::FreeBody) {
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
    else if (dof_type == DofType::Slider) {
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
    else if (dof_type == DofType::Hinge) {
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
    else if (dof_type == DofType::Ball) {
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

// J_C = C^T[e_{b1} S_1, e_{b2} S_2, ...], col-major
//  where e_{bi} = 1 if body i is an ancestor of b
//  C^T projects into the contact space
inline float* computeContactJacobian(DofObjectGroup &obj_grp,
                                     BodyGroupProperties &prop,
                                     BodyGroupMemory &mem,
                                     Mat3x3 &C,
                                     Vector3 &origin,
                                     float *J,
                                     uint32_t body_dof_offset,
                                     uint32_t jac_row,
                                     uint32_t j_num_rows,
                                     float coeff)
{
    BodyOffsets *offsets = mem.offsets(prop);
    BodyPhi* phis = mem.bodyPhi(prop);
    float *S =  mem.phiFull(prop);

    // Populate J_C by traversing up the hierarchy
    uint32_t body_idx = obj_grp.idx;
    while(body_idx != -1) {
        // S is no longer used, can overwrite
        BodyOffsets offset = offsets[body_idx];
        uint32_t velOffset = offsets[body_idx].velOffset;
        uint32_t S_offset = 2 * 6 * velOffset;
        uint32_t numDofs = BodyOffsets::getDofTypeDim(offset.dofType);
        float* S_i = S + S_offset;
        computePhiTrans(offset.dofType, phis[body_idx], S_i, origin);
        // Only use translational part of S
        for(CountT i = 0; i < numDofs; ++i) {
            float *J_col = J + j_num_rows * (body_dof_offset + velOffset + i) +
                    jac_row;
            float *S_col = S + 3 * i;
            for(CountT j = 0; j < 3; ++j) {
                J_col[j] = S_col[j];
            }
        }
        body_idx = offsets[body_idx].parent;
    }

    // Multiply by C^T to project into contact space
    for(CountT i = 0; i < prop.qvDim; ++i) {
        float *J_col = J + j_num_rows * (body_dof_offset + i) + jac_row;
        Vector3 J_col_vec = { J_col[0], J_col[1], J_col[2] };
        J_col_vec = C.transpose() * J_col_vec;
        J_col[0] = coeff * J_col_vec.x;
        J_col[1] = coeff * J_col_vec.y;
        J_col[2] = coeff * J_col_vec.z;
    }
    return J;
}

// y = Mx. Based on Table 6.5 in Featherstone
inline void mulM(BodyGroupProperties &prop, BodyGroupMemory &mem,
                 float *x, float *y) {
    CountT total_dofs = prop.qvDim;

    int32_t *expandedParent = mem.expandedParent(prop);

    float *massMatrix = mem.massMatrix(prop);

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
inline void solveM(BodyGroupProperties& prop, BodyGroupMemory& mem, float* x) {
    CountT total_dofs = prop.qvDim;

    int32_t *expandedParent = mem.expandedParent(prop);
    float *massMatrixLTDL = mem.massLTDLMatrix(prop);

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


// CRB: Compute the Mass Matrix (n_dofs x n_dofs)
inline void compositeRigidBody(Context &ctx,
                               BodyGroupProperties &prop,
                               BodyGroupMemory &mem)
{
    (void)ctx;
    // ----------------- Composite rigid body -----------------
    // Mass Matrix of this entire body group, column-major
    uint32_t total_dofs = prop.qvDim;
    BodyOffsets *offsets = mem.offsets(prop);
    BodySpatialVectors *spatialVectors = mem.spatialVectors(prop);

    float *S =  mem.phiFull(prop);
    float *M = mem.massMatrix(prop);
    memset(M, 0.f, total_dofs * total_dofs * sizeof(float));

    // Backward pass
    for (CountT i = prop.numBodies-1; i >= 0; --i) {
        uint32_t velOffset = offsets[i].velOffset;
        uint32_t S_offset = 2 * 6 * velOffset;
        uint32_t num_dofs = BodyOffsets::getDofTypeDim(offsets[i].dofType);
        float* S_i = S + S_offset;

        float *F = mem.scratch(prop);

        for(CountT col = 0; col < num_dofs; ++col) {
            float *S_col = S_i + 6 * col;
            float *F_col = F + 6 * col;
            spatialVectors[i].spatialInertia.multiply(S_col, F_col);
        }

        // M_{ii} = S_i^T I_i^C S_i = F^T S_i
        float *M_ii = M + velOffset * total_dofs + velOffset;
        for(CountT row = 0; row < num_dofs; ++row) {
            float *F_col = F + 6 * row; // take col for transpose
            for(CountT col = 0; col < num_dofs; ++col) {
                float *S_col = S_i + 6 * col;
                for(CountT k = 0; k < 6; ++k) {
                    M_ii[row + total_dofs * col] += F_col[k] * S_col[k];
                }
            }
        }

        // Traverse up hierarchy
        uint8_t j = offsets[i].parent;
        while(j != -1) {
            uint32_t velOffset_j = offsets[j].velOffset;
            uint32_t S_offset_j = 2 * 6 * velOffset_j;
            uint32_t j_num_dofs = BodyOffsets::getDofTypeDim(offsets[j].dofType);
            float *S_j = S + S_offset_j;

            // M_{ij} = M{ji} = F^T S_j
            float *M_ij = M + velOffset + total_dofs * velOffset_j; // row i, col j
            float *M_ji = M + velOffset_j + total_dofs * velOffset; // row j, col i
            for(CountT row = 0; row < num_dofs; ++row) {
                float *F_col = F + 6 * row; // take col for transpose
                for(CountT col = 0; col < j_num_dofs; ++col) {
                    float *S_col = S_j + 6 * col;
                    for(CountT k = 0; k < 6; ++k) {
                        M_ij[row + total_dofs * col] += F_col[k] * S_col[k];
                        M_ji[col + total_dofs * row] += F_col[k] * S_col[k];
                    }
                }
            }
            j = offsets[j].parent;
        }
    }

    // ----------------- Compute LTDL factorization -----------------
    // First copy M to LTDL
    float *LTDL = mem.massLTDLMatrix(prop);
    memcpy(LTDL, M, total_dofs * total_dofs * sizeof(float));

    // Helper
    auto ltdl = [&](int32_t row, int32_t col) -> float& {
        return LTDL[row + total_dofs * col];
    };

    int32_t *expandedParent = mem.expandedParent(prop);

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

    // ----------------- Sum Inertia Diagonals -----------------
    // Sum the diagonals of the mass matrix (used for solver scaling)
    float inertiaSum = 0.f;
    for (CountT i = 0; i < prop.qvDim; ++i) {
        inertiaSum += M[i + i * (prop.qvDim)];
    }
    prop.inertiaSum = inertiaSum;

    // ----------------- Compute Free Acceleration -----------------
    float *bias = mem.biasVector(prop);

    for (CountT i = 0; i < prop.qvDim; ++i) {
        bias[i] = -bias[i];
    }

    // This overwrites bias with the acceleration
    solveM(prop, mem, bias);
}

// Recursive Newton Euler algorithm: Compute bias forces and gravity
inline void rneAndCombineSpatialInertias(Context &ctx,
                                         BodyGroupProperties &prop,
                                         BodyGroupMemory &mem)
{
    PhysicsSystemState &physics_state = ctx.singleton<PhysicsSystemState>();
    uint32_t total_dofs = prop.qvDim;

    // Forward pass. Find in Plücker coordinates:
    //  1. velocities. v_i = v_{parent} + S * \dot{q_i}
    //  2. accelerations. a_i = a_{parent} + \dot{S} * \dot{q_i} + S * \ddot{q_i}
    //  3. forces. f_i = I_i a_i + v_i [spatial star cross] I_i v_i
    float* qv = mem.qv(prop);
    BodyOffsets* offsets = mem.offsets(prop);
    BodySpatialVectors* spatialVectors = mem.spatialVectors(prop);

    for (uint32_t i = 0; i < prop.numBodies; ++i) {
        BodyOffsets body_offset = offsets[i];
        DofType dof_type = body_offset.dofType;
        BodySpatialVectors& spatial_vector = spatialVectors[i];

        uint32_t num_dofs = BodyOffsets::getDofTypeDim(dof_type);
        uint32_t velOffset = body_offset.velOffset;
        uint32_t S_offset = 2 * 6 * velOffset;
        uint32_t S_dot_offset = S_offset + 6 * num_dofs;

        float* velocity = qv + velOffset;
        float* S = mem.phiFull(prop) + S_offset;
        float* S_dot = mem.phiFull(prop) + S_dot_offset;

        if (dof_type == DofType::FreeBody) {
            // Free bodies must be root of their hierarchy
            SpatialVector v_body = {
                {velocity[0], velocity[1], velocity[2]}, Vector3::zero()
            };
            computePhiDot(dof_type, S, S_dot, v_body);

            // v_0 = 0, a_0 = -g (fictitious upward acceleration)
            spatial_vector.sAcc = {-physics_state.g, Vector3::zero()};
            spatial_vector.sVel = {Vector3::zero(), Vector3::zero()};

            // S\dot{q_i} and \dot{S}\dot{q_i}
            for (uint32_t j = 0; j < 6; ++j) {
                for (uint32_t k = 0; k < num_dofs; ++k) {
                    spatial_vector.sVel[j] += S[j + 6 * k] * velocity[k];
                    spatial_vector.sAcc[j] += S_dot[j + 6 * k] * velocity[k];
                }
            }
        }
        else if (dof_type == DofType::FixedBody) {
            // Fixed bodies must also be root of their hierarchy
            // tmp_state.sVel = {Vector3::zero(), Vector3::zero()};
            // tmp_state.sAcc = {-physics_state.g, Vector3::zero()};
            if (body_offset.parent == -1) {
                spatial_vector.sVel = {Vector3::zero(), Vector3::zero()};
                spatial_vector.sAcc = {-physics_state.g, Vector3::zero()};
            } else {
                BodySpatialVectors& parent_spatial_vector = spatialVectors[body_offset.parent];
                spatial_vector.sVel = parent_spatial_vector.sVel;
                spatial_vector.sAcc = parent_spatial_vector.sAcc;
            }
        }
        else if (dof_type == DofType::Slider) {
            assert(body_offset.parent != -1);
            BodySpatialVectors& parent_spatial_vector = spatialVectors[body_offset.parent];
            spatial_vector.sVel = parent_spatial_vector.sVel;
            spatial_vector.sAcc = parent_spatial_vector.sAcc;

            // v_i = v_{parent} + S * \dot{q_i}, compute S * \dot{q_i}
            // a_i = a_{parent} + \dot{S} * \dot{q_i} [+ S * \ddot{q_i}, which is 0]
            computePhiDot(dof_type, S, S_dot, parent_spatial_vector.sVel);

            float q_dot = velocity[0];
            for (int j = 0; j < 6; ++j) {
                spatial_vector.sVel[j] += S[j] * q_dot;
                spatial_vector.sAcc[j] += S_dot[j] * q_dot;
            }
        }
        else if (dof_type == DofType::Hinge) {
            assert(body_offset.parent != -1);
            BodySpatialVectors& parent_spatial_vector = spatialVectors[body_offset.parent];
            spatial_vector.sVel = parent_spatial_vector.sVel;
            spatial_vector.sAcc = parent_spatial_vector.sAcc;

            // v_i = v_{parent} + S * \dot{q_i}, compute S * \dot{q_i}
            // a_i = a_{parent} + \dot{S} * \dot{q_i} [+ S * \ddot{q_i}, which is 0]
            // Note: we are using the parent velocity here (for hinge itself)
            computePhiDot(dof_type, S, S_dot, parent_spatial_vector.sVel);

            float q_dot = velocity[0];
            for (int j = 0; j < 6; ++j) {
                spatial_vector.sVel[j] += S[j] * q_dot;
                spatial_vector.sAcc[j] += S_dot[j] * q_dot;
            }
        }
        else if (dof_type == DofType::Ball) {
            assert(body_offset.parent != -1);
            BodySpatialVectors& parent_spatial_vector = spatialVectors[body_offset.parent];
            spatial_vector.sVel = parent_spatial_vector.sVel;
            spatial_vector.sAcc = parent_spatial_vector.sAcc;

            // v_i = v_{parent} + S * \dot{q_i}, compute S * \dot{q_i}
            // a_i = a_{parent} + \dot{S} * \dot{q_i} [+ S * \ddot{q_i}, which is 0]
            // Note: we are using the parent velocity here (for hinge itself)
            computePhiDot(dof_type, S, S_dot, parent_spatial_vector.sVel);

            float *q_dot = velocity;
            for (int j = 0; j < 6; ++j) {
                // TODO: Probably should switch to row major - this isn't
                // particularly cache-friendly.
                spatial_vector.sVel[j] += S[j + 6 * 0] * q_dot[0] +
                                     S[j + 6 * 1] * q_dot[1] +
                                     S[j + 6 * 2] * q_dot[2];

                spatial_vector.sAcc[j] += S_dot[j + 6 * 0] * q_dot[0] +
                                     S_dot[j + 6 * 1] * q_dot[1] +
                                     S_dot[j + 6 * 2] * q_dot[2];
            }
        } else {
            MADRONA_UNREACHABLE();
        }

        // f_i = I_i a_i + v_i [spatial star cross] I_i v_i
        spatial_vector.sForce = spatial_vector.spatialInertia.multiply(spatial_vector.sAcc);
        spatial_vector.sForce += spatial_vector.sVel.crossStar(
            spatial_vector.spatialInertia.multiply(spatial_vector.sVel));
    }

    // Backward pass to find bias forces
    float *tau = mem.biasVector(prop);

    CountT dof_index = total_dofs;
    for (CountT i = prop.numBodies-1; i >= 0; --i) {
        BodyOffsets body_offset = offsets[i];
        BodySpatialVectors& spatial_vector = spatialVectors[i];
        uint32_t num_dofs = BodyOffsets::getDofTypeDim(body_offset.dofType);
        uint32_t velOffset = body_offset.velOffset;
        uint32_t S_offset = 2 * 6 * velOffset;

        // tau_i = S_i^T f_i
        dof_index -= num_dofs;
        float* S = mem.phiFull(prop) + S_offset;
        for(CountT row = 0; row < num_dofs; ++row) {
            float *S_col = S + 6 * row;
            for(CountT k = 0; k < 6; ++k) {
                tau[dof_index + row] += S_col[k] * spatial_vector.sForce[k];
            }
        }

        // Add to parent's force
        if (body_offset.parent != -1) {
            BodySpatialVectors& parent_spatial_vector = spatialVectors[body_offset.parent];
            parent_spatial_vector.sForce += spatial_vector.sForce;
        }
    }

    // ----------------- Combine Spatial Inertias -----------------
    uint32_t numBodies = prop.numBodies;
    // Backward pass from children to parent
    for (CountT i = numBodies-1; i > 0; --i) {
        InertiaTensor& spatialInertia = spatialVectors[i].spatialInertia;
        uint32_t parentIdx = offsets[i].parent;
        InertiaTensor& spatialInertiaParent = spatialVectors[parentIdx].spatialInertia;
        spatialInertiaParent += spatialInertia;
    }
}

inline void processContacts(Context &ctx,
                            ContactConstraint &contact,
                            ContactTmpState &tmp_state)
{
    LinkParentDofObject &link_parent_ref = ctx.get<LinkParentDofObject>(contact.ref);
    LinkParentDofObject &link_parent_alt = ctx.get<LinkParentDofObject>(contact.alt);

    Entity ref_grp = link_parent_ref.bodyGroup;
    Entity alt_grp = link_parent_alt.bodyGroup;
    uint32_t ref_body_idx = link_parent_ref.bodyIdx;
    uint32_t alt_body_idx = link_parent_alt.bodyIdx;

    BodyGroupProperties &prop_ref = ctx.get<BodyGroupProperties>(ref_grp);
    BodyGroupProperties &prop_alt = ctx.get<BodyGroupProperties>(alt_grp);

    BodyGroupMemory &mem_ref = ctx.get<BodyGroupMemory>(ref_grp);
    BodyGroupMemory &mem_alt = ctx.get<BodyGroupMemory>(alt_grp);

    float mu_ref = mem_ref.mus(prop_ref)[ref_body_idx];
    float mu_alt = mem_alt.mus(prop_alt)[alt_body_idx];

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
    float mu = fminf(mu_ref, mu_alt);
    tmp_state.mu = mu;
}

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
    // TODO!
}

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
             tasks::computeGroupCOM,
                BodyGroupProperties,
                BodyGroupMemory
            >>({run_narrowphase});

        auto compute_spatial_inertia_and_phi = builder.addToGraph<ParallelForNode<Context,
             tasks::computeSpatialInertiasAndPhi,
                DofObjectGroup
            >>({compute_body_coms});

        auto rne_and_combine_spatial_inertias = builder.addToGraph<ParallelForNode<Context,
             tasks::rneAndCombineSpatialInertias,
                BodyGroupProperties,
                BodyGroupMemory
            >>({compute_spatial_inertia_and_phi});

        auto composite_rigid_body = builder.addToGraph<ParallelForNode<Context,
             tasks::compositeRigidBody,
                BodyGroupProperties,
                BodyGroupMemory
            >>({rne_and_combine_spatial_inertias});

        auto contact_node = builder.addToGraph<ParallelForNode<Context,
             tasks::processContacts,
                ContactConstraint,
                ContactTmpState
            >>({composite_rigid_body});

        auto thing = builder.addToGraph<ParallelForNode<Context,
             tasks::brobdingnag,
                CVSolveData
            >>({contact_node});
    }

    return cur_node;
}



void getSolverArchetypeIDs(uint32_t *contact_archetype_id,
                           uint32_t *joint_archetype_id)
{
    *contact_archetype_id = TypeTracker::typeID<Contact>();
    *joint_archetype_id = TypeTracker::typeID<Joint>();
}
}
