#include "cv.hpp"
#include <algorithm>
#include <madrona/state.hpp>
#include <madrona/utils.hpp>
#include <madrona/physics.hpp>
#include <madrona/context.hpp>
#include <madrona/cv_physics.hpp>
#include <madrona/taskgraph.hpp>

#include "cv_gpu.hpp"

#include "physics_impl.hpp"

using namespace madrona::math;
using namespace madrona::base;

namespace madrona::phys::cv {

namespace tasks {
void forwardKinematics(Context &,
                       BodyGroupMemory m,
                       BodyGroupProperties p)
{
    float *all_q = m.q(p);
    BodyTransform *all_transforms = m.bodyTransforms(p);
    BodyHierarchy *all_hiers = m.hierarchies(p);
    BodyPhi *all_phi = m.bodyPhi(p);
    BodyOffsets *all_offsets = m.offsets(p);

    { // Set the parent's state (we require that the root is fixed or free body
        uint32_t root_q_offset = all_offsets[0].posOffset;
        float *q = all_q + root_q_offset;

        Vector3 com = { q[0], q[1], q[2] };

        all_transforms[0] = {
            .com = com,
            .composedRot = { q[3], q[4], q[5], q[6] },
        };

        // omega remains unchanged, and v only depends on the COM position
        all_phi[0].phi[0] = com[0];
        all_phi[0].phi[1] = com[1];
        all_phi[0].phi[2] = com[2];
    }

    // Forward pass from parent to children
    for (int i = 1; i < (int)p.numBodies; ++i) {
        BodyOffsets offsets = all_offsets[i];

        const float *q = all_q + all_offsets[i].posOffset;

        BodyTransform *curr_transform = all_transforms + i;
        BodyPhi *curr_phi = all_phi + i;

        BodyHierarchy hier = all_hiers[i];
        BodyTransform parent_transform = all_transforms[offsets.parent];

        float s = p.globalScale;

        // We can calculate our stuff.
        switch (offsets.dofType) {
        case DofType::Hinge: {
            // Find the hinge axis orientation in world space
            Vector3 rotated_hinge_axis =
                parent_transform.composedRot.rotateVec(
                        hier.parentToChildRot.rotateVec(hier.axis));

            // Calculate the composed rotation applied to the child entity.
            curr_transform->composedRot = parent_transform.composedRot *
                                    hier.parentToChildRot *
                                    Quat::angleAxis(q[0], hier.axis);

            // Calculate the composed COM position of the child
            //  (parent COM + R_{parent} * (rel_pos_parent + R_{hinge} * rel_pos_local))
            curr_transform->com = parent_transform.com +
                s * parent_transform.composedRot.rotateVec(
                        hier.relPositionParent +
                        hier.parentToChildRot.rotateVec(
                            Quat::angleAxis(q[0], hier.axis).
                                rotateVec(hier.relPositionLocal))
                );

            // All we are getting here is the position of the hinge point
            // which is relative to the parent's COM.
            Vector3 anchor_pos = parent_transform.com +
                s * parent_transform.composedRot.rotateVec(
                        hier.relPositionParent);

            // Phi only depends on the hinge axis and the hinge point
            curr_phi->phi[0] = rotated_hinge_axis[0];
            curr_phi->phi[1] = rotated_hinge_axis[1];
            curr_phi->phi[2] = rotated_hinge_axis[2];
            curr_phi->phi[3] = anchor_pos[0];
            curr_phi->phi[4] = anchor_pos[1];
            curr_phi->phi[5] = anchor_pos[2];
        } break;

        case DofType::Slider: {
            Vector3 rotated_axis =
                parent_transform.composedRot.rotateVec(
                        hier.parentToChildRot.rotateVec(hier.axis));

            // The composed rotation for this body is the same as the parent's
            curr_transform->composedRot = parent_transform.composedRot *
                                    hier.parentToChildRot;

            curr_transform->com = parent_transform.com +
                s * parent_transform.composedRot.rotateVec(
                        hier.relPositionParent +
                        hier.parentToChildRot.rotateVec(
                            hier.relPositionLocal +
                            q[0] * hier.axis)
                );

            // This is the same as the comPos I guess?
            Vector3 axis = rotated_axis.normalize();

            curr_phi->phi[0] = axis[0];
            curr_phi->phi[1] = axis[1];
            curr_phi->phi[2] = axis[2];
        } break;

        case DofType::Ball: {
            Quat joint_rot = Quat{
                q[0], q[1], q[2], q[3]
            };

            // Calculate the composed rotation applied to the child entity.
            curr_transform->composedRot = parent_transform.composedRot *
                                    hier.parentToChildRot *
                                    joint_rot;

            // Calculate the composed COM position of the child
            //  (parent COM + R_{parent} * (rel_pos_parent + R_{ball} * rel_pos_local))
            curr_transform->com = parent_transform.com +
                s * parent_transform.composedRot.rotateVec(
                        hier.relPositionParent +
                        hier.parentToChildRot.rotateVec(
                            joint_rot.rotateVec(hier.relPositionLocal))
                );

            // All we are getting here is the position of the ball point
            // which is relative to the parent's COM.
            Vector3 anchor_pos = parent_transform.com +
                s * parent_transform.composedRot.rotateVec(
                        hier.relPositionParent);

            // Phi only depends on the hinge point and parent rotation
            curr_phi->phi[0] = anchor_pos[0];
            curr_phi->phi[1] = anchor_pos[1];
            curr_phi->phi[2] = anchor_pos[2];
            curr_phi->phi[3] = parent_transform.composedRot.w;
            curr_phi->phi[4] = parent_transform.composedRot.x;
            curr_phi->phi[5] = parent_transform.composedRot.y;
            curr_phi->phi[6] = parent_transform.composedRot.z;
        } break;

        case DofType::FixedBody: {
            curr_transform->composedRot = parent_transform.composedRot;

            // This is the origin of the body
            curr_transform->com =
                parent_transform.com +
                s * parent_transform.composedRot.rotateVec(
                        hier.relPositionParent +
                        hier.parentToChildRot.rotateVec(
                            hier.relPositionLocal)
                );

            // omega remains unchanged, and v only depends on the COM position
            curr_phi->phi[0] = curr_transform->com[0];
            curr_phi->phi[1] = curr_transform->com[1];
            curr_phi->phi[2] = curr_transform->com[2];
        } break;

        default: {
            // Only hinges have parents
            assert(false);
        } break;
        }
    }
}

// Computes the [6 x nDof] Phi matrix which maps from generalized velocities
// to Plücker coordinates
void computePhi(
    DofType dof_type,
    BodyPhi& body_phi,
    float* S,
    Vector3 origin)
{
    if (dof_type == DofType::FreeBody) {
        // S = [1_3x3 r^x; 0 1_3x3], column-major
        memset(S, 0.f, 6 * 6 * sizeof(float));
        // Diagonal identity
        for(CountT i = 0; i < 6; ++i) {
            S[i * 6 + i] = 1.f;
        }
        // r^x Skew symmetric matrix
        Vector3 comPos = {
            body_phi.phi[0],
            body_phi.phi[1], 
            body_phi.phi[2]
        };

        comPos -= origin;
        S[0 + 6 * 4] = -comPos.z;
        S[0 + 6 * 5] = comPos.y;
        S[1 + 6 * 3] = comPos.z;
        S[1 + 6 * 5] = -comPos.x;
        S[2 + 6 * 3] = -comPos.y;
        S[2 + 6 * 4] = comPos.x;
    } else if (dof_type == DofType::Slider) {
        // This is just the axis of the slider.
        S[0] = body_phi.phi[0];
        S[1] = body_phi.phi[1];
        S[2] = body_phi.phi[2];
        S[3] = 0.f;
        S[4] = 0.f;
        S[5] = 0.f;
    } else if (dof_type == DofType::Hinge) {
        // S = [r \times hinge; hinge]
        Vector3 hinge = {
            body_phi.phi[0],
            body_phi.phi[1],
            body_phi.phi[2]
        };

        Vector3 anchorPos = {
            body_phi.phi[3],
            body_phi.phi[4],
            body_phi.phi[5]
        };

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
        Vector3 anchor_pos = {
            body_phi.phi[0], 
            body_phi.phi[1],
            body_phi.phi[2]
        };

        anchor_pos -= origin;

        // We need to right multiply these by the parent composed rotation
        // matrix.
        Mat3x3 rx = Mat3x3::skewSym(anchor_pos);

        Quat parent_composed_rot = Quat {
            body_phi.phi[3],
            body_phi.phi[4],
            body_phi.phi[5],
            body_phi.phi[6]
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
}

// Same as computePhi, but only the first 3 rows (translational component)
void computePhiTrans(DofType dof_type,
                     BodyPhi &body_phi,
                     Vector3 origin,
                     float (&S)[18])
{
    if (dof_type == DofType::FreeBody) {
        memset(S, 0.f, 6 * 3 * sizeof(float));
        for(CountT i = 0; i < 3; ++i) {
            S[i * 3 + i] = 1.f;
        }
        Vector3 comPos = {
            body_phi.phi[0],
            body_phi.phi[1],
            body_phi.phi[2],
        };
        comPos -= origin;
        S[0 + 3 * 4] = -comPos.z;
        S[0 + 3 * 5] = comPos.y;
        S[1 + 3 * 3] = comPos.z;
        S[1 + 3 * 5] = -comPos.x;
        S[2 + 3 * 3] = -comPos.y;
        S[2 + 3 * 4] = comPos.x;
    }
    else if (dof_type == DofType::Slider) {
        S[0] = body_phi.phi[0];
        S[1] = body_phi.phi[1];
        S[2] = body_phi.phi[2];
    }
    else if (dof_type == DofType::Hinge) {
        Vector3 hinge = {
            body_phi.phi[0],
            body_phi.phi[1],
            body_phi.phi[2],
        };
        Vector3 anchorPos = {
            body_phi.phi[3],
            body_phi.phi[4],
            body_phi.phi[5],
        };
        anchorPos -= origin;
        Vector3 r_cross_hinge = anchorPos.cross(hinge);
        S[0] = r_cross_hinge.x;
        S[1] = r_cross_hinge.y;
        S[2] = r_cross_hinge.z;
    }
    else if (dof_type == DofType::Ball) {
        Vector3 anchor_pos = {
            body_phi.phi[0],
            body_phi.phi[1],
            body_phi.phi[2],
        };
        anchor_pos -= origin;
        Mat3x3 rx = Mat3x3::skewSym(anchor_pos);
        Quat parent_composed_rot = Quat{
            body_phi.phi[3],
            body_phi.phi[4],
            body_phi.phi[5],
            body_phi.phi[6]
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

// Computes the center of mass of the body group
void computeGroupCOM(Context &ctx,
                     BodyGroupProperties &prop,
                     BodyGroupMemory &mem)
{
    (void)ctx;

    Vector3 hierarchy_com = Vector3::zero();
    float total_mass = 0.f;

    uint32_t num_bodies = prop.numBodies;
    BodyInertial *inertials = mem.inertials(prop);
    BodyTransform* transforms = mem.bodyTransforms(prop);

    for (uint32_t i = 0; i < num_bodies; ++i) {
        BodyInertial body_inertia = inertials[i];
        BodyTransform body_transform = transforms[i];
        // If infinite mass (likely ground plane)
        if (1.f / body_inertia.mass == 0.f) {
            assert(num_bodies == 1);
            prop.comPos = body_transform.com;
            return;
        }

        hierarchy_com += body_inertia.mass * body_transform.com;
        total_mass += body_inertia.mass;
    }

    prop.comPos = hierarchy_com / total_mass;
}

void computeSpatialInertiasAndPhi(Context &ctx,
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
    Mat3x3 sym_mat = Mat3x3::skewSym(adjustedCom);
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

    uint8_t *max_ptr = (uint8_t *)mem.tmpPtr +
                       BodyGroupMemory::tmpNumBytes(prop);

    if (dof_type != DofType::None && dof_type != DofType::FixedBody) {
        ASSERT_PTR_ACCESS(S, 0, max_ptr);

        computePhi(dof_type, phis[obj_grp.idx], S, com_pos);
    }
}

inline void computePhiDot(DofType dof_type,
                          float (&S)[6 * 6],
                          float (&S_dot)[6 * 6],
                          SpatialVector &v_hat)
{
    if (dof_type == DofType::FreeBody) {
        // S_dot = [0_3x3 v^x; 0_3x3 0_3x3], column-major
        MADRONA_UNROLL
        for (uint32_t i = 0; i < 6 * 6; ++i) {
            S_dot[i] = 0.f;
        }

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
}

// J_C = C^T[e_{b1} S_1, e_{b2} S_2, ...], col-major
//  where e_{bi} = 1 if body i is an ancestor of b
//  C^T projects into the contact space
inline float * computeContactJacobian(BodyGroupProperties &prop,
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

    return J;
}

// y = Mx. Based on Table 6.5 in Featherstone
inline void mulM(
        BodyGroupProperties prop,
        BodyGroupMemory mem,
        float *x, float *y)
{
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
void solveM(
        BodyGroupProperties prop,
        BodyGroupMemory mem, 
        float* x)
{
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


#ifdef MADRONA_GPU_MODE
void compositeRigidBody(
        Context &,
        BodyGroupProperties prop,
        BodyGroupMemory mem)
{
    uint32_t lane_id = threadIdx.x % 32;
    uint32_t warp_id = threadIdx.x / 32;

    // ----------------- Composite rigid body -----------------
    // Mass Matrix of this entire body group, column-major
    uint32_t total_dofs = prop.qvDim;
    BodyOffsets *offsets = mem.offsets(prop);
    BodySpatialVectors *spatialVectors = mem.spatialVectors(prop);

    float *S =  mem.phiFull(prop);
    float *M = mem.massMatrix(prop);

    // memset(M, 0.f, total_dofs * total_dofs * sizeof(float));
    gpu_utils::warpSetZero(M, total_dofs * total_dofs * sizeof(float));
    __syncwarp();

    const int32_t num_smem_bytes_per_warp =
        mwGPU::SharedMemStorage::numBytesPerWarp();
    auto smem_buf = (uint8_t *)mwGPU::SharedMemStorage::buffer +
                    num_smem_bytes_per_warp * warp_id;

    // Backward pass
    for (CountT i = prop.numBodies-1; i >= 0; --i) {
        uint32_t velOffset = offsets[i].velOffset;
        uint32_t S_offset = 2 * 6 * velOffset;
        uint32_t num_dofs = BodyOffsets::getDofTypeDim(offsets[i].dofType);

        float* S_i = S + S_offset;

        // float *F = mem.scratch(prop);
        float *F = (float *)smem_buf;

        if (lane_id == 0) {
            for(CountT col = 0; col < num_dofs; ++col) {
                float *S_col = S_i + 6 * col;
                float *F_col = F + 6 * col;
                spatialVectors[i].spatialInertia.multiply(S_col, F_col);
            }
        }

        __syncwarp();

        // M_{ii} = S_i^T I_i^C S_i = F^T S_i
        float *M_ii = M + velOffset * total_dofs + velOffset;

        gpu_utils::warpLoop(num_dofs * num_dofs,
            [&](uint32_t iter) {
                uint32_t row = iter / num_dofs;
                uint32_t col = iter % num_dofs;

                float *F_col = F + 6 * row;
                float *S_col = S_i + 6 * col;

                #pragma unroll
                for (uint32_t k = 0; k < 6; ++k) {
                    M_ii[row + total_dofs * col] += F_col[k] * S_col[k];
                }
            });

        // Traverse up hierarchy
        uint8_t j = offsets[i].parent;

        while(j != 0xFF) {
            uint32_t velOffset_j = offsets[j].velOffset;
            uint32_t S_offset_j = 2 * 6 * velOffset_j;
            uint32_t j_num_dofs = BodyOffsets::getDofTypeDim(offsets[j].dofType);
            float *S_j = S + S_offset_j;

            // M_{ij} = M{ji} = F^T S_j
            float *M_ij = M + velOffset + total_dofs * velOffset_j; // row i, col j
            float *M_ji = M + velOffset_j + total_dofs * velOffset; // row j, col i


            gpu_utils::warpLoop(num_dofs * j_num_dofs,
                [&](uint32_t iter) {
                    uint32_t row = iter / j_num_dofs;
                    uint32_t col = iter % j_num_dofs;

                    float *F_col = F + 6 * row; // take col for transpose
                    float *S_col = S_j + 6 * col;

                    #pragma unroll
                    for(CountT k = 0; k < 6; ++k) {
                        M_ij[row + total_dofs * col] += F_col[k] * S_col[k];
                        M_ji[col + total_dofs * row] += F_col[k] * S_col[k];
                    }
                });

            j = offsets[j].parent;
        }
    }

    // ----------------- Compute LTDL factorization -----------------
    // First copy M to LTDL
    float *LTDL = mem.massLTDLMatrix(prop);
    gpu_utils::warpCopy(LTDL, M, total_dofs * total_dofs * sizeof(float));
    __syncwarp();

    if (lane_id == 0) {
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
    }

    // ----------------- Sum Inertia Diagonals -----------------
    // Sum the diagonals of the mass matrix (used for solver scaling)
    if (lane_id == 0) {
        float inertiaSum = 0.f;
        for (CountT i = 0; i < prop.qvDim; ++i) {
            inertiaSum += M[i + i * (prop.qvDim)];
        }

        prop.inertiaSum = inertiaSum;
    }

    // ----------------- Compute Free Acceleration -----------------
    float *bias = mem.biasVector(prop);

    gpu_utils::warpLoop(prop.qvDim, [&](uint32_t iter) {
            bias[iter] = -bias[iter];
        });
    __syncwarp();

    if (lane_id == 0) {
        // This overwrites bias with the acceleration
        solveM(prop, mem, bias);
    }
}

#else
// CRB: Compute the Mass Matrix (n_dofs x n_dofs)
void compositeRigidBody(
        Context &,
        BodyGroupProperties prop,
        BodyGroupMemory mem)
{
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
        while(j != 0xFF) {
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
#endif


// Recursive Newton Euler algorithm: Compute bias forces and gravity
inline void rneAndCombineSpatialInertias(
        Context &ctx,
        BodyGroupProperties prop,
        BodyGroupMemory mem)
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

    MADRONA_GPU_SINGLE_THREAD {
        for (uint32_t i = 0; i < prop.numBodies; ++i) {
            BodyOffsets body_offset = offsets[i];
            DofType dof_type = body_offset.dofType;
            BodySpatialVectors& spatial_vector = spatialVectors[i];

            uint32_t num_dofs = BodyOffsets::getDofTypeDim(dof_type);
            uint32_t velOffset = body_offset.velOffset;
            uint32_t S_offset = 2 * 6 * velOffset;
            // uint32_t S_dot_offset = S_offset + 6 * num_dofs;

            float *velocity = qv + velOffset;

            float *S_ptr = mem.phiFull(prop) + S_offset;
            // float* S_dot = mem.phiFull(prop) + S_dot_offset;

            float S[6 * 6];
            float S_dot[6 * 6];

            MADRONA_UNROLL
            for (uint32_t i = 0; i < 6; ++i) {
                if (i < num_dofs) {
                    MADRONA_UNROLL
                    for (uint32_t j = 0; j < 6; ++j) {
                        S[j + i * num_dofs] = S_ptr[j + i * num_dofs];
                    }
                }
            }

            // Common initialization for all joint types
            SpatialVector v_body;
            if (body_offset.parent == 0xFF || dof_type == DofType::FreeBody) {
                // v_0 = 0, a_0 = -g (fictitious upward acceleration)
                spatial_vector.sVel = {Vector3::zero(), Vector3::zero()};
                spatial_vector.sAcc = {-physics_state.g, Vector3::zero()};

                // Free bodies must be root of their hierarchy
                v_body = {velocity[0], velocity[1], velocity[2]};
            } else {
                BodySpatialVectors& parent_spatial_vector = spatialVectors[body_offset.parent];
                spatial_vector.sVel = parent_spatial_vector.sVel;
                spatial_vector.sAcc = parent_spatial_vector.sAcc;

                // Note: we are using the parent velocity here (for hinge itself)
                v_body = parent_spatial_vector.sVel;
            }

            if (dof_type != DofType::FixedBody) {
                computePhiDot(dof_type, S, S_dot, v_body);
            }

            if (dof_type == DofType::FreeBody) {
                // S\dot{q_i} and \dot{S}\dot{q_i}
                MADRONA_UNROLL
                for (uint32_t j = 0; j < 6; ++j) {
                    MADRONA_UNROLL
                    for (uint32_t k = 0; k < 6; ++k) {
                        spatial_vector.sVel[j] += S[j + 6 * k] * velocity[k];
                        spatial_vector.sAcc[j] += S_dot[j + 6 * k] * velocity[k];
                    }
                }
            } else if (dof_type == DofType::FixedBody) {
            } else if (dof_type == DofType::Slider || dof_type == DofType::Hinge) {
                assert(body_offset.parent != 0xFF);

                float q_dot = velocity[0];

                MADRONA_UNROLL
                for (int j = 0; j < 6; ++j) {
                    spatial_vector.sVel[j] += S[j] * q_dot;
                    spatial_vector.sAcc[j] += S_dot[j] * q_dot;
                }
            } else if (dof_type == DofType::Ball) {
                assert(body_offset.parent != 0xFF);

                float *q_dot = velocity;

                MADRONA_UNROLL
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

        // Backward pass to find bias forces (added to external forces)
        float *tau = mem.biasVector(prop);
        memcpy(tau, mem.f(prop), prop.qvDim * sizeof(float));

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
            if (body_offset.parent != 0xFF) {
                BodySpatialVectors& parent_spatial_vector = spatialVectors[body_offset.parent];
                parent_spatial_vector.sForce += spatial_vector.sForce;
            }
        }

        // ----------------- Combine Spatial Inertias -----------------
        uint32_t num_bodies = prop.numBodies;
        // Backward pass from children to parent
        for (CountT i = num_bodies-1; i > 0; --i) {
            InertiaTensor& spatial_inertia = spatialVectors[i].spatialInertia;
            uint32_t parent_idx = (uint32_t)offsets[i].parent;
            InertiaTensor& spatial_inertia_parent = 
                spatialVectors[parent_idx].spatialInertia;
            spatial_inertia_parent += spatial_inertia;
        }
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

    bool ref_static = mem_ref.isStatic(prop_ref)[ref_body_idx];
    bool alt_static = mem_alt.isStatic(prop_alt)[alt_body_idx];
    if (ref_static && alt_static) {
        contact.numPoints = 0;
        return;
    }

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

// This only runs on the CPU
inline void exportCPUSolverState(
        Context &ctx,
        CVSolveData &cv_sing)
{
    uint32_t world_id = ctx.worldID().idx;

    StateManager *state_mgr = getStateManager(ctx);
    float step_h = ctx.singleton<PhysicsSystemState>().h;

    BodyGroupProperties *all_properties = state_mgr->getWorldComponents<
        BodyGroupArchetype, BodyGroupProperties>(world_id);
    BodyGroupMemory *all_memories = state_mgr->getWorldComponents<
        BodyGroupArchetype, BodyGroupMemory>(world_id);
    CountT num_grps = state_mgr->numRows<BodyGroupArchetype>(world_id);

    // Create complete mass matrix
    uint32_t total_num_dofs = 0;
    for (CountT i = 0; i < num_grps; ++i) {
        total_num_dofs += all_properties[i].qvDim;
    }

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

    float *full_vel = (float *)ctx.tmpAlloc(
            num_full_dofs_bytes);
    memset(full_vel, 0, num_full_dofs_bytes);

    uint32_t processed_dofs = 0;

    for (CountT i = 0; i < num_grps; ++i) {
        BodyGroupMemory &m = all_memories[i];
        BodyGroupProperties &p = all_properties[i];

        float *local_mass = m.massMatrix(p);

        for (CountT row = 0; row < p.qvDim; ++row) {
            float *freeAcceleration = m.biasVector(p);
            full_free_acc[row + processed_dofs] = freeAcceleration[row];

            for (CountT col = 0; col < p.qvDim; ++col) {
                uint32_t mi = row + processed_dofs;
                uint32_t mj = col + processed_dofs;

                // The total mass matrix is row major but the
                // local mass matrix is column major.
                total_mass_mat[mj + mi * total_num_dofs] =
                    local_mass[row + all_properties[i].qvDim * col];

            }
        }

        processed_dofs += all_properties[i].qvDim;
    }

    // Full velocity
    processed_dofs = 0;
    for (CountT grp_idx = 0; grp_idx < num_grps; ++grp_idx) {
        BodyGroupProperties p = all_properties[grp_idx];
        BodyGroupMemory m = all_memories[grp_idx];

        BodyOffsets *bg_offsets = m.offsets(p);
        float *bg_qv = m.qv(p);

        for (CountT body_idx = 0; body_idx < p.numBodies; ++body_idx) {
            float *qv = bg_qv + bg_offsets[body_idx].velOffset;
            uint32_t num_dofs = BodyOffsets::getDofTypeDim(
                    bg_offsets[body_idx].dofType);

            for (CountT k = 0; k < num_dofs; ++k) {
                full_vel[processed_dofs] = qv[k];
                processed_dofs++;
            }
        }
    }

    // Create the contact Jacobian
    ContactConstraint *contacts = state_mgr->getWorldComponents<
        Contact, ContactConstraint>(world_id);
    ContactTmpState *contacts_tmp_state = state_mgr->getWorldComponents<
        Contact, ContactTmpState>(world_id);

    CountT num_contacts = state_mgr->numRows<Contact>(world_id);
    CountT total_contact_pts = 0;

    for (int i = 0; i < num_contacts; ++i) {
        total_contact_pts += contacts[i].numPoints;
    }

    // Assuming cone dimension 3. TODO: Rolling friction
    CountT num_full_contact_bytes = sizeof(float) * total_contact_pts * 3;
    float *full_mu = (float *)ctx.tmpAlloc(
            num_full_contact_bytes);
    float *full_penetration = (float *)ctx.tmpAlloc(
            num_full_contact_bytes);
    CountT processed_pts = 0;
    for (int i = 0; i < num_contacts; ++i) {
        ContactTmpState &tmp_state = contacts_tmp_state[i];
        for (int j = 0; j < contacts[i].numPoints; ++j) {
            full_mu[3 * processed_pts] = tmp_state.mu;
            full_mu[3 * processed_pts + 1] = tmp_state.mu;
            full_mu[3 * processed_pts + 2] = tmp_state.mu;
            full_penetration[processed_pts] = contacts[i].points[j].w;
            processed_pts++;
        }
    }

    // Jacobian is size 3n_c x n_dofs, column-major
    uint32_t J_rows = 3 * total_contact_pts;
    uint32_t J_cols = total_num_dofs;

    CountT jac_row = 0;

    uint32_t max_dofs = 0;

    // Prefix sum for each of the body groups
    uint32_t *block_start = (uint32_t *)ctx.tmpAlloc(
            num_grps * sizeof(uint32_t));
    uint32_t block_offset = 0;

    for (CountT i = 0; i < num_grps; ++i) {
        block_start[i] = block_offset;
        block_offset += all_properties[i].qvDim;
        all_properties[i].tmp.grpIndex = i;

        max_dofs = std::max(max_dofs, all_properties[i].qvDim);
    }

    float *J_c = (float *) ctx.tmpAlloc(
        J_rows * J_cols * sizeof(float));
    float *diagApprox_c = (float *) ctx.tmpAlloc(
        J_rows * sizeof(float));

    memset(J_c, 0, J_rows * J_cols * sizeof(float));

    { // Process contacts

        for (CountT ct_idx = 0; ct_idx < num_contacts; ++ct_idx) {
            ContactConstraint contact = contacts[ct_idx];
            ContactTmpState &tmp_state = contacts_tmp_state[ct_idx];

            LinkParentDofObject ref_link = ctx.get<LinkParentDofObject>(contact.ref);
            LinkParentDofObject alt_link = ctx.get<LinkParentDofObject>(contact.alt);

            BodyGroupMemory ref_m = ctx.get<BodyGroupMemory>(ref_link.bodyGroup);
            BodyGroupMemory alt_m = ctx.get<BodyGroupMemory>(alt_link.bodyGroup);

            BodyGroupProperties ref_p = ctx.get<BodyGroupProperties>(
                    ref_link.bodyGroup);
            BodyGroupProperties alt_p = ctx.get<BodyGroupProperties>(
                    alt_link.bodyGroup);

            // Diagonal approximation for contact is based on weight of body
            auto &ref_inertial = ref_m.inertials(ref_p)[ref_link.bodyIdx];
            auto &alt_inertial = alt_m.inertials(alt_p)[alt_link.bodyIdx];

            float inv_weight_trans = ref_inertial.approxInvMassTrans +
                alt_inertial.approxInvMassTrans;

            // Each of the contact points
            for(CountT pt_idx = 0; pt_idx < contact.numPoints; pt_idx++) {
                Vector3 contact_pt = contact.points[pt_idx].xyz();

                // Compute the Jacobians for each body at the contact point
                computeContactJacobian(ref_p,
                    ref_m, ref_link.bodyIdx,  tmp_state.C, contact_pt, J_c,
                    block_start[ref_p.tmp.grpIndex], jac_row, J_rows, -1.f,
                    (ct_idx == 0 && pt_idx == 0));

                computeContactJacobian(alt_p,
                    alt_m, alt_link.bodyIdx, tmp_state.C, contact_pt, J_c,
                    block_start[alt_p.tmp.grpIndex], jac_row, J_rows, 1.f,
                    (ct_idx == 0 && pt_idx == 0));

                // Compute the diagonal approximation
                diagApprox_c[jac_row] = diagApprox_c[jac_row + 1] =
                    diagApprox_c[jac_row + 2] = inv_weight_trans;

                jac_row += 3;
            }
        }

        cv_sing.J_c = J_c;
        cv_sing.diagApprox_c = diagApprox_c;
    }

    { // Process equality constraints
        // This gives us the start in the global array of generalized
        // velocities.
        block_start = (uint32_t *)ctx.tmpAlloc(
                num_grps * sizeof(uint32_t));
        block_offset = 0;
        max_dofs = 0;

        for (CountT i = 0; i < num_grps; ++i) {
            block_start[i] = block_offset;
            block_offset += all_properties[i].qvDim;
            all_properties[i].tmp.grpIndex = i;

            max_dofs = std::max(max_dofs, all_properties[i].qvDim);
        }

        // Starting row in the equality jacobian for each body group
        // Count number of equality constraints that are active
        uint32_t num_active_constraints = 0;
        for (uint32_t grp_idx = 0; grp_idx < num_grps; ++grp_idx) {
            BodyGroupMemory &m = all_memories[grp_idx];
            BodyGroupProperties &p = all_properties[grp_idx];
            if (p.numEq == 0) {
                continue;
            }
            BodyOffsets *curr_offsets = m.offsets(p);

            num_active_constraints += p.numEq;

#if 0
            for (uint32_t body_idx = 0; body_idx < p.numBodies; ++body_idx) {
                BodyOffsets offset = curr_offsets[body_idx];
                float *q = m.q(p) + offset.posOffset;
                if (offset.dofType == DofType::FixedBody) {
                    continue;
                }
                BodyLimitConstraint limit = m.limits(p)[offset.eqOffset];
                if (limit.type == BodyLimitConstraint::Type::None) {
                    continue;
                }

                switch (limit.type) {
                case BodyLimitConstraint::Type::Hinge: {
                    if(limit.hinge.isActive(q[0])) {
                        num_active_constraints++;
                    }
                } break;

                case BodyLimitConstraint::Type::Slider: {
                    if(limit.slider.isActive(q[0])) {
                        num_active_constraints++;
                    }
                } break;

                default: {
                    MADRONA_UNREACHABLE();
                } break;
                }
            }
#endif
        }

        uint32_t total_num_rows = num_active_constraints;

        float *J_e = (float *)ctx.tmpAlloc(
                total_num_rows * total_num_dofs * sizeof(float));
        memset(J_e, 0, total_num_rows * total_num_dofs * sizeof(float));
        auto Je = [&](int32_t row, int32_t col) -> float& {
            return J_e[row + total_num_rows * col];
        };

        float *diagApprox_e = (float *)ctx.tmpAlloc(
                total_num_rows * sizeof(float));

        float *residuals = (float *)ctx.tmpAlloc(
                total_num_rows * sizeof(float));
        memset(residuals, 0, total_num_rows * sizeof(float));

        // This is much easier to do with parallel execution
        num_active_constraints = 0;
        for (uint32_t grp_idx = 0; grp_idx < num_grps; ++grp_idx) {
            BodyGroupMemory &m = all_memories[grp_idx];
            BodyGroupProperties &p = all_properties[grp_idx];
            if (p.numEq == 0) {
                continue;
            }

            BodyOffsets *curr_offsets = m.offsets(p);

            for (uint32_t body_idx = 0; body_idx < p.numBodies; ++body_idx) {
                BodyOffsets offset = curr_offsets[body_idx];
                if (offset.dofType == DofType::FixedBody) {
                    continue;
                }

                float *q = m.q(p) + offset.posOffset;

                BodyLimitConstraint limit = m.limits(p)[offset.eqOffset];
                BodyInertial &inertial = m.inertials(p)[body_idx];

                if (limit.type == BodyLimitConstraint::Type::None) {
                    continue;
                }

                uint32_t glob_row_offset = num_active_constraints;
                uint32_t glob_col_offset = block_start[grp_idx] +
                                           offset.velOffset;

                switch (limit.type) {
                case BodyLimitConstraint::Type::Hinge: {
                    if(limit.hinge.isActive(q[0])) {
                        Je(glob_row_offset, glob_col_offset) =
                            limit.hinge.dConstraintViolation(q[0]);
                        residuals[glob_row_offset] = limit.hinge.constraintViolation(q[0]);
                        diagApprox_e[glob_row_offset] = inertial.approxInvMassDof[0];
                        num_active_constraints++;
                    }
                } break;

                case BodyLimitConstraint::Type::Slider: {
                    if(limit.slider.isActive(q[0])) {
                        Je(glob_row_offset, glob_col_offset) =
                            limit.slider.dConstraintViolation(q[0]);
                        residuals[glob_row_offset] = limit.slider.constraintViolation(q[0]);
                        diagApprox_e[glob_row_offset] = inertial.approxInvMassDof[0];
                        num_active_constraints++;
                    }
                } break;

                default: {
                    MADRONA_UNREACHABLE();
                } break;
                }
            }
        }

        cv_sing.J_e = J_e;
        cv_sing.eqResiduals = residuals;
        cv_sing.diagApprox_e = diagApprox_e;
        cv_sing.numRowsJe = total_num_rows;
        cv_sing.numColsJe = total_num_dofs;
    }

    cv_sing.totalNumDofs = total_num_dofs;
    cv_sing.numContactPts = total_contact_pts;
    cv_sing.h = step_h;

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

    BodyOffsets *offsets = m.offsets(p);
    BodyInertial *inertials = m.inertials(p);
    bool* is_static = m.isStatic(p);
    float total_static_mass = 0.f; // Total mass of all the static bodies
    // First sweep
    for(int32_t i = 0; i < p.numBodies; ++i) {
        uint32_t n_i = offsets[i].numDofs;
        if (i == 0) {
            map[i] = -1;
        } else {
            map[i] = map[i - 1] + (int32_t) n_i;
        }

        // Whether this body is fixed and root (or fixed and parent is fixed root)
        if(map[i] == -1 && offsets[i].dofType == DofType::FixedBody) {
            is_static[i] = true;
            total_static_mass += inertials[i].mass;
        } else {
            is_static[i] = false;
        }
    }
    // Finish expanded parent array
    for(int32_t i = 1; i < p.numBodies; ++i) {
        int32_t parent_idx = (int32_t)(offsets[i].parent == 0xFF ? -1 :
                                       offsets[i].parent);
        expandedParent[map[i - 1] + 1] = map[parent_idx];
        ASSERT_PTR_ACCESS(expandedParent, (map[i - 1] + 1), max_ptr);

        if(is_static[i]) {
            inertials[i].mass = total_static_mass;
        }
    }
}

inline void initHierarchies(Context &ctx,
                            InitBodyGroup body_grp)
{
    BodyGroupMemory &m = ctx.get<BodyGroupMemory>(body_grp.bodyGroup);
    BodyGroupProperties &p = ctx.get<BodyGroupProperties>(body_grp.bodyGroup);

    computeExpandedParent(ctx, m, p);
    forwardKinematics(ctx, m, p);
}

inline void combineSpatialInertias(Context &ctx,
                                   InitBodyGroup body_grp)
{
    auto& m = ctx.get<BodyGroupMemory>(body_grp.bodyGroup);
    auto& p = ctx.get<BodyGroupProperties>(body_grp.bodyGroup);

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

// We are using this for now.
#ifdef MADRONA_GPU_MODE
inline void computeInvMassGPU(
        Context &ctx,
        InitBodyGroup body_grp)
{
    using namespace gpu_utils;

    BodyGroupMemory &m = ctx.get<BodyGroupMemory>(body_grp.bodyGroup);
    BodyGroupProperties &p = ctx.get<BodyGroupProperties>(body_grp.bodyGroup);

    uint32_t lane_id = threadIdx.x % 32;
    uint32_t warp_id = threadIdx.x / 32;
    const int32_t num_smem_bytes_per_warp =
        mwGPU::SharedMemStorage::numBytesPerWarp();
    auto smem_buf = (uint8_t *)mwGPU::SharedMemStorage::buffer +
                    num_smem_bytes_per_warp * warp_id;

    uint32_t num_bytes_per_body = sizeof(float) * 
        (36 + 2 * 6 * p.qvDim);

    uint32_t bodies_in_smem = std::min(
            p.numBodies,
            num_smem_bytes_per_warp / num_bytes_per_body);

    float *gmem_buf = nullptr;
    if (lane_id == 0 && bodies_in_smem < 32) {
        uint32_t to_allocate = std::min(p.numBodies - bodies_in_smem, 
                                        32 - bodies_in_smem);

        if (to_allocate) {
            printf("need to allocate %d bodies in gmem out of %d\n", to_allocate, p.numBodies);
            gmem_buf = (float *)ctx.tmpAlloc(to_allocate * num_bytes_per_body);
        }
    }

    gmem_buf = (float *)__shfl_sync(0xFFFF'FFFF, (uint64_t)gmem_buf, 0);

    float *data_start = nullptr;
    if (lane_id < bodies_in_smem) {
        data_start = (float *)(
            (uint8_t *)smem_buf + 
                       lane_id * num_bytes_per_body); 
    } else if (lane_id < p.numBodies) {
        data_start = (float *)
            ((uint8_t *)gmem_buf +
                        (lane_id - bodies_in_smem) * num_bytes_per_body);
    }

    gmem_buf = (float *)__shfl_sync(0xFFFF'FFFF, (uint64_t)gmem_buf, 0);
    gmem_buf += lane_id * (36 + 36 + 6 * p.qvDim);

    BodyOffsets *offsets = m.offsets(p);
    BodyTransform *transforms = m.bodyTransforms(p);
    BodyInertial *inertials = m.inertials(p);

    warpLoop(
        p.numBodies,
        [&](uint32_t i_body) {
            float *A = data_start;
            float *J = A + 36;
            float *MinvJT = J + 6 * p.qvDim;

            BodyTransform transform = transforms[i_body];
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

            MADRONA_UNROLL
            for (CountT i = 0; i < 6; ++i) {
                for (CountT j = 0; j < p.qvDim; ++j) {
                    MinvJTb(j, i) = Jb(i, j);
                }
            }

            MADRONA_UNROLL
            for (CountT i = 0; i < 6; ++i) {
                float *col = MinvJT + i * p.qvDim;
                solveM( p, m, col);
            }

            // A = J M^{-1} J^T
            memset(A, 0.f, 36 * sizeof(float));

            MADRONA_UNROLL
            for (CountT i = 0; i < 6; ++i) {
                MADRONA_UNROLL
                for (CountT j = 0; j < 6; ++j) {
                    for (CountT k = 0; k < p.qvDim; ++k) {
                        Ab(i, j) += Jb(i, k) * MinvJTb(k, j);
                    }
                }
            }

            // Compute the inverse weight
            float a = inertials[i_body].approxInvMassTrans =
                (Ab(0, 0) + Ab(1, 1) + Ab(2, 2)) / 3.f;
            float b = inertials[i_body].approxInvMassRot =
                (Ab(3, 3) + Ab(4, 4) + Ab(5, 5)) / 3.f;

            printf("(body %d) approx inv mass trans = %f; approx inv mass rot %f\n", i_body, a, b);
        });

    warpLoop(
        p.numBodies,
        [&](uint32_t i_body) {
            float *A = data_start;
            float *J = A + 36;
            float *MinvJT = J + 6 * p.qvDim;

            BodyOffsets offset = offsets[i_body];
            auto &dof_inertial = inertials[i_body];

            uint32_t dof_offset = offset.velOffset;

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
        });
}
#endif

inline void computeInvMass(
        Context &ctx,
        InitBodyGroup body_grp)
{
    BodyGroupMemory &m = ctx.get<BodyGroupMemory>(body_grp.bodyGroup);
    BodyGroupProperties &p = ctx.get<BodyGroupProperties>(body_grp.bodyGroup);

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
    bool *is_static = m.isStatic(p);

    // Compute the inverse weight for each body
    for (CountT i_body = 0; i_body < p.numBodies; ++i_body) {
        BodyInertial &inertial = inertials[i_body];
        if(is_static[i_body]) {
            inertial.approxInvMassTrans = 1 / inertial.mass;
            inertial.approxInvMassRot = 3 / (inertial.inertia.d0 +
                                             inertial.inertia.d1 +
                                             inertial.inertia.d2);
            continue;
        }
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
        float a = inertial.approxInvMassTrans =
            (Ab(0, 0) + Ab(1, 1) + Ab(2, 2)) / 3.f;
        float b = inertial.approxInvMassRot =
            (Ab(3, 3) + Ab(4, 4) + Ab(5, 5)) / 3.f;

        printf("(body %d) approx inv mass trans = %f; approx inv mass rot %f\n", i_body, a, b);
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

inline void integrationStep(Context &ctx,
                            DofObjectGroup grp_info)
{
    float h = ctx.singleton<PhysicsSystemState>().h;
    
    if (!ctx.singleton<CVSolveData>().enablePhysics) {
        return;
    }

    BodyGroupMemory m = ctx.get<BodyGroupMemory>(grp_info.bodyGroup);
    BodyGroupProperties p = ctx.get<BodyGroupProperties>(grp_info.bodyGroup);

    BodyOffsets offset = m.offsets(p)[grp_info.idx];

    float *q = m.q(p) + offset.posOffset;
    float *qv = m.qv(p) + offset.velOffset;
    float *dqv = m.dqv(p) + offset.velOffset;

    if (offset.dofType == DofType::FreeBody) {
        // Symplectic Euler
        for (int i = 0; i < 6; ++i) {
            qv[i] += h * dqv[i];
        }
        for (int i = 0; i < 3; ++i) {
            q[i] += h * qv[i];
        }

        // From angular velocity to quaternion [Q_w, Q_x, Q_y, Q_z]
        Vector3 omega = { qv[3], qv[4], qv[5] };
        Quat rot_quat = { q[3], q[4], q[5], q[6] };
        Quat new_rot = {rot_quat.w, rot_quat.x, rot_quat.y, rot_quat.z};
        new_rot.w += 0.5f * h * (-rot_quat.x * omega.x - 
                                 rot_quat.y * omega.y - 
                                 rot_quat.z * omega.z);

        new_rot.x += 0.5f * h * (rot_quat.w * omega.x + 
                                 rot_quat.z * omega.y -
                                 rot_quat.y * omega.z);
        
        new_rot.y += 0.5f * h * (-rot_quat.z * omega.x + 
                                 rot_quat.w * omega.y +
                                 rot_quat.x * omega.z);

        new_rot.z += 0.5f * h * (rot_quat.y * omega.x - 
                                 rot_quat.x * omega.y +
                                 rot_quat.w * omega.z);

        new_rot = new_rot.normalize();

        q[3] = new_rot.w;
        q[4] = new_rot.x;
        q[5] = new_rot.y;
        q[6] = new_rot.z;
    }
    else if (offset.dofType == DofType::Slider) {
        qv[0] += h * dqv[0];
        q[0] += h * qv[0];
    }
    else if (offset.dofType == DofType::Hinge) {
        qv[0] += h * dqv[0];
        q[0] += h * qv[0];
    }
    else if (offset.dofType == DofType::FixedBody) {
        // Do nothing
    }
    else if (offset.dofType == DofType::Ball) {
        // Symplectic Euler
        for (int i = 0; i < 3; ++i) {
            qv[i] += h * dqv[i];
        }

        // From angular velocity to quaternion [Q_w, Q_x, Q_y, Q_z]
        Vector3 omega = { qv[0], qv[1], qv[2] };
        Quat rot_quat = { q[0], q[1], q[2], q[3] };
        Quat new_rot = { rot_quat.w, rot_quat.x, rot_quat.y, rot_quat.z };
        new_rot.w += 0.5f * h * (-rot_quat.x * omega.x -
                                 rot_quat.y * omega.y -
                                 rot_quat.z * omega.z);

        new_rot.x += 0.5f * h * (rot_quat.w * omega.x +
                                 rot_quat.z * omega.y -
                                 rot_quat.y * omega.z);

        new_rot.y += 0.5f * h * (-rot_quat.z * omega.x + 
                                 rot_quat.w * omega.y +
                                 rot_quat.x * omega.z);

        new_rot.z += 0.5f * h * (rot_quat.y * omega.x -
                                 rot_quat.x * omega.y +
                                 rot_quat.w * omega.z);

        new_rot = new_rot.normalize();

        q[0] = new_rot.w;
        q[1] = new_rot.x;
        q[2] = new_rot.y;
        q[3] = new_rot.z;
    }
    else {
        MADRONA_UNREACHABLE();
    }
}

inline void convertPostSolve(
        Context &ctx,
        Position &position,
        Rotation &rotation,
        Scale &scale,
        LinkParentDofObject &link)
{
    BodyGroupMemory &m = ctx.get<BodyGroupMemory>(link.bodyGroup);
    BodyGroupProperties &p = ctx.get<BodyGroupProperties>(link.bodyGroup);

    BodyTransform transforms = m.bodyTransforms(p)[link.bodyIdx];
    BodyObjectData obj_data = m.objectData(p)[link.objDataIdx];

    scale = obj_data.scale * p.globalScale;
    position = transforms.com + p.globalScale *
                    transforms.composedRot.rotateVec(obj_data.offset);
    rotation = transforms.composedRot * obj_data.rotation;

#if 0
    if (link.type == LinkParentDofObject::Type::Render) {
        printf("pos = %f %f %f; rot = %f %f %f %f; scale = %f %f %f\n",
                position.x, position.y, position.z,
                rotation.w, rotation.x, rotation.y, rotation.z,
                scale.d0, scale.d1, scale.d2);
    }
#endif
}
}

TaskGraphNodeID setupPrepareTasks(TaskGraphBuilder &builder,
                                  TaskGraphNodeID narrowphase)
{
    // Initialize memory and run forward kinematics
    auto cur_node = builder.addToGraph<ParallelForNode<Context,
         tasks::initHierarchies,
            InitBodyGroup
         >>({narrowphase});

    cur_node = builder.addToGraph<ParallelForNode<Context,
         tasks::computeGroupCOM,
            BodyGroupProperties,
            BodyGroupMemory
        >>({cur_node});

    cur_node = builder.addToGraph<ParallelForNode<Context,
         tasks::computeSpatialInertiasAndPhi,
            DofObjectGroup
        >>({cur_node});

#ifdef MADRONA_GPU_MODE
    cur_node = builder.addToGraph<CustomParallelForNode<Context,
         tasks::rneAndCombineSpatialInertias, 32, 1,
            BodyGroupProperties,
            BodyGroupMemory
        >>({cur_node});
#else
    cur_node = builder.addToGraph<ParallelForNode<Context,
         tasks::rneAndCombineSpatialInertias,
            BodyGroupProperties,
            BodyGroupMemory
        >>({cur_node});
#endif

    // Only run combine spatial inertias on initialization, skipping RNE
    cur_node = builder.addToGraph<ParallelForNode<Context,
         tasks::combineSpatialInertias,
            InitBodyGroup
        >>({cur_node});

#ifdef MADRONA_GPU_MODE
    cur_node = builder.addToGraph<CustomParallelForNode<Context,
         tasks::compositeRigidBody, 32, 1,
            BodyGroupProperties,
            BodyGroupMemory
        >>({cur_node});
#else
    cur_node = builder.addToGraph<ParallelForNode<Context,
         tasks::compositeRigidBody,
            BodyGroupProperties,
            BodyGroupMemory
        >>({cur_node});
#endif

    // Only compute inv mass on initialization
#ifdef MADRONA_GPU_MODE
    cur_node = builder.addToGraph<CustomParallelForNode<Context,
         tasks::computeInvMassGPU, 32, 1,
         InitBodyGroup
     >>({cur_node});
#else
    cur_node = builder.addToGraph<ParallelForNode<Context,
         tasks::computeInvMass,
         InitBodyGroup
     >>({cur_node});
#endif

    cur_node = builder.addToGraph<ParallelForNode<Context,
         tasks::processContacts,
            ContactConstraint,
            ContactTmpState
        >>({cur_node});

    // Only run contact processing on initialization
    cur_node = builder.addToGraph<ParallelForNode<Context, 
        tasks::convertPostSolve,
            Position,
            Rotation,
            Scale,
            LinkParentDofObject
        >>({cur_node});

    cur_node = builder.addToGraph<
        ClearTmpNode<InitBodyGroupArchetype>>({cur_node});

#ifndef MADRONA_GPU_MODE
    cur_node = builder.addToGraph<ParallelForNode<Context,
         tasks::exportCPUSolverState,
            CVSolveData
        >>({cur_node});
#endif

    return cur_node;
}

TaskGraphNodeID setupCVInitTasks(
        TaskGraphBuilder &builder,
        Span<const TaskGraphNodeID> deps)
{
    auto node =
        builder.addToGraph<ParallelForNode<Context, tasks::convertPostSolve,
            Position,
            Rotation,
            Scale,
            LinkParentDofObject
        >>(deps);

    return node;
}

TaskGraphNodeID setupPostTasks(TaskGraphBuilder &builder,
                               TaskGraphNodeID solve)
{
    auto cur_node = builder.addToGraph<ParallelForNode<Context,
         tasks::integrationStep,
            DofObjectGroup
        >>({solve});

    cur_node = builder.addToGraph<ParallelForNode<Context,
         tasks::forwardKinematics,
            BodyGroupMemory,
            BodyGroupProperties
        >>({cur_node});

    cur_node =
        builder.addToGraph<ParallelForNode<Context, tasks::convertPostSolve,
            Position,
            Rotation,
            Scale,
            LinkParentDofObject
        >>({cur_node});

    return cur_node;
}

}
