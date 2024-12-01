#include <madrona/state.hpp>
#include <madrona/physics.hpp>
#include <madrona/context.hpp>
#include <madrona/cvphysics.hpp>
#include <madrona/taskgraph.hpp>

#include "physics_impl.hpp"

using namespace madrona::math;
using namespace madrona::base;

namespace madrona::phys::cv {

struct Contact : Archetype<
    ContactConstraint,
    ContactTmpState
> {};

struct Joint : Archetype<
    JointConstraint
> {};

struct CVRigidBodyState : Bundle<
    CVPhysicalComponent
> {};

struct CVHierarchyCounter
{
   uint32_t num_bodies;
};

struct CVSingleton {
    CVXSolve *cvxSolve;
};

namespace tasks {

// We are going to need a different way of solving on GPU mode
#ifdef MADRONA_GPU_MODE
struct GaussMinimizationNode : NodeBase {
    GaussMinimizationNode(StateManager *state_mgr);

    void solve(int32_t invocation_idx);

    static TaskGraph::NodeID addToGraph(
            TaskGraph::Builder &builder,
            Span<const TaskGraph::NodeID> dependencies);

    DofObjectPosition *positions;
    DofObjectVelocity *velocities;
    DofObjectNumDofs *numDofs;
    ObjectID *objIDs;

    // This is generated from narrowphase.
    ContactConstraint *contacts;

    // World offsets of the contact constraints.
    int32_t *contactWorldOffsets;

    // World offsets of the positions / velocities.
    int32_t *dofObjectWorldOffsets;
    int32_t *dofObjectWorldCounts;
};

GaussMinimizationNode::GaussMinimizationNode(
        StateManager *s)
    : positions(s->getArchetypeComponent<DofObjectArchetype,
            DofObjectPosition>()),
      velocities(s->getArchetypeComponent<DofObjectArchetype,
            DofObjectVelocity>()),
      numDofs(s->getArchetypeComponent<DofObjectArchetype,
            DofObjectNumDofs>()),
      objIDs(s->getArchetypeComponent<DofObjectArchetype,
            ObjectID>()),
      contacts(s->getArchetypeComponent<Contact,
            ContactConstraint>()),
      contactWorldOffsets(s->getArchetypeWorldOffsets<Contact>()),
      dofObjectWorldOffsets(s->getArchetypeWorldOffsets<DofObjectArchetype>()),
      dofObjectWorldCounts(s->getArchetypeWorldCounts<DofObjectArchetype>())
{
}

void GaussMinimizationNode::solve(int32_t invocation_idx)
{
    uint32_t total_resident_warps = (blockDim.x * gridDim.x) / 32;

    uint32_t total_num_worlds = mwGPU::GPUImplConsts::get().numWorlds;
    uint32_t world_idx = (blockDim.x * blockIdx.x + threadIdx.x) / 32;

    uint32_t warp_id = threadIdx.x / 32;
    uint32_t lane_id = threadIdx.x % 32;



    const int32_t num_smem_bytes_per_warp =
        mwGPU::SharedMemStorage::numBytesPerWarp();
    auto smem_buf = (uint8_t *)mwGPU::SharedMemStorage::buffer +
                    num_smem_bytes_per_warp * warp_id;

    enum PhaseSMemBytes {

    };

    // TODO: do the magic here!
    // Note, the refs that are in the ContactConstraint are not to the
    // DOF object but to the entity which holds the position / rotation
    // components. This just means we have to inefficiently go through another
    // indirection to get the Dof positions / velocities of affected objects
    // (with that indirection being CVPhysicalComponent).
    // We can fix this by having the contact generation directly output
    // the locs of the DOF objects instead but that we can do in the future.
    assert(blockDim.x == consts::numMegakernelThreads);

    while (world_idx < total_num_worlds) {
        { // Compute data for mass matrix
            ObjectManager &state_mgr =
                mwGPU::getStateManager()->getSingleton<ObjectData>(
                        world_idx).mgr;

            uint32_t num_phys_objs = dofObjectWorldCounts[world_idx];
            uint32_t phys_objs_offset = dofObjectWorldOffsets[world_idx];

            uint32_t current_phys_obj = lane_id;

            uint32_t per_world_alloc = num_phys_objs *
                ((3 * 3 * 4) + // Inverse generalized mass rotation components
                 // TODO
                 0);

            // TODO: Possibly merge this over threadblock to avoid
            // device <-> cpu communication.
            void *per_world_bytes =
                mwGPU::TmpAllocator::get().alloc(per_world_alloc);

            while (current_phys_obj < num_phys_objs) {
                DofObjectPosition gen_pos = positions[current_phys_obj +
                                                      phys_objs_offset];
                ObjectID obj_id = objIDs[current_phys_obj +
                                         phys_objs_offset];

                const RigidBodyMetadata &metadata =
                    obj_mgr.metadata[obj_id.idx];

                Vector3 inv_inertia = metadata.invInertiaTensor;

                Quat rot_quat = {
                    gen_pos.q[3],
                    gen_pos.q[4],
                    gen_pos.q[5],
                    gen_pos.q[6],
                };

                Mat3x3 body_jacob = Mat3x3::fromQuat(rot_quat);
                Diag3x3 inv_inertia = Diag3x3::fromVec(
                    inv_inertia);

                Mat3x3 inv_rot_mass = body_jacob * inv_inertia;
                inv_rot_mass = body_jacob * inv_rot_mass.transpose();

                *((Mat3x3 *)per_world_bytes + current_phys_obj) = inv_rot_mass;

                current_phys_obj += 32;
            }

            __syncwarp();
        }


        world_idx += total_resident_warps;
    }
}

TaskGraph::NodeID GaussMinimizationNode::addToGraph(
        TaskGraph::Builder &builder,
        Span<const TaskGraph::NodeID> deps)
{
    using namespace mwGPU;

    // First, we must sort the physics DOF object archetypes by world.
    auto sort_sys =
        builder.addToGraph<SortArchetypeNode<
            DofObjectArchetype, WorldID>>(
            deps);
    auto post_sort_reset_tmp =
        builder.addToGraph<ResetTmpAllocNode>({sort_sys});

    StateManager *state_mgr = getStateManager();

    auto data_id = builder.constructNodeData<GaussMinimizationNode>(
            state_mgr);
    auto &gauss_data = builder.getDataRef(data_id);

    // For now, we are going with the persistent threads approach where each
    // thread block is going to process a world.
    uint32_t num_invocations = (uint32_t)gridDim.x;

    TaskGraph::NodeID solve_node = builder.addNodeFn<
        &GaussMinimizationNode::solve>(data_id, { post_sort_reset_tmp },
                Optional<TaskGraph::NodeID>::none(),
                num_invocations,
                // This is the thread block dimension
                consts::numMegakernelThreads);

    return solve_node;
}
#else
static Mat3x3 skewSymmetricMatrix(Vector3 v)
{
    return {
        {
            { 0.f, v.z, -v.y },
            { -v.z, 0.f, v.x },
            { v.y, -v.x, 0.f }
        }
    };
}

static void forwardKinematics(Context &ctx,
                              BodyGroupHierarchy &body_grp)
{
    { // Set the parent's state
        auto &position = ctx.get<DofObjectPosition>(body_grp.bodies[0]);
        auto &tmp_state = ctx.get<DofObjectTmpState>(body_grp.bodies[0]);

        tmp_state.comPos = {
            position.q[0],
            position.q[1],
            position.q[2]
        };

        tmp_state.composedRot = {
            position.q[3],
            position.q[4],
            position.q[5],
            position.q[6]
        };

        // omega remains unchanged, and v only depends on the COM position
        tmp_state.phi.v[0] = tmp_state.comPos[0];
        tmp_state.phi.v[1] = tmp_state.comPos[1];
        tmp_state.phi.v[2] = tmp_state.comPos[2];
    }

    // Forward pass from parent to children
    for (int i = 1; i < body_grp.numBodies; ++i) {
        auto &position = ctx.get<DofObjectPosition>(body_grp.bodies[i]);
        auto &num_dofs = ctx.get<DofObjectNumDofs>(body_grp.bodies[i]);
        auto &tmp_state = ctx.get<DofObjectTmpState>(body_grp.bodies[i]);
        auto &hier_desc = ctx.get<DofObjectHierarchyDesc>(body_grp.bodies[i]);

        Entity parent_e = hier_desc.parent;
        DofObjectTmpState &parent_tmp_state =
            ctx.get<DofObjectTmpState>(parent_e);

        // We can calculate our stuff.
        switch (num_dofs.numDofs) {
        case (uint32_t)DofType::Hinge: {
            // Find the hinge axis orientation in world space
            Vector3 rotated_hinge_axis =
                parent_tmp_state.composedRot.rotateVec(hier_desc.hingeAxis);

            // Calculate the composed rotation applied to the child entity.
            tmp_state.composedRot = parent_tmp_state.composedRot *
                Quat::angleAxis(position.q[0], hier_desc.hingeAxis);

            // Calculate the composed COM position of the child
            //  (parent COM + R_{parent} * (rel_pos_parent + R_{hinge} * rel_pos_local))
            tmp_state.comPos = parent_tmp_state.comPos +
                parent_tmp_state.composedRot.rotateVec(
                        hier_desc.relPositionParent +
                        Quat::angleAxis(position.q[0], hier_desc.hingeAxis).
                            rotateVec(hier_desc.relPositionLocal)
                );

            // All we are getting here is the position of the hinge point
            // which is relative to the parent's COM.
            tmp_state.anchorPos = parent_tmp_state.comPos +
                parent_tmp_state.composedRot.rotateVec(
                        hier_desc.relPositionParent);

            // Phi only depends on the hinge axis and the hinge point
            tmp_state.phi.v[0] = rotated_hinge_axis[0];
            tmp_state.phi.v[1] = rotated_hinge_axis[1];
            tmp_state.phi.v[2] = rotated_hinge_axis[2];
            tmp_state.phi.v[3] = tmp_state.anchorPos[0];
            tmp_state.phi.v[4] = tmp_state.anchorPos[1];
            tmp_state.phi.v[5] = tmp_state.anchorPos[2];

        } break;

        case (uint32_t)DofType::Ball: {
            Quat joint_rot = Quat{
                position.q[0], position.q[1], position.q[2], position.q[3] 
            };

            // Calculate the composed rotation applied to the child entity.
            tmp_state.composedRot = parent_tmp_state.composedRot * joint_rot;

            // Calculate the composed COM position of the child
            //  (parent COM + R_{parent} * (rel_pos_parent + R_{ball} * rel_pos_local))
            tmp_state.comPos = parent_tmp_state.comPos +
                parent_tmp_state.composedRot.rotateVec(
                        hier_desc.relPositionParent +
                        joint_rot.rotateVec(hier_desc.relPositionLocal)
                );

            // All we are getting here is the position of the ball point
            // which is relative to the parent's COM.
            tmp_state.anchorPos = parent_tmp_state.comPos +
                parent_tmp_state.composedRot.rotateVec(
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

        default: {
            // Only hinges have parents
            assert(false);
        } break;
        }
    }
}

static void computeCenterOfMass(Context &ctx,
                                BodyGroupHierarchy &body_grp) {

    Vector3 hierarchy_com = Vector3::zero();
    float total_mass = 0.f;
    for (int i = 0; i < body_grp.numBodies; ++i) {
        DofObjectTmpState &tmp_state = ctx.get<DofObjectTmpState>(
                body_grp.bodies[i]);
        hierarchy_com += tmp_state.spatialInertia.mass * tmp_state.comPos;
        total_mass += tmp_state.spatialInertia.mass;
    }
    body_grp.comPos = hierarchy_com / total_mass;
}

// Pre-CRB: compute the spatial inertia in common Plücker coordinates
static void computeSpatialInertia(Context &ctx,
                                  BodyGroupHierarchy &body_grp) {
    ObjectManager &obj_mgr = *ctx.singleton<ObjectData>().mgr;

    for(int i = 0; i < body_grp.numBodies; i++)
    {
        DofObjectNumDofs num_dofs = ctx.get<DofObjectNumDofs>(body_grp.bodies[i]);
        if(num_dofs.numDofs == (uint32_t)DofType::FixedBody) {
            return;
        }

        ObjectID obj_id = ctx.get<ObjectID>(body_grp.bodies[i]);
        DofObjectTmpState &tmp_state = ctx.get<DofObjectTmpState>(body_grp.bodies[i]);
        RigidBodyMetadata metadata = obj_mgr.metadata[obj_id.idx];
        Diag3x3 inertia = Diag3x3::fromVec(metadata.mass.invInertiaTensor).inv();
        float mass = 1.f / metadata.mass.invMass;

        // We need to find inertia tensor in world space orientation
        Mat3x3 rot_mat = Mat3x3::fromQuat(tmp_state.composedRot);
        // I_world = R * I * R^T (since R^T transforms from world to local)
        Mat3x3 i_world_frame = rot_mat * inertia * rot_mat.transpose();

        // Compute the 3x3 skew-symmetric matrix (r^x) (where r is from Plücker origin to COM)
        Vector3 adjustedCom = tmp_state.comPos - body_grp.comPos;
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
}

// Compute spatial inertia of subtree rooted at the body.
static void combineSpatialInertias(Context &ctx,
                                   BodyGroupHierarchy &body_grp)
{
    // Backward pass from children to parent
    for (CountT i = body_grp.numBodies-1; i > 0; --i) {
        auto &current_hier_desc = ctx.get<DofObjectHierarchyDesc>(
                body_grp.bodies[i]);
        auto &current_tmp_state = ctx.get<DofObjectTmpState>(
                body_grp.bodies[i]);
        auto &parent_tmp_state = ctx.get<DofObjectTmpState>(
                body_grp.bodies[current_hier_desc.parentIndex]);
        parent_tmp_state.spatialInertia += current_tmp_state.spatialInertia;
    }
}

// Fully computes the Phi matrix from generalized velocities to Plücker coordinates
static float* computePhi(Context &ctx,
                         DofObjectNumDofs &num_dofs,
                         Vector3 origin,
                         Phi &phi)
{
    uint32_t world_id = ctx.worldID().idx;
    StateManager *state_mgr = ctx.getStateManager();
    float *S;

    if (num_dofs.numDofs == (uint32_t)DofType::FreeBody) {
        // S = [1_3x3 r^x; 0 1_3x3], column-major
        S = (float *) state_mgr->tmpAlloc(world_id,
            6 * 6 * sizeof(float));
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
    }
    else if (num_dofs.numDofs == (uint32_t)DofType::Hinge) {
        // S = [r \times hinge; hinge]
        S = (float *) state_mgr->tmpAlloc(world_id,
            6 * sizeof(float));
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
    }
    else if (num_dofs.numDofs == (uint32_t)DofType::Ball) {
        // This will just get right-multiplied by the angular velocity
        S = (float *)state_mgr->tmpAlloc(world_id,
            6 * 3 * sizeof(float));
        memset(S, 0, 6 * 3 * sizeof(float));

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
    }
    else {
        MADRONA_UNREACHABLE();
    }
    return S;
}

// Computes only the first three rows of Phi (for the translational part)
static float* computePhiTrans(Context &ctx,
                              DofObjectNumDofs &num_dofs,
                              Vector3 origin,
                              Phi &phi)
{
    uint32_t world_id = ctx.worldID().idx;
    StateManager *state_mgr = ctx.getStateManager();
    float *S;

    if (num_dofs.numDofs == (uint32_t)DofType::FreeBody) {
        // S = [1_3x3 r^x], column-major
        S = (float *) state_mgr->tmpAlloc(world_id,
            3 * 6 * sizeof(float));
        memset(S, 0.f, 3 * 6 * sizeof(float));
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
    else if (num_dofs.numDofs == (uint32_t)DofType::Hinge) {
        // S = [r \times hinge]
        S = (float *) state_mgr->tmpAlloc(world_id,
            3 * sizeof(float));
        Vector3 hinge = {phi.v[0], phi.v[1], phi.v[2]};
        Vector3 anchorPos = {phi.v[3], phi.v[4], phi.v[5]};
        anchorPos -= origin;
        Vector3 r_cross_hinge = anchorPos.cross(hinge);
        S[0] = r_cross_hinge.x;
        S[1] = r_cross_hinge.y;
        S[2] = r_cross_hinge.z;
    }
    else if (num_dofs.numDofs == (uint32_t)DofType::Ball) {
        S = (float *)state_mgr->tmpAlloc(world_id,
            3 * 3 * sizeof(float));

        Vector3 anchor_pos = {phi.v[0], phi.v[1], phi.v[2]};
        anchor_pos -= origin;

        // Anchor position in the coordinate space, and parent rotation
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
        MADRONA_UNREACHABLE();
    }
    return S;
}

static float* computePhiDot(Context &ctx,
                            DofObjectNumDofs &num_dofs,
                            SpatialVector &v_hat,
                            Vector3 origin,
                            Phi &phi)
{
    uint32_t world_id = ctx.worldID().idx;
    StateManager *state_mgr = ctx.getStateManager();
    float *S_dot;

    if (num_dofs.numDofs == (uint32_t)DofType::FreeBody) {
        // S = [0_3x3 v^x; 0_3x3 0_3x3], column-major
        S_dot = (float *) state_mgr->tmpAlloc(world_id,
            6 * 6 * sizeof(float));
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
    else if (num_dofs.numDofs == (uint32_t)DofType::Hinge) {
        S_dot = (float *) state_mgr->tmpAlloc(world_id,
            6 * sizeof(float));
        // S = [r \times hinge; hinge]
        float* S = computePhi(ctx, num_dofs, origin, phi);
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
    else if (num_dofs.numDofs == (uint32_t)DofType::Ball) {
        S_dot = (float *)state_mgr->tmpAlloc(world_id,
            6 * 3 * sizeof(float));
        memset(S_dot, 0.f, 6 * 3 * sizeof(float));

        float* S = computePhi(ctx, num_dofs, origin, phi);
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
    else {
        MADRONA_UNREACHABLE();
    }
    return S_dot;
}

static float* computeContactJacobian(Context &ctx,
                                     DofObjectNumDofs &num_dofs,
                                     BodyGroupHierarchy &body_grp,
                                     DofObjectHierarchyDesc &hier_desc,
                                     Mat3x3 &C,
                                     Vector3 &origin,
                                     Phi &phi)
{
    uint32_t world_id = ctx.worldID().idx;
    StateManager *state_mgr = ctx.getStateManager();
    // J_C = C^T[e_{b1} S_1, e_{b2} S_2, ...], col-major
    //  where e_{bi} = 1 if body i is an ancestor of b
    //  C^T projects into the contact space
    float *J = (float *) state_mgr->tmpAlloc(world_id,
        3 * body_grp.numDofs * sizeof(float));
    memset(J, 0.f, 3 * body_grp.numDofs * sizeof(float));

    // Compute prefix sum to determine the start of the block for each body
    uint32_t block_start[body_grp.numBodies];
    uint32_t block_offset = 0;
    for (CountT i = 0; i < body_grp.numBodies; ++i) {
        block_start[i] = block_offset;
        block_offset += ctx.get<DofObjectNumDofs>(
                body_grp.bodies[i]).numDofs;
    }

    // Populate J_C by traversing up the hierarchy
    int32_t curr_idx = hier_desc.index;
    while(curr_idx != -1) {
        auto &curr_tmp_state = ctx.get<DofObjectTmpState>(
                body_grp.bodies[curr_idx]);
        auto &curr_num_dofs = ctx.get<DofObjectNumDofs>(
                body_grp.bodies[curr_idx]);
        auto &curr_hier_desc = ctx.get<DofObjectHierarchyDesc>(
                body_grp.bodies[curr_idx]);

        // Populate columns of J_C
        float *S_trans = computePhiTrans(
                ctx, curr_num_dofs, origin, curr_tmp_state.phi);
        for(CountT i = 0; i < curr_num_dofs.numDofs; ++i) {
            float *J_col = J + 3 * (block_start[curr_idx] + i);
            float *S_col = S_trans + 3 * i;
            for(CountT j = 0; j < 3; ++j) {
                J_col[j] = S_col[j];
            }
        }
        curr_idx = curr_hier_desc.parentIndex;
    }

    // Multiply by C^T to project into contact space
    for(CountT i = 0; i < body_grp.numDofs; ++i) {
        float *J_col = J + 3 * i;
        Vector3 J_col_vec = { J_col[0], J_col[1], J_col[2] };
        J_col_vec = C.transpose() * J_col_vec;
        J_col[0] = J_col_vec.x;
        J_col[1] = J_col_vec.y;
        J_col[2] = J_col_vec.z;
    }
    return J;
}


// Solves M^{-1}x, overwriting x. Based on Table 6.5 in Featherstone
static void solveM(BodyGroupHierarchy &body_grp, float *x) {
    CountT total_dofs = body_grp.numDofs;
    auto ltdl = [&](int32_t row, int32_t col) -> float& {
        return body_grp.massMatrixLTDL[row + total_dofs * col];
    };
    // M=L^TDL, so first solve L^{-T} x
    for (int32_t i = (int32_t) total_dofs - 1; i >= 0; --i) {
        int32_t j = body_grp.expandedParent[i];
        while (j != -1) {
            x[j] -= ltdl(i, j) * x[i];
            j = body_grp.expandedParent[j];
        }
    }
    // D^{-1} x
    for (int32_t i = 0; i < total_dofs; ++i) {
        x[i] /= ltdl(i, i);
    }
    // L^{-1} x
    for (int32_t i = 0; i < total_dofs; ++i) {
        int32_t j = body_grp.expandedParent[i];
        while (j != -1) {
            x[i] -= ltdl(i, j) * x[j];
            j = body_grp.expandedParent[j];
        }
    }
}


// CRB: Compute the Mass Matrix (n_dofs x n_dofs)
static void compositeRigidBody(Context &ctx,
                               BodyGroupHierarchy &body_grp)
{
    uint32_t world_id = ctx.worldID().idx;
    StateManager *state_mgr = ctx.getStateManager();

    // Mass Matrix of this entire body group, column-major
    uint32_t total_dofs = body_grp.numDofs;
    float *M = (float *) state_mgr->tmpAlloc(world_id,
        total_dofs * total_dofs * sizeof(float));
    memset(M, 0.f, total_dofs * total_dofs * sizeof(float));

    // Compute prefix sum to determine the start of the block for each body
    uint32_t block_start[body_grp.numBodies];
    uint32_t block_offset = 0;
    for (CountT i = 0; i < body_grp.numBodies; ++i) {
        block_start[i] = block_offset;
        block_offset += ctx.get<DofObjectNumDofs>(
                body_grp.bodies[i]).numDofs;
    }

    // Backward pass
    for (CountT i = body_grp.numBodies-1; i >= 0; --i) {
        auto &i_hier_desc = ctx.get<DofObjectHierarchyDesc>(
                body_grp.bodies[i]);
        auto &i_tmp_state = ctx.get<DofObjectTmpState>(
                body_grp.bodies[i]);
        auto &i_num_dofs = ctx.get<DofObjectNumDofs>(
                body_grp.bodies[i]);
        float *S_i = computePhi(
                ctx, i_num_dofs, body_grp.comPos, i_tmp_state.phi);

        // Temporary store for F = I_i^C S_i, column-major
        float *F = (float *) state_mgr->tmpAlloc(world_id,
            6 * i_num_dofs.numDofs * sizeof(float));
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
                body_grp.bodies[j]).parentIndex;
            auto &j_tmp_state = ctx.get<DofObjectTmpState>(
                body_grp.bodies[j]);
            auto &j_num_dofs = ctx.get<DofObjectNumDofs>(
                body_grp.bodies[j]);

            float *S_j = computePhi(
                    ctx, j_num_dofs, body_grp.comPos, j_tmp_state.phi);

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
                body_grp.bodies[j]).parent;
        }
    }

    body_grp.massMatrix = M;
}

// Computes the expanded parent array (based on Table 6.4 of Featherstone)
// TODO: this only needs to be done once (probably after construction)
static void computeExpandedParent(Context &ctx,
                                  BodyGroupHierarchy &body_grp) {
    CountT N_B = body_grp.numBodies;
    // Initialize n-N_B elements
    for(int32_t i = 0; i < body_grp.numDofs; ++i) {
        body_grp.expandedParent[i] = i - 1;
    }
    // Create a mapping from body index to start of block
    int32_t map[N_B];
    map[0] = -1;
    for(int32_t i = 1; i < N_B; ++i) {
        uint32_t n_i = ctx.get<DofObjectNumDofs>(body_grp.bodies[i]).numDofs;
        map[i] = map[i - 1] + (int32_t) n_i;
    }
    // Finish expanded parent array
    for(int32_t i = 1; i < N_B; ++i) {
        int32_t parent_idx = ctx.get<DofObjectHierarchyDesc>(
            body_grp.bodies[i]).parentIndex;
        body_grp.expandedParent[map[i - 1] + 1] = map[parent_idx];
    }
}

// Computes the LTDL factorization of the mass matrix
static void factorizeMassMatrix(Context &ctx,
                                BodyGroupHierarchy &body_grp) {
    uint32_t world_id = ctx.worldID().idx;
    StateManager *state_mgr = ctx.getStateManager();

    // First copy in the mass matrix
    uint32_t total_dofs = body_grp.numDofs;
    float *LTDL = (float *) state_mgr->tmpAlloc(world_id,
        total_dofs * total_dofs * sizeof(float));
    memcpy(LTDL, body_grp.massMatrix,
        total_dofs * total_dofs * sizeof(float));
    // Helper
    auto ltdl = [&](int32_t row, int32_t col) -> float& {
        return LTDL[row + total_dofs * col];
    };

    // Backward pass through DOFs
    for (int32_t k = (int32_t) total_dofs - 1; k >= 0; --k) {
        int32_t i = body_grp.expandedParent[k];
        while (i != -1) {
            // Temporary storage
            float a = ltdl(k, i) / ltdl(k, k);
            int32_t j = i;
            while (j != -1) {
                ltdl(i, j) = ltdl(i, j) - a * ltdl(k, j);
                j = body_grp.expandedParent[j];
            }
            ltdl(k, i) = a;
            i = body_grp.expandedParent[i];
        }
    }
    body_grp.massMatrixLTDL = LTDL;
}

// Computes the unconstrained acceleration of each body
static void computeFreeAcceleration(Context &ctx,
                                    BodyGroupHierarchy &body_grp) {
    StateManager *state_mgr = ctx.getStateManager();
    uint32_t world_id = ctx.worldID().idx;
    // Copy in the bias forces
    CountT num_dofs = body_grp.numDofs;
    float *freeAcc = (float *) state_mgr->tmpAlloc(world_id,
        num_dofs * sizeof(float));
    memcpy(freeAcc, body_grp.biasForces, num_dofs * sizeof(float));
    // Negate the bias forces, solve
    for (CountT i = 0; i < num_dofs; ++i) {
        freeAcc[i] = -freeAcc[i];
    }
    solveM(body_grp, freeAcc);
    body_grp.freeAcceleration = freeAcc;
}

// RNE: Compute bias forces and gravity
static void recursiveNewtonEuler(Context &ctx,
                                BodyGroupHierarchy &body_grp)
{
    PhysicsSystemState &physics_state = ctx.singleton<PhysicsSystemState>();
    StateManager *state_mgr = ctx.getStateManager();

    uint32_t world_id = ctx.worldID().idx;
    uint32_t total_dofs = body_grp.numDofs;
    float *tau = (float *) state_mgr->tmpAlloc(world_id,
        total_dofs * sizeof(float));
    memset(tau, 0.f, total_dofs * sizeof(float));

    // Forward pass. Find in Plücker coordinates:
    //  1. velocities. v_i = v_{parent} + S * \dot{q_i}
    //  2. accelerations. a_i = a_{parent} + \dot{S} * \dot{q_i} + S * \ddot{q_i}
    //  3. forces. f_i = I_i a_i + v_i [spatial star cross] I_i v_i

    // First handle root of hierarchy
    {
        auto num_dofs = ctx.get<DofObjectNumDofs>(body_grp.bodies[0]);
        auto &tmp_state = ctx.get<DofObjectTmpState>(body_grp.bodies[0]);
        auto velocity = ctx.get<DofObjectVelocity>(body_grp.bodies[0]);

        float *S = computePhi(
            ctx, num_dofs, body_grp.comPos, tmp_state.phi);
        SpatialVector v_body = {
            {velocity.qv[0], velocity.qv[1], velocity.qv[2]}, Vector3::zero()};
        float *S_dot = computePhiDot(
                ctx, num_dofs, v_body, body_grp.comPos, tmp_state.phi);

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

        // f_i = I_i a_i + v_i [spatial star cross] I_i v_i
        tmp_state.sForce = tmp_state.spatialInertia.multiply(tmp_state.sAcc);
        tmp_state.sForce += tmp_state.sVel.crossStar(
            tmp_state.spatialInertia.multiply(tmp_state.sVel));
    }

    // Forward pass from parents to children
    for (int i = 1; i < body_grp.numBodies; ++i) {
        auto num_dofs = ctx.get<DofObjectNumDofs>(body_grp.bodies[i]);
        auto &tmp_state = ctx.get<DofObjectTmpState>(body_grp.bodies[i]);
        auto velocity = ctx.get<DofObjectVelocity>(body_grp.bodies[i]);
        auto &hier_desc = ctx.get<DofObjectHierarchyDesc>(body_grp.bodies[i]);
        assert(hier_desc.parent != Entity::none());
        auto parent_tmp_state = ctx.get<DofObjectTmpState>(hier_desc.parent);

        if (num_dofs.numDofs == (uint32_t)DofType::Hinge) {
            tmp_state.sVel = parent_tmp_state.sVel;
            tmp_state.sAcc = parent_tmp_state.sAcc;

            // v_i = v_{parent} + S * \dot{q_i}, compute S * \dot{q_i}
            float *S = computePhi(
                    ctx, num_dofs, body_grp.comPos, tmp_state.phi);
            // a_i = a_{parent} + \dot{S} * \dot{q_i} [+ S * \ddot{q_i}, which is 0]
            // Note: we are using the parent velocity here (for hinge itself)
            float *S_dot = computePhiDot(ctx, num_dofs, parent_tmp_state.sVel,
                body_grp.comPos, tmp_state.phi);

            float q_dot = velocity.qv[0];
            for (int j = 0; j < 6; ++j) {
                tmp_state.sVel[j] += S[j] * q_dot;
                tmp_state.sAcc[j] += S_dot[j] * q_dot;
            }

            // f_i = I_i a_i + v_i [spatial star cross] I_i v_i
            tmp_state.sForce = tmp_state.spatialInertia.multiply(tmp_state.sAcc);
            tmp_state.sForce += tmp_state.sVel.crossStar(
                tmp_state.spatialInertia.multiply(tmp_state.sVel));
        }
        else if (num_dofs.numDofs == (uint32_t)DofType::Ball) {
            tmp_state.sVel = parent_tmp_state.sVel;
            tmp_state.sAcc = parent_tmp_state.sAcc;

            // v_i = v_{parent} + S * \dot{q_i}, compute S * \dot{q_i}
            float *S = computePhi(
                    ctx, num_dofs, body_grp.comPos, tmp_state.phi);

            // a_i = a_{parent} + \dot{S} * \dot{q_i} [+ S * \ddot{q_i}, which is 0]
            // Note: we are using the parent velocity here (for hinge itself)
            float *S_dot = computePhiDot(ctx, num_dofs, parent_tmp_state.sVel,
                body_grp.comPos, tmp_state.phi);

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

            // f_i = I_i a_i + v_i [spatial star cross] I_i v_i
            tmp_state.sForce = tmp_state.spatialInertia.multiply(tmp_state.sAcc);
            tmp_state.sForce += tmp_state.sVel.crossStar(
                tmp_state.spatialInertia.multiply(tmp_state.sVel));
        } else { // Fixed body, Free body
            MADRONA_UNREACHABLE();
        }
    }

    // Backward pass to find bias forces
    CountT dof_index = total_dofs;
    for (CountT i = body_grp.numBodies-1; i >= 0; --i) {
        auto &hier_desc = ctx.get<DofObjectHierarchyDesc>(body_grp.bodies[i]);
        auto num_dofs = ctx.get<DofObjectNumDofs>(body_grp.bodies[i]);
        auto &tmp_state = ctx.get<DofObjectTmpState>(body_grp.bodies[i]);

        // tau_i = S_i^T f_i
        dof_index -= num_dofs.numDofs;
        float *S = computePhi(ctx, num_dofs, body_grp.comPos, tmp_state.phi);
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

    body_grp.biasForces = tau;
}

static void processContacts(Context &ctx,
                            ContactConstraint &contact,
                            ContactTmpState &tmp_state)
{
    ObjectManager &obj_mgr = *ctx.singleton<ObjectData>().mgr;
    CVPhysicalComponent ref = ctx.get<CVPhysicalComponent>(
            contact.ref);
    CVPhysicalComponent alt = ctx.get<CVPhysicalComponent>(
            contact.alt);
    CountT objID_i = ctx.get<ObjectID>(ref.physicsEntity).idx;
    CountT objID_j = ctx.get<ObjectID>(alt.physicsEntity).idx;

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
    float mu = std::min(obj_mgr.metadata[objID_i].friction.muS,
                        obj_mgr.metadata[objID_j].friction.muS);
    tmp_state.mu = mu;
}

static void gaussMinimizeFn(Context &ctx,
                            CVSingleton &cv_sing)
{
    uint32_t world_id = ctx.worldID().idx;

    StateManager *state_mgr = ctx.getStateManager();
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

    CountT num_mass_mat_bytes = sizeof(float) *
        total_num_dofs * total_num_dofs;
    // Row-major
    float *total_mass_mat = (float *) state_mgr->tmpAlloc(
            world_id, num_mass_mat_bytes);
    memset(total_mass_mat, 0, num_mass_mat_bytes);

    CountT num_full_dofs_bytes = sizeof(float) * total_num_dofs;
    float *full_free_acc = (float *)state_mgr->tmpAlloc(world_id,
            total_num_dofs * sizeof(float));
    float *full_vel = (float *)state_mgr->tmpAlloc(world_id,
            num_full_dofs_bytes);
    memset(full_free_acc, 0, num_full_dofs_bytes);
    memset(full_vel, 0, num_full_dofs_bytes);

    uint32_t processed_dofs = 0;
    for (CountT i = 0; i < num_grps; ++i) {
        float *local_mass = hiers[i].massMatrix;

        for (CountT row = 0; row < hiers[i].numDofs; ++row) {
            full_free_acc[row + processed_dofs] = hiers[i].freeAcceleration[row];

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
                    hiers[i].bodies[j]);
            DofObjectNumDofs num_dofs = ctx.get<DofObjectNumDofs>(
                    hiers[i].bodies[j]);
            for (CountT k = 0; k < num_dofs.numDofs; ++k) {
                full_vel[processed_dofs] = vel.qv[k];
                processed_dofs++;
            }
        }
    }

    // Create the contact Jacobian
    ContactConstraint *contacts = state_mgr->getWorldComponents<
        Contact, ContactConstraint>(world_id);
    ContactTmpState *contacts_tmp_state = state_mgr->getWorldComponents<
        Contact, ContactTmpState>(world_id);
    // Count contacts
    CountT num_contacts = state_mgr->numRows<Contact>(world_id);
    CountT total_contact_pts = 0;
    for (int i = 0; i < num_contacts; ++i) {
        total_contact_pts += contacts[i].numPoints;
    }

    // Process mu and penetrations for each point
    CountT num_full_contact_bytes = sizeof(float) * total_contact_pts;
    float *full_mu = (float *)state_mgr->tmpAlloc(world_id,
            num_full_contact_bytes);
    float *full_penetration = (float *)state_mgr->tmpAlloc(world_id,
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

    // Prefix sum for each of the body groups
    uint32_t block_start[num_grps];
    uint32_t block_offset = 0;
    for (CountT i = 0; i < num_grps; ++i) {
        block_start[i] = block_offset;
        block_offset += hiers[i].numDofs;
        hiers[i].tmpIdx = i;
    }

    // Jacobian is size 3n_c x n_dofs, column-major
    uint32_t J_rows = 3 * total_contact_pts;
    uint32_t J_cols = total_num_dofs;
    float *J_c = (float *) state_mgr->tmpAlloc(world_id,
        J_rows * J_cols * sizeof(float));
    memset(J_c, 0, J_rows * J_cols * sizeof(float));

    CountT jac_row = 0;
    for(CountT ct_idx = 0; ct_idx < num_contacts; ++ct_idx) {
        ContactConstraint contact = contacts[ct_idx];
        ContactTmpState &tmp_state = contacts_tmp_state[ct_idx];
        CVPhysicalComponent ref = ctx.get<CVPhysicalComponent>(
                contact.ref);
        CVPhysicalComponent alt = ctx.get<CVPhysicalComponent>(
                contact.alt);
        DofObjectTmpState &ref_tmp_state = ctx.get<DofObjectTmpState>(
                ref.physicsEntity);
        DofObjectTmpState &alt_tmp_state = ctx.get<DofObjectTmpState>(
                alt.physicsEntity);
        DofObjectNumDofs &ref_num_dofs = ctx.get<DofObjectNumDofs>(
                ref.physicsEntity);
        DofObjectNumDofs &alt_num_dofs = ctx.get<DofObjectNumDofs>(
                alt.physicsEntity);
        bool ref_fixed = ref_num_dofs.numDofs == (uint32_t)DofType::FixedBody;
        bool alt_fixed = alt_num_dofs.numDofs == (uint32_t)DofType::FixedBody;
        // Each of the contact points
        for(CountT pt_idx = 0; pt_idx < contact.numPoints; pt_idx++) {
            Vector3 contact_pt = contact.points[pt_idx].xyz();
            // Compute the Jacobians for each body at the contact point
            if(!ref_fixed) {
                DofObjectHierarchyDesc &ref_hier_desc = ctx.get<DofObjectHierarchyDesc>(
                        ref.physicsEntity);
                BodyGroupHierarchy &ref_grp = ctx.get<BodyGroupHierarchy>(
                        ref_hier_desc.bodyGroup);
                float* J_ref = computeContactJacobian(ctx,
                    ref_num_dofs, ref_grp, ref_hier_desc,
                    tmp_state.C, contact_pt, ref_tmp_state.phi);
                for(CountT i = 0; i < ref_grp.numDofs; ++i) {
                    float *J_col = J_c +
                        J_rows * (block_start[ref_grp.tmpIdx] + i) + jac_row;
                    float *J_ref_col = J_ref + 3 * i;
                    for(CountT j = 0; j < 3; ++j) {
                        J_col[j] -= J_ref_col[j];
                    }
                }
            }
            if(!alt_fixed) {
                DofObjectHierarchyDesc &alt_hier_desc = ctx.get<DofObjectHierarchyDesc>(
                        alt.physicsEntity);
                BodyGroupHierarchy &alt_grp = ctx.get<BodyGroupHierarchy>(
                        alt_hier_desc.bodyGroup);
                float *J_alt = computeContactJacobian(ctx,
                    alt_num_dofs, alt_grp, alt_hier_desc,
                    tmp_state.C, contact_pt, alt_tmp_state.phi);
                for(CountT i = 0; i < alt_grp.numDofs; ++i) {
                    float *J_col = J_c +
                        J_rows * (block_start[alt_grp.tmpIdx] + i) + jac_row;
                    float *J_alt_col = J_alt + 3 * i;
                    for(CountT j = 0; j < 3; ++j) {
                        J_col[j] += J_alt_col[j];
                    }
                }
            }
            jac_row += 3;
        }
    }

    // Convert J to row-major
    float *J_c_rm = (float *) state_mgr->tmpAlloc(world_id,
        J_rows * J_cols * sizeof(float));
    for(CountT i = 0; i < J_rows; ++i) {
        for(CountT j = 0; j < J_cols; ++j) {
            J_c_rm[j + i * J_cols] = J_c[i + j * J_rows];
        }
    }

    // Call the solver
    if (cv_sing.cvxSolve && cv_sing.cvxSolve->fn) {
        cv_sing.cvxSolve->totalNumDofs = total_num_dofs;
        cv_sing.cvxSolve->numContactPts = total_contact_pts;
        cv_sing.cvxSolve->h = physics_state.h;
        cv_sing.cvxSolve->mass = total_mass_mat;
        cv_sing.cvxSolve->free_acc = full_free_acc;
        cv_sing.cvxSolve->vel = full_vel;
        cv_sing.cvxSolve->J_c = J_c_rm;
        cv_sing.cvxSolve->mu = full_mu;
        cv_sing.cvxSolve->penetrations = full_penetration;

        cv_sing.cvxSolve->callSolve.store_release(1);
        while (cv_sing.cvxSolve->callSolve.load_acquire() != 2);
        cv_sing.cvxSolve->callSolve.store_relaxed(0);

        float *res = cv_sing.cvxSolve->resPtr;

        if (res) {
            // Update the body accelerations
            uint32_t processed_dofs = 0;
            for (CountT i = 0; i < num_grps; ++i)
            {
                for (CountT j = 0; j < hiers[i].numBodies; j++)
                {
                    auto body = hiers[i].bodies[j];
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


static void integrationStep(Context &ctx,
                            DofObjectPosition &position,
                            DofObjectVelocity &velocity,
                            DofObjectAcceleration &acceleration,
                            DofObjectNumDofs &numDofs)
{
    PhysicsSystemState &physics_state = ctx.singleton<PhysicsSystemState>();
    float h = physics_state.h;

    if (numDofs.numDofs == (uint32_t)DofType::FreeBody) {
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
    else if (numDofs.numDofs == (uint32_t)DofType::Hinge) {
        velocity.qv[0] += h * acceleration.dqv[0];
        position.q[0] += h * velocity.qv[0];
    }
    else if (numDofs.numDofs == (uint32_t)DofType::FixedBody) {
        // Do nothing
    }
    else if (numDofs.numDofs == (uint32_t)DofType::Ball) {
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
#endif

// Convert all the generalized coordinates here.
static void convertPostSolve(
        Context &ctx,
        Position &position,
        Rotation &rotation,
        const CVPhysicalComponent &phys)
{
    // TODO: use some forward kinematics results here
    Entity physical_entity = phys.physicsEntity;

    DofObjectNumDofs num_dofs = ctx.get<DofObjectNumDofs>(physical_entity);
    DofObjectTmpState tmp_state = ctx.get<DofObjectTmpState>(physical_entity);

    if (num_dofs.numDofs == (uint32_t)DofType::FreeBody) {
        position = tmp_state.comPos;
        rotation = tmp_state.composedRot;
    }
    else if (num_dofs.numDofs == (uint32_t)DofType::Hinge) {
        position = tmp_state.comPos;
        rotation = tmp_state.composedRot;
    }
    else if (num_dofs.numDofs == (uint32_t)DofType::FixedBody) {
        // Do nothing
    }
    else if (num_dofs.numDofs == (uint32_t)DofType::Ball) {
        position = tmp_state.comPos;
        rotation = tmp_state.composedRot;
    }
    else {
        MADRONA_UNREACHABLE();
    }
}

}

void registerTypes(ECSRegistry &registry)
{
    registry.registerComponent<CVPhysicalComponent>();

    registry.registerSingleton<CVSingleton>();
    registry.registerSingleton<CVHierarchyCounter>();

    registry.registerComponent<DofObjectPosition>();
    registry.registerComponent<DofObjectVelocity>();
    registry.registerComponent<DofObjectAcceleration>();
    registry.registerComponent<DofObjectNumDofs>();
    registry.registerComponent<DofObjectTmpState>();
    registry.registerComponent<DofObjectHierarchyDesc>();
    registry.registerComponent<ContactTmpState>();

    registry.registerArchetype<DofObjectArchetype>();
    registry.registerArchetype<Contact>();
    registry.registerArchetype<Joint>();

    registry.registerComponent<BodyGroupHierarchy>();
    registry.registerArchetype<BodyGroup>();

    registry.registerBundle<CVRigidBodyState>();
    registry.registerBundleAlias<SolverBundleAlias, CVRigidBodyState>();
}

void setCVGroupRoot(Context &ctx,
                    Entity body_group,
                    Entity body)
{
    Entity physics_entity =
        ctx.get<CVPhysicalComponent>(body).physicsEntity;

    auto &hierarchy = ctx.get<DofObjectHierarchyDesc>(physics_entity);
    hierarchy.sync.store_relaxed(0);
    hierarchy.leaf = true;
    hierarchy.index = 0;
    hierarchy.parentIndex = -1;
    hierarchy.parent = Entity::none();
    hierarchy.bodyGroup = ctx.loc(body_group);

    auto &body_grp_hier = ctx.get<BodyGroupHierarchy>(body_group);

    body_grp_hier.numBodies = 1;
    body_grp_hier.bodies[0] = physics_entity;
    body_grp_hier.numDofs = ctx.get<DofObjectNumDofs>(physics_entity).numDofs;
}

void makeCVPhysicsEntity(Context &ctx, 
                         Entity e,
                         Position position,
                         Rotation rotation,
                         ObjectID obj_id,
                         DofType dof_type)
{
    Entity physical_entity = ctx.makeEntity<DofObjectArchetype>();

    auto &pos = ctx.get<DofObjectPosition>(physical_entity);
    auto &vel = ctx.get<DofObjectVelocity>(physical_entity);
    auto &acc = ctx.get<DofObjectAcceleration>(physical_entity);
    auto &tmp_state = ctx.get<DofObjectTmpState>(physical_entity);

#if 0
    auto &hierarchy = ctx.get<DofObjectHierarchyDesc>(physical_entity);
    hierarchy.sync.store_relaxed(0);
    hierarchy.leaf = true;
#endif

    switch (dof_type) {
    case DofType::FreeBody: {
        pos.q[0] = position.x;
        pos.q[1] = position.y;
        pos.q[2] = position.z;

        pos.q[3] = rotation.w;
        pos.q[4] = rotation.x;
        pos.q[5] = rotation.y;
        pos.q[6] = rotation.z;

        for(int i = 0; i < 6; i++) {
            vel.qv[i] = 0.f;
            acc.dqv[i] = 0.f;
        }

        tmp_state.sVel = {{vel.qv[0], vel.qv[1], vel.qv[2]},
                          {vel.qv[3], vel.qv[4], vel.qv[5]}};
    } break;

    case DofType::Hinge: {
        pos.q[0] = 0.0f;
        vel.qv[0] = 0.f;
    } break;
    
    case DofType::Ball: {
        pos.q[0] = rotation.w;
        pos.q[1] = rotation.x;
        pos.q[2] = rotation.y;
        pos.q[3] = rotation.z;

        for(int i = 0; i < 3; i++) {
            vel.qv[i] = 0.f;
            acc.dqv[i] = 0.f;
        }
    } break;

    case DofType::FixedBody: {
    } break;
    }


    ctx.get<ObjectID>(physical_entity) = obj_id;
    ctx.get<DofObjectNumDofs>(physical_entity).numDofs = (uint32_t)dof_type;

    ctx.get<CVPhysicalComponent>(e) = {
        .physicsEntity = physical_entity,
    };

#if 0
#ifdef MADRONA_GPU_MODE
    static_assert(false, "Need to implement GPU DOF object hierarchy")
#else
    // By default, no parent
    hierarchy.parent = Entity::none();
#endif
#endif
}

void cleanupPhysicalEntity(Context &ctx, Entity e)
{
    CVPhysicalComponent physical_comp = ctx.get<CVPhysicalComponent>(e);
    ctx.destroyEntity(physical_comp.physicsEntity);
}

TaskGraphNodeID setupCVSolverTasks(TaskGraphBuilder &builder,
                                   TaskGraphNodeID broadphase,
                                   CountT num_substeps)
{
    auto cur_node = broadphase;

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

#ifdef MADRONA_GPU_MODE
        auto gauss_node = builder.addToGraph<tasks::GaussMinimizationNode>(
                {run_narrowphase});

        gauss_node = builder.addToGraph<ResetTmpAllocNode>(
                {gauss_node});
#else
        auto forward_kinematics = builder.addToGraph<ParallelForNode<Context,
             tasks::forwardKinematics,
                BodyGroupHierarchy
            >>({run_narrowphase});

        auto compute_center_of_mass = builder.addToGraph<ParallelForNode<Context,
             tasks::computeCenterOfMass,
                BodyGroupHierarchy
            >>({forward_kinematics});

        auto compute_spatial_inertia = builder.addToGraph<ParallelForNode<Context,
             tasks::computeSpatialInertia,
                BodyGroupHierarchy
            >>({compute_center_of_mass});

        auto recursive_newton_euler = builder.addToGraph<ParallelForNode<Context,
             tasks::recursiveNewtonEuler,
                BodyGroupHierarchy
            >>({compute_spatial_inertia});

        auto combine_spatial_inertia = builder.addToGraph<ParallelForNode<Context,
             tasks::combineSpatialInertias,
                BodyGroupHierarchy
            >>({recursive_newton_euler});

        auto composite_rigid_body = builder.addToGraph<ParallelForNode<Context,
             tasks::compositeRigidBody,
                BodyGroupHierarchy
            >>({combine_spatial_inertia});

        auto compute_expanded_parent = builder.addToGraph<ParallelForNode<Context,
             tasks::computeExpandedParent,
                BodyGroupHierarchy
            >>({composite_rigid_body});

        auto factorize_mass_matrix = builder.addToGraph<ParallelForNode<Context,
             tasks::factorizeMassMatrix,
                BodyGroupHierarchy
            >>({compute_expanded_parent});

        auto compute_free_acc = builder.addToGraph<ParallelForNode<Context,
             tasks::computeFreeAcceleration,
                BodyGroupHierarchy
            >>({factorize_mass_matrix});

        auto contact_node = builder.addToGraph<ParallelForNode<Context,
             tasks::processContacts,
                ContactConstraint,
                ContactTmpState
            >>({compute_free_acc});

        auto gauss_node = builder.addToGraph<ParallelForNode<Context,
             tasks::gaussMinimizeFn,
                CVSingleton
            >>({contact_node});

        auto int_node = builder.addToGraph<ParallelForNode<Context,
             tasks::integrationStep,
                 DofObjectPosition,
                 DofObjectVelocity,
                 DofObjectAcceleration,
                 DofObjectNumDofs
            >>({gauss_node});
#endif

        auto post_forward_kinematics = builder.addToGraph<ParallelForNode<Context,
             tasks::forwardKinematics,
                BodyGroupHierarchy
            >>({int_node});

        cur_node =
            builder.addToGraph<ParallelForNode<Context, tasks::convertPostSolve,
                Position,
                Rotation,
                CVPhysicalComponent
            >>({post_forward_kinematics});

        cur_node = builder.addToGraph<
            ClearTmpNode<Contact>>({cur_node});

        cur_node = builder.addToGraph<ResetTmpAllocNode>({cur_node});
    }

    auto clear_broadphase = builder.addToGraph<
        ClearTmpNode<CandidateTemporary>>({cur_node});
    
    return clear_broadphase;
}



void getSolverArchetypeIDs(uint32_t *contact_archetype_id,
                           uint32_t *joint_archetype_id)
{
    *contact_archetype_id = TypeTracker::typeID<Contact>();
    *joint_archetype_id = TypeTracker::typeID<Joint>();
}

void init(Context &ctx, CVXSolve *cvx_solve)
{
    ctx.singleton<CVSingleton>().cvxSolve = cvx_solve;
    ctx.singleton<CVHierarchyCounter>().num_bodies = 0;
}

void setCVEntityParentHinge(Context &ctx,
                            Entity body_grp,
                            Entity parent, Entity child,
                            Vector3 rel_pos_parent,
                            Vector3 rel_pos_child,
                            Vector3 hinge_axis)
{
    Entity child_physics_entity =
        ctx.get<CVPhysicalComponent>(child).physicsEntity;
    Entity parent_physics_entity =
        ctx.get<CVPhysicalComponent>(parent).physicsEntity;

    BodyGroupHierarchy &grp = ctx.get<BodyGroupHierarchy>(body_grp);

    auto &hier_desc = ctx.get<DofObjectHierarchyDesc>(child_physics_entity);
    auto &parent_hier_desc =
        ctx.get<DofObjectHierarchyDesc>(parent_physics_entity);

    hier_desc.parent = parent_physics_entity;
    hier_desc.relPositionParent = rel_pos_parent;
    hier_desc.relPositionLocal = rel_pos_child;

    hier_desc.hingeAxis = hinge_axis;

    hier_desc.leaf = true;
    hier_desc.bodyGroup = ctx.loc(body_grp);

    hier_desc.index = grp.numBodies;
    hier_desc.parentIndex = parent_hier_desc.index;


    grp.bodies[grp.numBodies] = child_physics_entity;

    // Make the parent no longer a leaf
    ctx.get<DofObjectHierarchyDesc>(parent_physics_entity).leaf = false;

    ++grp.numBodies;
    grp.numDofs += ctx.get<DofObjectNumDofs>(child_physics_entity).numDofs;
}

void setCVEntityParentBall(Context &ctx,
                           Entity body_grp,
                           Entity parent, Entity child,
                           Vector3 rel_pos_parent,
                           Vector3 rel_pos_child)
{
    Entity child_physics_entity =
        ctx.get<CVPhysicalComponent>(child).physicsEntity;
    Entity parent_physics_entity =
        ctx.get<CVPhysicalComponent>(parent).physicsEntity;

    BodyGroupHierarchy &grp = ctx.get<BodyGroupHierarchy>(body_grp);

    auto &hier_desc = ctx.get<DofObjectHierarchyDesc>(child_physics_entity);
    auto &parent_hier_desc =
        ctx.get<DofObjectHierarchyDesc>(parent_physics_entity);

    hier_desc.parent = parent_physics_entity;
    hier_desc.relPositionParent = rel_pos_parent;
    hier_desc.relPositionLocal = rel_pos_child;

    hier_desc.leaf = true;
    hier_desc.bodyGroup = ctx.loc(body_grp);

    hier_desc.index = grp.numBodies;
    hier_desc.parentIndex = parent_hier_desc.index;


    grp.bodies[grp.numBodies] = child_physics_entity;

    // Make the parent no longer a leaf
    ctx.get<DofObjectHierarchyDesc>(parent_physics_entity).leaf = false;

    ++grp.numBodies;
    grp.numDofs += ctx.get<DofObjectNumDofs>(child_physics_entity).numDofs;
}

Entity makeCVBodyGroup(Context &ctx)
{
    Entity e = ctx.makeEntity<BodyGroup>();
    ctx.get<BodyGroupHierarchy>(e).numBodies = 0;
    return e;
}
    
}
