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
                Quat::angleAxis(position.q[0], rotated_hinge_axis);

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

        default: {
            // Only hinges have parents
            assert(false);
        } break;
        }
    }
}

// Pre-CRB: compute the spatial inertia in common Plücker coordinates
static void computeSpatialInertia(Context &ctx,
                               DofObjectNumDofs &num_dofs,
                               DofObjectTmpState &tmp_state,
                               ObjectID &obj_id)
{
    if(num_dofs.numDofs == (uint32_t)DofType::FixedBody) {
        return;
    }

    ObjectManager &obj_mgr = *ctx.singleton<ObjectData>().mgr;
    RigidBodyMetadata &metadata = obj_mgr.metadata[obj_id.idx];

    Diag3x3 inertia = Diag3x3::fromVec(metadata.mass.invInertiaTensor).inv();
    float mass = 1.f / metadata.mass.invMass;

    // We need to find inertia tensor in world space orientation
    Mat3x3 rot_mat = Mat3x3::fromQuat(tmp_state.composedRot);
    Mat3x3 i_world_frame = inertia * rot_mat;
    i_world_frame = i_world_frame.transpose() * rot_mat;

    // Compute the 3x3 skew-symmetric matrix (r^x) (where r is from Plücker origin to COM)
    Mat3x3 sym_mat = skewSymmetricMatrix(tmp_state.comPos);
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
    tmp_state.spatialInertia.mCom = mass * tmp_state.comPos;
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
        Vector3 r_cross_hinge = anchorPos.cross(hinge);
        S[0] = r_cross_hinge.x;
        S[1] = r_cross_hinge.y;
        S[2] = r_cross_hinge.z;
        S[3] = hinge.x;
        S[4] = hinge.y;
        S[5] = hinge.z;
    }
    else {
        MADRONA_UNREACHABLE();
    }
    return S;
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

        float *S_i = computePhi(ctx, i_num_dofs, i_tmp_state.phi);

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

            float *S_j = computePhi(ctx, j_num_dofs, j_tmp_state.phi);

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

static void recursiveNewtonEuler(Context &ctx,
                                BodyGroupHierarchy &body_grp) {
    PhysicsSystemState &physics_state = ctx.singleton<PhysicsSystemState>();
    // Forward pass. Find in Plücker coordinates:
    //  1. velocities. v_i = v_{parent} + S * \dot{q_i}
    //  2. accelerations. TODO
    //  3. forces. TODO

    // First handle root of hierarchy
    auto num_dofs = ctx.get<DofObjectNumDofs>(body_grp.bodies[0]);
    auto tmp_state = ctx.get<DofObjectTmpState>(body_grp.bodies[0]);
    auto velocity = ctx.get<DofObjectVelocity>(body_grp.bodies[0]);
    float *S = computePhi(ctx, num_dofs, tmp_state.phi);
    float *v = (float *) ctx.getStateManager()->tmpAlloc(
        ctx.worldID().idx, 6 * sizeof(float));
    for (int j = 0; j < 6; ++j) {
        v[j] = 0.f;
        for (int k = 0; k < num_dofs.numDofs; ++k) {
            v[j] += S[j + 6 * k] * velocity.qv[k];
        }
    }
    tmp_state.vTrans = {v[0], v[1], v[2]};
    tmp_state.vRot = {v[3], v[4], v[5]};
    tmp_state.aTrans = physics_state.g;
    tmp_state.aRot = {0.f, 0.f, 0.f};

    // Forward pass from parents to children
    for (int i = 1; i < body_grp.numBodies; ++i) {
        auto num_dofs = ctx.get<DofObjectNumDofs>(body_grp.bodies[i]);
        auto tmp_state = ctx.get<DofObjectTmpState>(body_grp.bodies[i]);
        auto velocity = ctx.get<DofObjectVelocity>(body_grp.bodies[i]);
        auto &hier_desc = ctx.get<DofObjectHierarchyDesc>(body_grp.bodies[i]);
        float *S = computePhi(ctx, num_dofs, tmp_state.phi);

        // Output: Plücker coordinates velocity, acceleration, force
        float *v = (float *) ctx.getStateManager()->tmpAlloc(
            ctx.worldID().idx, 6 * sizeof(float));
        float *a = (float *) ctx.getStateManager()->tmpAlloc(
            ctx.worldID().idx, 6 * sizeof(float));
        float *f = (float *) ctx.getStateManager()->tmpAlloc(
            ctx.worldID().idx, 6 * sizeof(float));

        if (num_dofs.numDofs == (uint32_t)DofType::Hinge) {
            // v_i = v_{parent} + S * \dot{q_i}, compute S * \dot{q_i}
            for (int j = 0; j < 6; ++j) {
                v[j] = velocity.qv[0] * S[j];
            }
            // a_i = a_{parent} + \dot{S} * \dot{q_i} + S * \ddot{q_i} (\ddot{q_i} = 0)
            //TODO!
        }
        else { // Fixed body, Free body
            MADRONA_UNREACHABLE();
        }

        // Store in tmp state
        tmp_state.vTrans = {v[0], v[1], v[2]};
        tmp_state.vRot = {v[3], v[4], v[5]};
        tmp_state.aTrans = {a[0], a[1], a[2]};
        tmp_state.aRot = {a[3], a[4], a[5]};

        // Add in velocity, acceleration, force from parent
        if(hier_desc.parent != Entity::none()) {
            auto parentTmpState = ctx.get<DofObjectTmpState>(
                body_grp.bodies[hier_desc.parentIndex]);
            tmp_state.vTrans += parentTmpState.vTrans;
            tmp_state.vRot += parentTmpState.vRot;
            tmp_state.aTrans += parentTmpState.aTrans;
            tmp_state.aRot += parentTmpState.aRot;
        }
    }

    // Backward pass to find bias forces
    for (CountT i = body_grp.numBodies-1; i >= 0; --i) {
        auto &hier_desc = ctx.get<DofObjectHierarchyDesc>(body_grp.bodies[i]);
        // tau_i = S_i^T f_i

        // Add to parent's force (TODO: convince myself this makes sense)
        if (hier_desc.parent != Entity::none()) {
        }
    }
}

static void processContacts(Context &ctx,
                            ContactConstraint &contact,
                            ContactTmpState &tmp_state)
{
    CVPhysicalComponent ref = ctx.get<CVPhysicalComponent>(
            contact.ref);
    CVPhysicalComponent alt = ctx.get<CVPhysicalComponent>(
            contact.alt);

    auto &ref_pos = ctx.get<DofObjectPosition>(ref.physicsEntity);
    auto &alt_pos = ctx.get<DofObjectPosition>(alt.physicsEntity);
    Vector3 ref_com = Vector3(ref_pos.q[0], ref_pos.q[1], ref_pos.q[2]);
    Vector3 alt_com = Vector3(alt_pos.q[0], alt_pos.q[1], alt_pos.q[2]);

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

    tmp_state.n = n;
    tmp_state.t = t;
    tmp_state.s = s;

    CountT i = 0;
    float max_penetration = -FLT_MAX;
    for(; i < contact.numPoints; ++i) {
        Vector4 point = contact.points[i];
        // Compute the relative positions of the contact
        tmp_state.rRefComToPt[i] = point.xyz() - ref_com;
        tmp_state.rAltComToPt[i] = point.xyz() - alt_com;
        tmp_state.maxPenetration = std::max(max_penetration, point.w);
    }

    tmp_state.num_contacts = contact.numPoints;

    // Get friction coefficient
    ObjectManager &obj_mgr = *ctx.singleton<ObjectData>().mgr;
    CountT objID_i = ctx.get<ObjectID>(ref.physicsEntity).idx;
    CountT objID_j = ctx.get<ObjectID>(alt.physicsEntity).idx;
    RigidBodyMetadata &metadata_i = obj_mgr.metadata[objID_i];
    RigidBodyMetadata &metadata_j = obj_mgr.metadata[objID_j];
    tmp_state.mu = std::min(metadata_i.friction.muS,
                            metadata_j.friction.muS);
}

static void gaussMinimizeFn(Context &ctx,
                            CVSingleton &cv_sing)
{
}


static void integrationStep(Context &ctx,
                            DofObjectPosition &position,
                            DofObjectVelocity &velocity,
                            DofObjectNumDofs &numDofs,
                            DofObjectTmpState &tmp_state,
                            ObjectID &objID)
{
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
    DofObjectPosition pos = ctx.get<DofObjectPosition>(physical_entity);

    if (num_dofs.numDofs == (uint32_t)DofType::FreeBody) {
        position.x = pos.q[0];
        position.y = pos.q[1];
        position.z = pos.q[2];

        rotation = Quat {
            pos.q[3],
            pos.q[4],
            pos.q[5],
            pos.q[6],
        };
    }
    else if (num_dofs.numDofs == (uint32_t)DofType::Hinge) {
        // TODO: May need some forward kinematics here
    }
    else if (num_dofs.numDofs == (uint32_t)DofType::FixedBody) {
        // Do nothing
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

        vel.qv[0] = 0.f;
        vel.qv[1] = 0.f;
        vel.qv[2] = 0.f;
        vel.qv[3] = 0.f;
        vel.qv[4] = 0.f;
        vel.qv[5] = 0.f;
    } break;

    case DofType::Hinge: {
        pos.q[0] = 0.f;
        vel.qv[0] = 0.f;
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

        auto compute_spatial_inertia = builder.addToGraph<ParallelForNode<Context,
             tasks::computeSpatialInertia,
                DofObjectNumDofs,
                DofObjectTmpState,
                ObjectID
            >>({forward_kinematics});

        auto combine_spatial_inertia = builder.addToGraph<ParallelForNode<Context,
             tasks::combineSpatialInertias,
                BodyGroupHierarchy
            >>({compute_spatial_inertia});

        auto composite_rigid_body = builder.addToGraph<ParallelForNode<Context,
             tasks::compositeRigidBody,
                BodyGroupHierarchy
            >>({combine_spatial_inertia});

        auto recursive_newton_euler = builder.addToGraph<ParallelForNode<Context,
             tasks::recursiveNewtonEuler,
                BodyGroupHierarchy
            >>({composite_rigid_body});

        auto contact_node = builder.addToGraph<ParallelForNode<Context,
             tasks::processContacts,
                ContactConstraint,
                ContactTmpState
            >>({recursive_newton_euler});

        auto gauss_node = builder.addToGraph<ParallelForNode<Context,
             tasks::gaussMinimizeFn,
                CVSingleton
            >>({contact_node});

        auto int_node = builder.addToGraph<ParallelForNode<Context,
             tasks::integrationStep,
                 DofObjectPosition,
                 DofObjectVelocity,
                 DofObjectNumDofs,
                 DofObjectTmpState,
                 ObjectID
            >>({gauss_node});
#endif

        cur_node =
            builder.addToGraph<ParallelForNode<Context, tasks::convertPostSolve,
                Position,
                Rotation,
                CVPhysicalComponent
            >>({int_node});

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
