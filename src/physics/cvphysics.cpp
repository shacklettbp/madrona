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

// Forward Kinematics: compute world positions and orientations
// We also
static void forwardKinematics(Context &ctx,
                               DofObjectPosition &position,
                               DofObjectNumDofs &num_dofs,
                               DofObjectTmpState &tmp_state,
                               DofObjectHierarchyDesc &hier_desc)
{
    // We are root
    if (hier_desc.parent == Entity::none()) {
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

        hier_desc.sync.store_release(1);
    } else { // We have a parent
        Entity parent_e = hier_desc.parent;

        DofObjectHierarchyDesc &parent_hier_desc =
            ctx.get<DofObjectHierarchyDesc>(parent_e);
        DofObjectTmpState &parent_tmp_state =
            ctx.get<DofObjectTmpState>(parent_e);

        // Wait for parent to finish computing their world pos / orientation
        while (parent_hier_desc.sync.load_acquire() != 1) {}

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

            // TODO: check this is ok (child only needs COM and quaternions right?)
            // We're ok to let any children read our tmp state here
            hier_desc.sync.store_release(1);

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

// Pre-CRB: compute the spatial inertia (inertia in a single common Plücker coordinates)
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

    // Compute the 3x3 symmetric matrix (m r^x r^xT) (where r is from Plücker origin to COM)
    Mat3x3 sym_mat = skewSymmetricMatrix(tmp_state.comPos);
    sym_mat = sym_mat * sym_mat.transpose();
    sym_mat = sym_mat * mass;

    // (I_world + m r^x r^xT)
    Mat3x3 inertia_mat = i_world_frame + sym_mat;

    // Take only the upper triangular part (since it's symmetric)
    tmp_state.spatialInertia.spatial_inertia[0] = inertia_mat[0][0];
    tmp_state.spatialInertia.spatial_inertia[1] = inertia_mat[1][1];
    tmp_state.spatialInertia.spatial_inertia[2] = inertia_mat[2][2];
    tmp_state.spatialInertia.spatial_inertia[3] = inertia_mat[1][0];
    tmp_state.spatialInertia.spatial_inertia[4] = inertia_mat[2][0];
    tmp_state.spatialInertia.spatial_inertia[5] = inertia_mat[2][1];

    // Rest of parameters
    tmp_state.spatialInertia.mass = mass;
    tmp_state.spatialInertia.com = tmp_state.comPos;
}

// CRB: Compute the Mass Matrix (n_dofs x n_dofs)
static void compositeRigidBody(Context &ctx,
                               CVSingleton &cv_sing)
{
    uint32_t world_id = ctx.worldID().idx;
    StateManager *state_mgr = ctx.getStateManager();

    DofObjectTmpState *tmp_states = state_mgr->getWorldComponents<
        DofObjectArchetype, DofObjectTmpState>(world_id);
    DofObjectHierarchyDesc *hier_descs = state_mgr->getWorldComponents<
        DofObjectArchetype, DofObjectHierarchyDesc>(world_id);
    DofObjectNumDofs *num_dofs = state_mgr->getWorldComponents<
        DofObjectArchetype, DofObjectNumDofs>(world_id);

    CountT num_bodies = state_mgr->numRows<DofObjectArchetype>(world_id);
    CountT total_dofs = 0;
    for (int i = 0; i < num_bodies; ++i) { total_dofs += num_dofs[i].numDofs; }

    // Initialize M as n_dofs x n_dofs matrix
    float *M = (float *)state_mgr->tmpAlloc(
            world_id, sizeof(float) * total_dofs * total_dofs );
    memset(M, 0, sizeof(float) * total_dofs * total_dofs);

    // Want to map from hierarchy numbering to table numbering
    uint32_t *hier_to_table = (uint32_t *)state_mgr->tmpAlloc(
            world_id, sizeof(uint32_t) * num_bodies);
    for(CountT i = 0; i < num_bodies; ++i) {
        hier_to_table[hier_descs[i].numbering] = i;
    }

    // Backwards pass: children first
    for (CountT i = num_bodies; i > 0; --i) {
        uint32_t table_idx = hier_to_table[i]; // index in the ECS table
        Entity parent = hier_descs[table_idx].parent;


        InertiaTensor I_c = tmp_states[table_idx].spatialInertia;
        // Add spatial inertia to parent's value
        if(parent != Entity::none())
        {
            uint32_t parent_number = ctx.get<DofObjectHierarchyDesc>(parent).numbering;
            uint32_t parent_table_idx = hier_to_table[parent_number];
            tmp_states[parent_table_idx].spatialInertia += I_c;
        }

        // F = I_i^c * S_i
        tmp_states[table_idx].phi; // -> TODO: convert this to S_i

        // M_{ii} = S_i^T I_i^c S_i = S_i^T F

        CountT j = i;
        // Traverse up the hierarchy
        while(hier_descs[hier_to_table[j]].parent != Entity::none())
        {
            j = ctx.get<DofObjectHierarchyDesc>(hier_descs[j].parent).numbering;
            // M_ij = F^T S_j = S_i^T I_i^c S_j

            // M_ji = M_ij^T (symmetric) - maybe don't need to store this
        }
    }

    // TODO: store M
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
    uint32_t world_id = ctx.worldID().idx;

    StateManager *state_mgr = ctx.getStateManager();
    PhysicsSystemState &physics_state = ctx.singleton<PhysicsSystemState>();

    CountT num_bodies = state_mgr->numRows<DofObjectArchetype>(world_id);

    // Recover necessary pointers.
    DofObjectNumDofs *num_dofs = state_mgr->getWorldComponents<
        DofObjectArchetype, DofObjectNumDofs>(world_id);
    DofObjectTmpState *tmp_states = state_mgr->getWorldComponents<
        DofObjectArchetype, DofObjectTmpState>(world_id);
    DofObjectVelocity *body_vels = state_mgr->getWorldComponents<
        DofObjectArchetype, DofObjectVelocity>(world_id);
    DofObjectHierarchyDesc *hier_descs = state_mgr->getWorldComponents<
        DofObjectArchetype, DofObjectHierarchyDesc>(world_id);

    // Offets to know where particular DOF coordinates are set
    uint32_t *vel_coord_offsets = (uint32_t *)state_mgr->tmpAlloc(
            world_id, sizeof(uint32_t) * num_bodies);
    uint32_t num_vel_coords = 0;

    { // Compute the prefix sum of entities' num velocity dofs.
        for (int i = 0; i < num_bodies; ++i) {
            vel_coord_offsets[i] = num_vel_coords;
            num_vel_coords += num_dofs[i].numDofs;
        }
    }

    // c*(i) is the set of bodies in the subtree rooted at i.
    // For each body i, this has an array of flags saying whether
    // body j is in the subtree rooted at body i.
    bool *subtree_flags = (bool *)state_mgr->tmpAlloc(
            world_id, sizeof(bool) * num_bodies * num_bodies);
    memset(subtree_flags, 0, sizeof(bool) * num_bodies * num_bodies);
    auto c_star = [subtree_flags, num_bodies](uint32_t i, uint32_t j) -> bool & {
        return subtree_flags[i * num_bodies + j];
    };

    for (int i = 0; i < num_bodies; ++i) {
        Entity parent = hier_descs[i].parent;
        if (parent != Entity::none() &&
            c_star[parent.row, i] == 0) {
            c_star[parent.row, i] = 1;

            // Walk up
            uint32_t current = parent.row;
            parent = hier_descs[parent.row].parent;
            while (parent != Entity::none()) {
                c_star[parent.row, current] = 1;
                parent = hier_descs[parent.row].parent;
            }
        }
    }

    float inertia_mat_tmp[6][6] = {};
    memset(inertia_mat_tmp, 0, sizeof(float) * 6 * 6);

    auto add_inertia_sym = [&inertia_mat_tmp](uint32_t off_row,
                                               uint32_t off_col,
                                               float *values) {
        inertia_mat_tmp[off_row+0][off_col+0] += values[0];
        inertia_mat_tmp[off_row+1][off_col+1] += values[1];
        inertia_mat_tmp[off_row+2][off_col+2] += values[2];

        inertia_mat_tmp[off_row+1][off_col+0] = 
            (inertia_mat_tmp[off_row+0][off_col+1] +=
            values[3]);

        inertia_mat_tmp[off_row+2][off_col+0] = 
            (inertia_mat_tmp[off_row+0][off_col+2] +=
            values[4]);

        inertia_mat_tmp[off_row+2][off_col+1] = 
            (inertia_mat_tmp[off_row+1][off_col+2] +=
            values[5]);
    };

    // skew symmetric
    auto populate_inertia_ssym = [&inertia_mat_tmp](uint32_t off_row,
                                                    uint32_t off_col,
                                                    float *values,
                                                    bool transpose = false) {
        float coeff = transpose ? -1.f : 1.f;
        inertia_mat_tmp[0][1] = -values[3] * coeff;
        inertia_mat_tmp[0][2] = values[2] * coeff;
        inertia_mat_tmp[1][0] = values[3] * coeff;
        inertia_mat_tmp[1][2] = -values[1] * coeff;
        inertia_mat_tmp[2][0] = -values[2] * coeff;
        inertia_mat_tmp[2][1] = values[1] * coeff;
    };

    auto right_multiply_phi = [&](uint32_t body_idx,
                                  float *inertia_matrix) {
        if (num_dofs[body_idx].numDofs == 6) {
            
        }
    };

    for (int i = 0; i < num_bodies; ++i) {
        for (int j = 0; j < num_bodies, ++j) {
            uint32_t row = vel_coord_offsets[i];
            uint32_t col = vel_coord_offsets[j];

            uint32_t block_width = num_dofs[j].numDofs;
            uint32_t block_height = num_dofs[i].numDofs;

            if (subtree_flags[j][i]) {
                // Top left is (I_world + m * r^x r^xT), which we already stored
                add_inertia_sym(0, 0, tmp_states[i].spatialInertia.spatial_inertia);

                // Top right, bottom left is (m * r^x)
                populate_inertia_ssym(0, 3, tmp_states[i].spatialInertia.com);
                populate_inertia_ssym(3, 0, tmp_states[i].spatialInertia.com, true);

                inertia_mat_tmp[3][3] = tmp_states[i].spatialInertia.mass;
                inertia_mat_tmp[4][4] = tmp_states[i].spatialInertia.mass;
                inertia_mat_tmp[5][5] = tmp_states[i].spatialInertia.mass;


                
            } else if (subtree_flags[i][j]) {
                // Use inertia matrix from j
                populate_inertia_sym(0, 0, tmp_states[j].spatialInertia.vInertia);
                add_inertia_sym(0, 0, tmp_states[j].spatialInertia.comSquared);

                populate_inertia_ssym(0, 3, tmp_states[j].spatialInertia.com);
                populate_inertia_ssym(3, 0, tmp_states[j].spatialInertia.com, true);

                inertia_mat_tmp[3][3] = tmp_states[j].spatialInertia.mass;
                inertia_mat_tmp[4][4] = tmp_states[j].spatialInertia.mass;
                inertia_mat_tmp[5][5] = tmp_states[j].spatialInertia.mass;
            }

            memset(inertia_mat_tmp, 0, sizeof(float) * 6 * 6);
            memset(matmul_tmp, 0, sizeof(float) * 6 * 6);
        }
    }
    
#if 0
    // Allocate space for all the linear and angular velocity jacobians:
    // For now, we will keep them all separate and feed them into python.
    // Each body needs a 3x[num_vel_coords] jacobian matrix for both the
    // linear and angular velocities.
    CountT num_lin_vel_jac_bytes = sizeof(float) * num_bodies * 3 * num_vel_coords;
    CountT num_ang_vel_jac_bytes = sizeof(float) * num_bodies * 3 * num_vel_coords;

    float *lin_vel_jac = (float *)state_mgr->allocTmp(
            world_id, num_lin_vel_jac_bytes);
    float *ang_vel_jac = (float *)state_mgr->allocTmp(
            world_id, num_ang_vel_jac_bytes);

    memset(lin_vel_jac, 0, num_lin_vel_jac_bytes);
    memset(ang_vel_jac, 0, num_ang_vel_jac_bytes);

    { // Compute these jacobians
        // First, compute the individual local jacobians before doing a 
        // prefix sum.
        float *lin_vel_i, *ang_vel_i;
        auto j_v_i = [&lin_vel_i, num_vel_coords]
            (uint32_t row, uint32_t col) -> float & {
            return lin_vel_i[col + row * num_vel_coords];
        };

        auto j_w_i = [&ang_vel_i, num_vel_coords]
            (uint32_t row, uint32_t col) -> float & {
            return ang_vel_i[col + row * num_vel_coords];
        };
        
        for (int i = 0; i < num_bodies; ++i) {
            lin_vel_i = lin_vel_jac + sizeof(float) * i * 3 * num_vel_coords;
            ang_vel_i = ang_vel_jac + sizeof(float) * i * 3 * num_vel_coords;
            
            uint32_t vel_coord_offset = vel_coord_offsets[i];

            if (num_dofs[i].numDofs == (uint32_t)DofType::FreeBody) {
                // The jacobians are just identity at the right spot.
                j_v_i[0, vel_coord_offset] = 1.f;
                j_v_i[0, vel_coord_offset+1] = 1.f;
                j_v_i[0, vel_coord_offset+2] = 1.f;

                j_w_i[0, vel_coord_offset+3] = 1.f;
                j_w_i[0, vel_coord_offset+4] = 1.f;
                j_w_i[0, vel_coord_offset+5] = 1.f;
            } else if (num_dofs[i].numDofs == (uint32_t)DofType::Hinge) {
                DofObjectHierarchyDesc *hier_desc = &hier_descs[i];

                // For hinges, it's a little more complicated. We need to walk
                // up the hierarchy not only to figure out the row indices of
                // the parent entities, but also the values in the matrices.
                while (hier_desc->parent != Entity::none()) {
                    
                }
            }
        }
    }
#endif
}

#if 0
static void gaussMinimizeFn(Context &ctx,
                            DofObjectPosition &position,
                            DofObjectVelocity &velocity,
                            DofObjectNumDofs &numDofs,
                            DofObjectTmpState &tmp_state,
                            ObjectID &objID)
{
    (void)numDofs;

    // TODO: Gather all the physics objects and do the minimization
    ObjectManager &obj_mgr = *ctx.singleton<ObjectData>().mgr;
    PhysicsSystemState &physics_state = ctx.singleton<PhysicsSystemState>();

    // Phase 1: compute the mass matrix data
    RigidBodyMetadata &metadata = obj_mgr.metadata[objID.idx];
    float invMass = metadata.mass.invMass;
    Vector3 invInertia = metadata.mass.invInertiaTensor;

    Diag3x3 inv_inertia_body = Diag3x3::fromVec(invInertia);
    Diag3x3 inertia_body = inv_inertia_body.inv();
    Quat rot_quat = {
        position.q[3],
        position.q[4],
        position.q[5],
        position.q[6],
    };
    Vector3 omega = {
        velocity.qv[3],
        velocity.qv[4],
        velocity.qv[5],
    };

    Mat3x3 rot_mat = Mat3x3::fromQuat(rot_quat);

    // Inertia and Inverse Inertia in world frame
    Mat3x3 I_world_frame = rot_mat * inertia_body;
    I_world_frame = rot_mat * I_world_frame.transpose();
    Mat3x3 inv_I_world_frame = rot_mat * inv_inertia_body;
    inv_I_world_frame = rot_mat * inv_I_world_frame.transpose();

    // Step 2: compute the external and gyroscopic forces
    // Step 2.1: gravity
    Vector3 force_gravity = physics_state.g / invMass;
    Vector3 trans_forces = force_gravity;

    // Step 2.2: gyroscopic moments
    Vector3 gyro_moment = -(omega.cross(I_world_frame * omega));
    Vector3 rot_moments = gyro_moment;

    // Infinite mass object, just zero out the forces
    if(invMass == 0) {
        trans_forces = Vector3::zero();
        rot_moments = Vector3::zero();
    }

    tmp_state.invMass = invMass;
    tmp_state.invInertia = inv_I_world_frame;
    tmp_state.externalForces = trans_forces;
    tmp_state.externalMoment = rot_moments;
}
#endif


#if 0
template <typename MatrixT>
static inline Mat3x3 rightMultiplyCross(const MatrixT &m,
                                        const Vector3 &v)
{
    return {
        {
                { -m[2][0]*v.y + m[1][0]*v.z, -m[2][1]*v.y + m[1][1]*v.z, -m[2][2]*v.y + m[1][2]*v.z },
               { m[2][0]*v.x - m[0][0]*v.z, m[2][1]*v.x - m[0][1]*v.z, m[2][2]*v.x - m[0][2]*v.z },
               { -m[1][0]*v.x + m[0][0]*v.y, -m[1][1]*v.x + m[0][1]*v.y, -m[1][2]*v.x + m[0][2]*v.y }
            }
    };
}
#endif

#if 0
static void solveSystem(Context &ctx,
                        CVSingleton &cv_sing)
{
    uint32_t world_id = ctx.worldID().idx;

    StateManager *state_mgr = ctx.getStateManager();
    PhysicsSystemState &physics_state = ctx.singleton<PhysicsSystemState>();

    CountT num_contacts = state_mgr->numRows<Contact>(world_id);
    CountT num_bodies = state_mgr->numRows<DofObjectArchetype>(world_id);

    ContactConstraint *contacts = state_mgr->getWorldComponents<
        Contact, ContactConstraint>(world_id);
    ContactTmpState *contacts_tmp_state = state_mgr->getWorldComponents<
        Contact, ContactTmpState>(world_id);
    DofObjectTmpState *tmp_states = state_mgr->getWorldComponents<
        DofObjectArchetype, DofObjectTmpState>(world_id);
    DofObjectVelocity *body_vels = state_mgr->getWorldComponents<
        DofObjectArchetype, DofObjectVelocity>(world_id);

    // Each contact archetype can have multiple contacts
    CountT total_contacts = 0;
    for(CountT i = 0; i < num_contacts; ++i) {
        ContactTmpState &contact_tmp_state = contacts_tmp_state[i];
        total_contacts += contact_tmp_state.num_contacts;
    }
    ContactPointInfo *contact_point_info = (ContactPointInfo *)state_mgr->tmpAlloc(
            world_id, sizeof(ContactPointInfo) * total_contacts);

    CountT contact_pt_idx = 0;
    for(CountT i = 0; i < num_contacts; ++i) {
        ContactTmpState &contact_tmp_state = contacts_tmp_state[i];
        for(CountT pt_idx = 0; pt_idx < contact_tmp_state.num_contacts; ++pt_idx) {
            contact_point_info[contact_pt_idx].parentIdx = i;
            contact_point_info[contact_pt_idx].subIdx = pt_idx;
            contact_pt_idx++;
        }
    }

    CountT num_dof = 6 * num_bodies;
    // Compute b, v (stack values for each body)
    float *btr = (float *)state_mgr->tmpAlloc(
            world_id, sizeof(float) * num_dof);
    float *vptr = (float *)state_mgr->tmpAlloc(
            world_id, sizeof(float) * num_dof);
    for(CountT i = 0; i < num_bodies; ++i) {
        DofObjectTmpState &body_tmp_state = tmp_states[i];
        DofObjectVelocity &body_vel = body_vels[i];
        CountT body_idx = i * 6;
        btr[body_idx] = body_tmp_state.externalForces[0];
        btr[body_idx + 1] = body_tmp_state.externalForces[1];
        btr[body_idx + 2] = body_tmp_state.externalForces[2];
        btr[body_idx + 3] = body_tmp_state.externalMoment[0];
        btr[body_idx + 4] = body_tmp_state.externalMoment[1];
        btr[body_idx + 5] = body_tmp_state.externalMoment[2];
        vptr[body_idx] = body_vel.qv[0];
        vptr[body_idx + 1] = body_vel.qv[1];
        vptr[body_idx + 2] = body_vel.qv[2];
        vptr[body_idx + 3] = body_vel.qv[3];
        vptr[body_idx + 4] = body_vel.qv[4];
        vptr[body_idx + 5] = body_vel.qv[5];
    }

    CountT jacob_size = (3 * total_contacts) * (num_dof);

    float *jacob_ptr = (float *)state_mgr->tmpAlloc(
            world_id, sizeof(float) * jacob_size);
    memset(jacob_ptr, 0, sizeof(float) * jacob_size);
    // Lambda for setting in col major order
    auto j_entry = [jacob_ptr, total_contacts, num_dof](int col, int row) -> float & {
        assert(col < num_dof);
        assert(row < 3 * total_contacts);
        return jacob_ptr[row + col * (3 * total_contacts)];
    };

    for(CountT i = 0; i < total_contacts; i++)
    {
        ContactPointInfo &pt_info = contact_point_info[i];
        ContactConstraint &contact = contacts[pt_info.parentIdx];
        ContactTmpState &contact_tmp_state = contacts_tmp_state[pt_info.parentIdx];

        // Get the loc of the two bodies
        CVPhysicalComponent &ref_phys_comp = ctx.get<CVPhysicalComponent>(
                contact.ref);
        CountT ref_idx = ctx.loc(ref_phys_comp.physicsEntity).row;

        CVPhysicalComponent &alt_phys_comp = ctx.get<CVPhysicalComponent>(
                contact.alt);
        CountT alt_idx = ctx.loc(alt_phys_comp.physicsEntity).row;

        // Location of body ref and alt
        CountT ref_col_start = ref_idx * 6;
        CountT alt_col_start = alt_idx * 6;

        Mat3x3 ref_linear_cT;
        Mat3x3 alt_linear_cT;

        CountT row_start = 3 * i;
        // J = [... -C^T, C^T r_i^x, ... C^T, -C^T r_j^x ...]
        for (int i = 0; i < 3; ++i) {
            ref_linear_cT[i][0] = j_entry(ref_col_start + i, row_start) = -contact_tmp_state.n[i];
            ref_linear_cT[i][1] = j_entry(ref_col_start + i, row_start+1) = -contact_tmp_state.t[i];
            ref_linear_cT[i][2] = j_entry(ref_col_start + i, row_start+2) = -contact_tmp_state.s[i];

            alt_linear_cT[i][0] = j_entry(alt_col_start + i, row_start) = contact_tmp_state.n[i];
            alt_linear_cT[i][1] = j_entry(alt_col_start + i, row_start+1) = contact_tmp_state.t[i];
            alt_linear_cT[i][2] = j_entry(alt_col_start + i, row_start+2) = contact_tmp_state.s[i];
        }

        // C^T r_i^x, C^T r_j^x
        ref_linear_cT = rightMultiplyCross(ref_linear_cT,
                                          -contact_tmp_state.rRefComToPt[pt_info.subIdx]); // need negative to cancel first one
        alt_linear_cT = rightMultiplyCross(alt_linear_cT,
                                          -contact_tmp_state.rAltComToPt[pt_info.subIdx]);
        for (int col = 0; col < 3; ++col)
        {
            for (int row = 0; row < 3; ++row) {
                j_entry(ref_col_start + 3 + col, row_start + row) = ref_linear_cT[col][row];
                j_entry(alt_col_start + 3 + col, row_start + row) = alt_linear_cT[col][row];
            }
        }
    }

    // Create M^{-1}
    CountT M_inv_size = num_dof * num_dof;
    float *M_inv_ptr = (float *)state_mgr->tmpAlloc(
            world_id, sizeof(float) * M_inv_size);
    memset(M_inv_ptr, 0, sizeof(float) * M_inv_size);

    auto M_inv_entry = [M_inv_ptr, num_dof](int col, int row) -> float & {
        assert(row < num_dof);
        assert(col < num_dof);
        return M_inv_ptr[row + col * num_dof];
    };

    for(CountT body_idx = 0; body_idx < num_bodies; ++body_idx) {
        DofObjectTmpState &tmp_state = tmp_states[body_idx];
        CountT idx_start = body_idx * 6;

        // Diagonal 1 / mass entries
        for (int i = 0; i < 3; ++i) {
            M_inv_entry(idx_start + i, idx_start + i) = tmp_state.invMass;
        }
        // Inverse inertia entries
        for (int i = 0; i < 3; ++i) {
            M_inv_entry(idx_start + i + 3, idx_start + 3) = tmp_state.invInertia[i][0];
            M_inv_entry(idx_start + i + 3, idx_start + 4) = tmp_state.invInertia[i][1];
            M_inv_entry(idx_start + i + 3, idx_start + 5) = tmp_state.invInertia[i][2];
        }
    }

    // Build A=J_c M^{-1} J_c^T
    // First, compute J_c M^{-1} (3 * num_contacts x num_dof)
    float *J_M_inv_ptr = (float *)state_mgr->tmpAlloc(
            world_id, sizeof(float) * (3 * total_contacts) * num_dof);
    memset(J_M_inv_ptr, 0, sizeof(float) * (3 * total_contacts) * num_dof);
    // We can re-use the lambda earlier since they share same num rows
    auto jm_inv_entry = [J_M_inv_ptr, total_contacts, num_dof](int col, int row) -> float & {
        assert(row < 3 * total_contacts);
        assert(col < num_dof);

        return J_M_inv_ptr[row + col * (3 * total_contacts)];
    };
    for(CountT i = 0; i < 3 * total_contacts; ++i) {
        for(CountT j = 0; j < num_dof; ++j) {
            for(CountT k = 0; k < num_dof; ++k) {
                // (k, j) should be (j, k)
                jm_inv_entry(j, i) += j_entry(k, i) * M_inv_entry(j, k);
            }
        }
    }

    // Then, compute A = J_c M^{-1} J_c^T
    CountT A_size = (3 * total_contacts) * (3 * total_contacts);
    float *A_ptr = (float *)state_mgr->tmpAlloc(
            world_id, sizeof(float) * A_size);
    memset(A_ptr, 0, sizeof(float) * A_size);
    auto A_entry = [A_ptr, total_contacts](int col, int row) -> float & {
        assert(row < 3 * total_contacts);
        assert(col < 3 * total_contacts);

        return A_ptr[row + col * (3 * total_contacts)];
    };
    for(CountT i = 0; i < 3 * total_contacts; ++i) {
        for(CountT j = 0; j < 3 * total_contacts; ++j) {
            for(CountT k = 0; k < num_dof; ++k) {
                A_entry(j, i) += jm_inv_entry(k, i) * j_entry(k, j);
            }
        }
    }

    // Compute v0 = J_c (v + h M^{-1} b)
    // First compute v + h M^{-1} b
    float *v_h_M_inv_b = (float *)state_mgr->tmpAlloc(
            world_id, sizeof(float) * 6 * num_bodies);
    for(CountT i = 0; i < 6 * num_bodies; ++i) {
        v_h_M_inv_b[i] = vptr[i];
        for(CountT j = 0; j < 6 * num_bodies; ++j) {
            v_h_M_inv_b[i] += physics_state.h * M_inv_entry(j, i) * btr[j];
        }
    }

    // Finally, compute v0 (multiply by J_c)
    float *v0 = (float *)state_mgr->tmpAlloc(
            world_id, sizeof(float) * 3 * total_contacts);
    for(CountT i = 0; i < 3 * total_contacts; ++i) {
        v0[i] = 0.f;
        for(CountT j = 0; j < 6 * num_bodies; ++j) {
            v0[i] += j_entry(j, i) * v_h_M_inv_b[j];
        }
    }


    // Start building f_C
    float *f_C = (float *)state_mgr->tmpAlloc(
            world_id, sizeof(float) * 3 * total_contacts);
    memset(f_C, 0, sizeof(float) * 3 * total_contacts);

    // Begin solving for f_C
    for(CountT i = 0; i < 3 * total_contacts; i += 3) {
        f_C[i] = 10.f; //init guess: TODO: make this smarter
    }

    float *mu_tmp_array = (float *)state_mgr->tmpAlloc(
            world_id, sizeof(float) * total_contacts);
    float *penetration_tmp_array = (float *)state_mgr->tmpAlloc(
            world_id, sizeof(float) * total_contacts);
    for (int i = 0; i < total_contacts; ++i) {
        uint32_t parent = contact_point_info[i].parentIdx;
        mu_tmp_array[i] = contacts_tmp_state[parent].mu;
        penetration_tmp_array[i] = contacts_tmp_state[parent].maxPenetration;
    }

    if (cv_sing.cvxSolve && cv_sing.cvxSolve->fn) {
        cv_sing.cvxSolve->aPtr = A_ptr;
        cv_sing.cvxSolve->aRows = 3 * total_contacts;
        cv_sing.cvxSolve->aCols = 3 * total_contacts;
        cv_sing.cvxSolve->v0Ptr = v0; 
        cv_sing.cvxSolve->v0Rows = 3 * total_contacts;
        cv_sing.cvxSolve->muPtr = mu_tmp_array;
        cv_sing.cvxSolve->penetrationsPtr = penetration_tmp_array;
        cv_sing.cvxSolve->fcRows = 3 * total_contacts;

        cv_sing.cvxSolve->callSolve.store_release(1);
        while (cv_sing.cvxSolve->callSolve.load_acquire() != 2);
        cv_sing.cvxSolve->callSolve.store_relaxed(0);

        float *res = cv_sing.cvxSolve->resPtr;

#if 0
        float* res = cv_sing.cvxSolve->fn(
                cv_sing.cvxSolve->data, 
                A_ptr, 3 * total_contacts, 3 * total_contacts,
                v0, 3 * total_contacts,
                mu_tmp_array,
                penetration_tmp_array,
                3 * total_contacts);
#endif

        if (res) {
            for(CountT i = 0; i < 3 * total_contacts; ++i) {
                f_C[i] = res[i];
            }
        }
    }

    float* g = (float *)state_mgr->tmpAlloc(
            world_id, sizeof(float) * 3 * total_contacts);

    // Populates gradient and returns the norm
    float *v_C = (float *)state_mgr->tmpAlloc(world_id, sizeof(float) * total_contacts * 3);

    auto grad = [&] (float *gradient, const float* f, const float kappa) -> float {
        memset(v_C, 0, sizeof(float) * total_contacts * 3);

        // v_C = A f_C + v0
        for(CountT i = 0; i < 3 * total_contacts; ++i) {
            v_C[i] = v0[i];
            for(CountT j = 0; j < 3 * total_contacts; ++j) {
                v_C[i] += A_entry(j, i) * f[j];
            }
        }

        // Gradient of main objective is v_C
        for(CountT i = 0; i < 3 * total_contacts; ++i) {
            gradient[i] = v_C[i];
        }
        // Constraint barriers
        for(CountT i = 0; i < total_contacts; ++i)
        {
            ContactPointInfo &pt_info = contact_point_info[i];
            ContactTmpState &contact_tmp_state = contacts_tmp_state[pt_info.parentIdx];
            CountT idx = 3 * i;
            // Constraint 1: positivity at normals
            float s1 = f[idx];
            gradient[idx] += -kappa * (1 / s1);

            // Second constraint - friction cone
            float mu = contact_tmp_state.mu;
            float s2 = (mu * mu * f[idx] * f[idx]) - f[idx + 1] * f[idx + 1] - f[idx + 2] * f[idx + 2];
            gradient[idx] += -kappa * (2 * (mu * mu) * f[idx]) / s2;
            gradient[idx + 1] += kappa * (2 * f[idx + 1]) / s2;
            gradient[idx + 2] += kappa * (2 * f[idx + 2]) / s2;
            // // Third constraint - avoid penetration
            // // float pen_depth = contact_tmp_state.maxDepth / physics_state.h;
            for(CountT j = 0; j < 3 * total_contacts; ++j) {
                gradient[j] += -kappa * A_entry(idx, j) / (v_C[idx]);
            }
        }
        float norm = 0.f;
        for(CountT i = 0; i < 3 * total_contacts; ++i) {
            norm += gradient[i] * gradient[i];
        }
        return sqrt(norm);
    };

    // Naive gradient descent
    // float kappa = 1.f;
    // while(kappa > 0.0000001f) {
    //     for(CountT gd_iter = 0; gd_iter < 1000; ++gd_iter)
    //     {
    //         for(CountT i = 0; i < 3 * total_contacts; ++i) {
    //             f_C[i] -= 0.01f * g[i];
    //         }
    //         grad(g, f_C, kappa);
    //     }
    //     kappa /= 10.f;
    // }
    //
    // Divide by h (counteract the multiplication by h in the velocity update)
    for(CountT i = 0; i < 3 * total_contacts; ++i) {
        f_C[i] /= physics_state.h;
    }

    // Post-solve f_C. Impulse is J_c^T f_C
    float *contact_force = (float *)state_mgr->tmpAlloc(
            world_id, sizeof(float) * num_dof);
    for(CountT i = 0; i < num_dof; ++i) {
        contact_force[i] = 0.f;
        for(CountT j = 0; j < 3 * total_contacts; ++j) {
            contact_force[i] += j_entry(i, j) * f_C[j];
        }
    }

    // Add to bodies
    for(CountT i = 0; i < num_bodies; ++i) {
        DofObjectTmpState &body_tmp_state = tmp_states[i];
        CountT idx_start = i * 6;

        if(body_tmp_state.invMass == 0) {
            continue;
        }

        body_tmp_state.externalForces += Vector3(contact_force[idx_start],
                                                contact_force[idx_start + 1],
                                                contact_force[idx_start + 2]);
        body_tmp_state.externalMoment += Vector3(contact_force[idx_start + 3],
                                                contact_force[idx_start + 4],
                                                contact_force[idx_start + 5]);
    }
}
#endif

static void integrationStep(Context &ctx,
                            DofObjectPosition &position,
                            DofObjectVelocity &velocity,
                            DofObjectNumDofs &numDofs,
                            DofObjectTmpState &tmp_state,
                            ObjectID &objID)
{
#if 0
    PhysicsSystemState &physics_state = ctx.singleton<PhysicsSystemState>();

    // Phase 1: compute the mass matrix data

    Quat rot_quat = {
        position.q[3],
        position.q[4],
        position.q[5],
        position.q[6],
    };

    float invMass = tmp_state.invMass;
    Mat3x3 inv_I_world_frame = tmp_state.invInertia;

    Vector3 trans_forces = tmp_state.externalForces;
    Vector3 rot_moments = tmp_state.externalMoment;

    Vector3 delta_v = invMass * trans_forces;
    Vector3 delta_omega = inv_I_world_frame * rot_moments;
    float h = physics_state.h;
    for (int i = 0; i < 3; ++i) {
        velocity.qv[i] += h * delta_v[i];
    }
    for (int i = 3; i < 6; ++i) {
        velocity.qv[i] += h * delta_omega[i - 3];
    }

    // Step N: Integrate position
    for (int i = 0; i < 3; ++i) {
        position.q[i] += h * velocity.qv[i];
    }

    // From angular velocity to quaternion [Q_w, Q_x, Q_y, Q_z]
    Vector3 omega = { velocity.qv[3], velocity.qv[4], velocity.qv[5] };
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
#endif
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

    registry.registerBundle<CVRigidBodyState>();
    registry.registerBundleAlias<SolverBundleAlias, CVRigidBodyState>();
}

void makeCVPhysicsEntity(Context &ctx, Entity e,
                         Position position,
                         Rotation rotation,
                         ObjectID obj_id,
                         DofType dof_type)
{
    CVHierarchyCounter &hier_counter = ctx.singleton<CVHierarchyCounter>();
    Entity physical_entity = ctx.makeEntity<DofObjectArchetype>();

    auto &pos = ctx.get<DofObjectPosition>(physical_entity);
    auto &vel = ctx.get<DofObjectVelocity>(physical_entity);

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

    auto &hierarchy = ctx.get<DofObjectHierarchyDesc>(physical_entity);
    hierarchy.sync.store_relaxed(0);
    hierarchy.leaf = true;
    hierarchy.numbering = hier_counter.num_bodies++;

#ifdef MADRONA_GPU_MODE
    static_assert(false, "Need to implement GPU DOF object hierarchy")
#else
    // By default, no parent
    hierarchy.parent = Entity::none();
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
                DofObjectPosition,
                DofObjectNumDofs,
                DofObjectTmpState,
                DofObjectHierarchyDesc
            >>({run_narrowphase});

        auto compute_spatial_inertia = builder.addToGraph<ParallelForNode<Context,
             tasks::computeSpatialInertia,
                DofObjectNumDofs,
                DofObjectTmpState,
                ObjectID
            >>({forward_kinematics});

        auto composite_rigid_body = builder.addToGraph<ParallelForNode<Context,
             tasks::compositeRigidBody,
                CVSingleton
            >>({compute_spatial_inertia});

        auto contact_node = builder.addToGraph<ParallelForNode<Context,
             tasks::processContacts,
                ContactConstraint,
                ContactTmpState
            >>({composite_rigid_body});

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
                            Entity parent, Entity child,
                            Vector3 rel_pos_parent,
                            Vector3 rel_pos_child,
                            Vector3 hinge_axis)
{
    Entity child_physics_entity =
        ctx.get<CVPhysicalComponent>(child).physicsEntity;
    Entity parent_physics_entity =
        ctx.get<CVPhysicalComponent>(parent).physicsEntity;

#ifdef MADRONA_GPU_MODE
    static_assert(false, "Need to implement GPU DOF object hierarchy");
#else
    auto &hier_desc = ctx.get<DofObjectHierarchyDesc>(child_physics_entity);
    hier_desc.parent = parent_physics_entity;
    hier_desc.relPositionParent = rel_pos_parent;
    hier_desc.relPositionLocal = rel_pos_child;
    hier_desc.hingeAxis = hinge_axis;
    hier_desc.leaf = true;

    // Make the parent no longer a leaf
    ctx.get<DofObjectHierarchyDesc>(parent_physics_entity).leaf = false;
#endif
}
    
}
