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

struct CVSingleton {
    // Just to have something to loop over for CPU solver
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
static void gaussMinimizeFn(Context &ctx,
                            DofObjectPosition &position,
                            DofObjectVelocity &velocity,
                            DofObjectNumDofs &numDofs,
                            DofObjectTmpState &tmp_state,
                            ObjectID &objID)
{
    // TODO: Gather all the physics objects and do the minimization
    StateManager *state_mgr = ctx.getStateManager();
    ObjectManager &obj_mgr = *ctx.singleton<ObjectData>().mgr;
    PhysicsSystemState &physics_state = ctx.singleton<PhysicsSystemState>();
    // DofObjectPosition *positions = state_mgr->getWorldComponents<
    //     DofObjectArchetype, DofObjectPosition>(ctx.worldID().idx);

    // Phase 1: compute the mass matrix data
    RigidBodyMetadata &metadata = obj_mgr.metadata[objID.idx];
    Diag3x3 inv_inertia = Diag3x3::fromVec(metadata.mass.invInertiaTensor);
    Diag3x3 inertia = inv_inertia.inv();
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

    Mat3x3 body_jacob = Mat3x3::fromQuat(rot_quat);

    // Inertia and Inverse Inertia in world frame
    Mat3x3 I_world_frame = body_jacob * inertia;
    I_world_frame = body_jacob * I_world_frame.transpose();
    Mat3x3 inv_I_world_frame = body_jacob * inv_inertia;
    inv_I_world_frame = body_jacob * inv_I_world_frame.transpose();

    // Step 2: compute the external and gyroscopic forces
    // Step 2.1: gravity
    Vector3 force_gravity = physics_state.g / metadata.mass.invMass;
    Vector3 trans_forces = force_gravity;

    // Step 2.2: gyroscopic moments
    Vector3 gyro_moment = -(omega.cross(I_world_frame * omega));
    Vector3 rot_moments = gyro_moment;

    tmp_state.invInertia = inv_I_world_frame;
    tmp_state.externalForces = trans_forces;
    tmp_state.externalMoment = rot_moments;

#if 0
    // Step 3: Contact Jacobians
    // TODO!

#endif
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

    printf("ref: %f %f %f\n", ref_pos.q[0], ref_pos.q[1],
                              ref_pos.q[2]);
    printf("alt: %f %f %f\n", alt_pos.q[0], alt_pos.q[1],
                              alt_pos.q[2]);

    // Create a coordinate system for the contact
    Vector3 n = contact.normal.normalize();
    Vector3 t{};
    Vector3 s{};

    Vector3 x_axis = {1.f, 0.f, 0.f};
    if(n.cross(x_axis).length() > 0.01f) {
        t = n.cross(x_axis).normalize();
    } else {
        t = n.cross({0.f, 1.f, 0.f}).normalize();
    }
    s = n.cross(t).normalize();

    // Get the average contact
    float penetration_sum = 0.f;
    Vector3 avg_contact = Vector3::zero();
    for(CountT i = 0; i < contact.numPoints; ++i) {
        Vector4 point = contact.points[i];
        penetration_sum += point.w;
        avg_contact += point.w * point.xyz();
    }
    avg_contact /= penetration_sum;

    // Compute the relative positions of the contact
    Vector3 rRef = avg_contact - ref_com;
    Vector3 rAlt = avg_contact - alt_com;

    tmp_state.n = n;
    tmp_state.t = t;
    tmp_state.s = s;
    tmp_state.rRef = rRef;
    tmp_state.rAlt = rAlt;
}

static void integrationStep(Context &ctx,
                            DofObjectPosition &position,
                            DofObjectVelocity &velocity,
                            DofObjectNumDofs &numDofs,
                            DofObjectTmpState &tmp_state,
                            ObjectID &objID)
{
    StateManager *state_mgr = ctx.getStateManager();
    ObjectManager &obj_mgr = *ctx.singleton<ObjectData>().mgr;
    PhysicsSystemState &physics_state = ctx.singleton<PhysicsSystemState>();

    // Phase 1: compute the mass matrix data
    RigidBodyMetadata &metadata = obj_mgr.metadata[objID.idx];

    Quat rot_quat = {
        position.q[3],
        position.q[4],
        position.q[5],
        position.q[6],
    };

    Mat3x3 inv_I_world_frame = tmp_state.invInertia;

    Vector3 trans_forces = tmp_state.externalForces;
    Vector3 rot_moments = tmp_state.externalMoment;

    Vector3 delta_v = metadata.mass.invMass * trans_forces;
    Vector3 delta_omega = inv_I_world_frame * rot_moments;
    float h = physics_state.h;
    if (metadata.mass.invMass > 0) {
        for (int i = 0; i < 3; ++i) {
            velocity.qv[i] += h * delta_v[i];
        }
        for (int i = 3; i < 6; ++i) {
            velocity.qv[i] += h * delta_omega[i - 3];
        }
    }

    // Step N: Integrate position
    for (int i = 0; i < 3; ++i) {
        if (metadata.mass.invMass > 0) {
            position.q[i] += h * velocity.qv[i];
        }
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
}
#endif

// Convert all the generalized coordinates here.
static void convertPostSolve(
        Context &ctx,
        Position &position,
        Rotation &rotation,
        const CVPhysicalComponent &phys)
{
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
    } else {
        MADRONA_UNREACHABLE();
    }
}

}

void registerTypes(ECSRegistry &registry)
{
    registry.registerComponent<CVPhysicalComponent>();

    registry.registerSingleton<CVSingleton>();

    registry.registerComponent<DofObjectPosition>();
    registry.registerComponent<DofObjectVelocity>();
    registry.registerComponent<DofObjectNumDofs>();
    registry.registerComponent<DofObjectTmpState>();
    registry.registerComponent<ContactTmpState>();

    registry.registerArchetype<DofObjectArchetype>();
    registry.registerArchetype<Contact>();
    registry.registerArchetype<Joint>();

    registry.registerBundle<CVRigidBodyState>();
    registry.registerBundleAlias<SolverBundleAlias, CVRigidBodyState>();
}

void makeFreeBodyEntityPhysical(Context &ctx, Entity e,
                                Position position,
                                Rotation rotation,
                                base::ObjectID obj_id) {
    Entity physical_entity = ctx.makeEntity<DofObjectArchetype>();

    auto &pos = ctx.get<DofObjectPosition>(physical_entity);

    pos.q[0] = position.x;
    pos.q[1] = position.y;
    pos.q[2] = position.z;

    pos.q[3] = rotation.w;
    pos.q[4] = rotation.x;
    pos.q[5] = rotation.y;
    pos.q[6] = rotation.z;

    auto &vel = ctx.get<DofObjectVelocity>(physical_entity);

    vel.qv[0] = 0.f;
    vel.qv[1] = 0.f;
    vel.qv[2] = 0.f;
    vel.qv[3] = 0.f;
    vel.qv[4] = 0.f;
    vel.qv[5] = 0.f;
    ctx.get<base::ObjectID>(physical_entity) = obj_id;

    ctx.get<DofObjectNumDofs>(physical_entity).numDofs = 6;

    ctx.get<CVPhysicalComponent>(e) = {
        .physicsEntity = physical_entity,
    };
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
        auto gauss_node = builder.addToGraph<ParallelForNode<Context,
             tasks::gaussMinimizeFn,
                DofObjectPosition,
                DofObjectVelocity,
                DofObjectNumDofs,
                DofObjectTmpState,
                ObjectID
            >>({run_narrowphase});

        auto contact_node = builder.addToGraph<ParallelForNode<Context,
             tasks::processContacts,
                ContactConstraint,
                ContactTmpState
            >>({gauss_node});

        auto int_node = builder.addToGraph<ParallelForNode<Context,
             tasks::integrationStep,
                 DofObjectPosition,
                 DofObjectVelocity,
                 DofObjectNumDofs,
                 DofObjectTmpState,
                 ObjectID
            >>({contact_node});
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

void init(Context &ctx)
{
    // Nothing for now
    (void)ctx;
}
    
}
