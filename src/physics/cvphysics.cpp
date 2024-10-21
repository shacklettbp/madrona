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
    ContactConstraint
> {};

struct Joint : Archetype<
    JointConstraint
> {};

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

    // This is generated from narrowphase.
    ContactConstraint *contacts;

    // World offsets of the contact constraints.
    int32_t *contactWorldOffsets;

    // World offsets of the positions / velocities.
    int32_t *dofObjectWorldOffsets;
};

GaussMinimizationNode::GaussMinimizationNode(
        StateManager *s)
    : positions(s->getArchetypeComponent<DofObjectArchetype, 
            DofObjectPosition>()),
      velocities(s->getArchetypeComponent<DofObjectArchetype,
            DofObjectVelocity>()),
      numDofs(s->getArchetypeComponent<DofObjectArchetype,
            DofObjectNumDofs>()),
      contactWorldOffsets(s->getArchetypeWorldOffsets<Contact>()),
      dofObjectWorldOffsets(s->getArchetypeWorldOffsets<DofObjectArchetype>())
{
}

void GaussMinimizationNode::solve(int32_t invocation_idx)
{
    // TODO: do the magic here!
    // Note, the refs that are in the ContactConstraint are not to the
    // DOF object but to the entity which holds the position / rotation
    // components. This just means we have to inefficiently go through another
    // indirection to get the Dof positions / velocities of affected objects
    // (with that indirection being CVPhysicalComponent).
    // We can fix this by having the contact generation directly output
    // the locs of the DOF objects instead but that we can do in the future.
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
    assert(blockDim.x == consts::numMegakernelThreads);

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
                            DofObjectPosition &pos,
                            DofObjectVelocity &vel,
                            const DofObjectNumDofs &num_dofs)
{
    (void)ctx;
    (void)pos;
    (void)vel;
    (void)num_dofs;
}
#endif

// Convert all the generalized coordinates here.
static void convertPostSolve(
        Context &ctx,
        Position &position,
        Rotation &rotation,
        const CVPhysicalComponent &phys)
{
    Entity physical_entity = ctx.makeEntity<DofObjectArchetype>();
    
    DofObjectNumDofs num_dofs = ctx.get<DofObjectNumDofs>(physical_entity);
    DofObjectPosition pos = ctx.get<DofObjectPosition>(physical_entity);

    if (num_dofs.numDofs == (uint32_t)DofType::FreeBody) {
        position.x = pos.q[0];
        position.y = pos.q[1];
        position.z = pos.q[2];

        rotation = Quat::fromAngularVec(
            Vector3{
                pos.q[3],
                pos.q[4],
                pos.q[5] 
            });
    } else {
        MADRONA_UNREACHABLE();
    }
}

}
    
void registerTypes(ECSRegistry &registry)
{
    registry.registerComponent<CVPhysicalComponent>();

    registry.registerComponent<DofObjectPosition>();
    registry.registerComponent<DofObjectVelocity>();
    registry.registerComponent<DofObjectNumDofs>();

    registry.registerArchetype<DofObjectArchetype>();
    registry.registerArchetype<Contact>();
    registry.registerArchetype<Joint>();
}

void makeFreeBodyEntityPhysical(Context &ctx, Entity e,
                                Position position,
                                Rotation rotation)
{
    Entity physical_entity = ctx.makeEntity<DofObjectArchetype>();

    auto &pos = ctx.get<DofObjectPosition>(physical_entity);

    pos.q[0] = position.x;
    pos.q[1] = position.y;
    pos.q[2] = position.z;

    Vector3 pyr = rotation.extractPYR();

    pos.q[3] = pyr.x;
    pos.q[4] = pyr.x;
    pos.q[5] = pyr.x;

    auto &vel = ctx.get<DofObjectVelocity>(physical_entity);

    vel.qv[0] = 0.f;
    vel.qv[1] = 0.f;
    vel.qv[2] = 0.f;
    vel.qv[3] = 0.f;
    vel.qv[4] = 0.f;
    vel.qv[5] = 0.f;

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
#else
        auto gauss_node = builder.addToGraph<ParallelForNode<Context,
             tasks::gaussMinimizeFn,
                DofObjectPosition,
                DofObjectVelocity,
                DofObjectNumDofs
            >>({run_narrowphase});
#endif

        cur_node =
            builder.addToGraph<ParallelForNode<Context, tasks::convertPostSolve,
                Position,
                Rotation,
                CVPhysicalComponent
            >>({gauss_node});
    }
    
    return cur_node;
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
