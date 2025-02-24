#include "cv.hpp"
#include "physics_impl.hpp"

namespace madrona::phys::cv {

void refreshPointers(Context &ctx,
                     BodyGroupMemory &m);

TaskGraphNodeID setupCVSolverTasks(TaskGraphBuilder &builder,
                                   TaskGraphNodeID broadphase,
                                   CountT num_substeps)
{
    auto cur_node = broadphase;

#ifdef MADRONA_GPU_MODE
    cur_node = builder.addToGraph<
        SortArchetypeNode<BodyGroupArchetype, WorldID>>({cur_node});
    cur_node = builder.addToGraph<
        SortArchetypeNode<DofObjectArchetype, WorldID>>({sort_sys});
    cur_node = builder.addToGraph<ResetTmpAllocNode>({sort_sys});
#endif

    for (CountT i = 0; i < num_substeps; ++i) {
        cur_node = narrowphase::setupTasks(builder, {cur_node});

#ifdef MADRONA_GPU_MODE
        // We need to sort the contacts by world.
        cur_node = builder.addToGraph<
            SortArchetypeNode<Contact, WorldID>>(
                {cur_node});
        cur_node = builder.addToGraph<ResetTmpAllocNode>(
                {cur_node});
#endif

        cur_node = setupPrepareTasks(builder, cur_node);
        cur_node = setupSolveTasks(builder, cur_node);
        cur_node = setupPostTasks(builder, cur_node);
    }

    return cur_node;
}

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

void getSolverArchetypeIDs(uint32_t *contact_archetype_id,
                           uint32_t *joint_archetype_id)
{
    *contact_archetype_id = TypeTracker::typeID<Contact>();
    *joint_archetype_id = TypeTracker::typeID<Joint>();
}

void init(Context &ctx, CVXSolve *cvx_solve)
{
    ctx.singleton<CVSolveData>().cvxSolve = cvx_solve;
    // ctx.singleton<CVSolveData>().solverScratchMemory = MemoryRange::none();
    // ctx.singleton<CVSolveData>().accRefMemory = MemoryRange::none();
    ctx.singleton<CVSolveData>().scratchAllocatedBytes = 0;
    ctx.singleton<CVSolveData>().accRefAllocatedBytes = 0;
}

void registerTypes(ECSRegistry &registry)
{
    registry.registerSingleton<CVSolveData>();

    registry.registerComponent<DofObjectProxies>();
    registry.registerComponent<DofObjectGroup>();
    registry.registerComponent<BodyGroupMemory>();
    registry.registerComponent<BodyGroupProperties>();
    registry.registerComponent<ContactTmpState>();
    registry.registerComponent<LinkParentDofObject>();

    registry.registerArchetype<BodyGroupArchetype>();
    registry.registerArchetype<DofObjectArchetype>();
    registry.registerArchetype<Contact>();
    registry.registerArchetype<Joint>();
    registry.registerArchetype<LinkCollider>();
    registry.registerArchetype<LinkVisual>();

    registry.registerMemoryRangeElement<MRElement128b>();
    registry.registerMemoryRangeElement<SolverScratch256b>();
}

}
