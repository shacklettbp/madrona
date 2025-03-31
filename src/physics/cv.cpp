#include "cv.hpp"
#include "physics_impl.hpp"

namespace madrona::phys::cv {

namespace tasks {
void refreshPointers(Context &ctx,
                     BodyGroupMemory &m)
{
    m.qVectorsPtr = ctx.memoryRangePointer<MRElement128b>(m.qVectors);
    m.tmpPtr = ctx.memoryRangePointer<SolverScratch256b>(m.tmp);
}
}

TaskGraphNodeID setupCVSolverTasks(TaskGraphBuilder &builder,
                                   TaskGraphNodeID broadphase,
                                   CountT num_substeps)
{
    auto cur_node = broadphase;

    cur_node = builder.addToGraph<ParallelForNode<Context, 
             tasks::refreshPointers,
                BodyGroupMemory>>({cur_node});

    bool replay_mode = false;
    if (num_substeps == 0) {
        replay_mode = true;
        num_substeps = 1;
    }
    for (CountT i = 0; i < num_substeps; ++i) {
        if (!replay_mode) {
            cur_node = narrowphase::setupTasks(builder, {cur_node});
        }

#ifdef MADRONA_GPU_MODE
        cur_node = builder.addToGraph<
            CompactArchetypeNode<Contact>>({cur_node});
#endif

        if (!replay_mode) {
            cur_node = setupPrepareTasks(builder, cur_node);
            cur_node = setupSolveTasks(builder, cur_node);
        }

        cur_node = setupPostTasks(builder, cur_node, replay_mode);

        cur_node = builder.addToGraph<
            ClearTmpNode<Contact>>({cur_node});

        cur_node = builder.addToGraph<
            ResetTmpAllocNode>({cur_node});
    }

    cur_node = builder.addToGraph<
        ClearTmpNode<CandidateTemporary>>({cur_node});

#ifdef MADRONA_GPU_MODE
    cur_node = builder.addToGraph<
        SortMemoryRangeNode<MRElement128b>>({cur_node});
    cur_node = builder.addToGraph<ResetTmpAllocNode>({cur_node});

    cur_node = builder.addToGraph<
        SortMemoryRangeNode<SolverScratch256b, false>>({cur_node});
    cur_node = builder.addToGraph<ResetTmpAllocNode>({cur_node});
#endif

    cur_node = builder.addToGraph<ParallelForNode<Context, 
             tasks::refreshPointers,
                BodyGroupMemory>>({cur_node});

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
    ctx.singleton<CVSolveData>().enablePhysics = 1;
}

void registerTypes(ECSRegistry &registry)
{
    registry.registerSingleton<CVSolveData>();

    registry.registerComponent<DofObjectProxies>();
    registry.registerComponent<DofObjectGroup>();
    registry.registerComponent<BodyGroupMemory>();
    registry.registerComponent<BodyGroupProperties>();
    registry.registerComponent<InitBodyGroup>();
    registry.registerComponent<DestroyBodyGroup>();
    registry.registerComponent<ContactTmpState>();
    registry.registerComponent<LinkParentDofObject>();

    registry.registerArchetype<BodyGroupArchetype>();
    registry.registerArchetype<DofObjectArchetype>();
    registry.registerArchetype<Contact>();
    registry.registerArchetype<Joint>();
    registry.registerArchetype<LinkCollider>();
    registry.registerArchetype<LinkVisual>();
    registry.registerArchetype<InitBodyGroupArchetype>();
    registry.registerArchetype<DestroyBodyGroupArchetype>();

    registry.registerMemoryRangeElement<MRElement128b>();
    registry.registerMemoryRangeElement<SolverScratch256b>();
}

void setEnablePhysics(Context &ctx, bool value)
{
    ctx.singleton<CVSolveData>().enablePhysics = true;
    // ctx.singleton<CVSolveData>().enablePhysics = value;
}

}
