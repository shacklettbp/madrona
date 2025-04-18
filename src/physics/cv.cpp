#include "cv.hpp"
#include <madrona/sync.hpp>
#include "physics_impl.hpp"

#ifdef CV_COUNT_GPU_CLOCKS
extern "C" {
DEFINE_STAGE_VARS(com);
DEFINE_STAGE_VARS(inertias);
DEFINE_STAGE_VARS(rne);
DEFINE_STAGE_VARS(crb);
DEFINE_STAGE_VARS(invMass);
DEFINE_STAGE_VARS(processContacts);
DEFINE_STAGE_VARS(convert);
DEFINE_STAGE_VARS(destroy);
DEFINE_STAGE_VARS(init);
DEFINE_STAGE_VARS(damp);
DEFINE_STAGE_VARS(intg);
DEFINE_STAGE_VARS(fk);
DEFINE_STAGE_VARS(narrowphase);
DEFINE_STAGE_VARS(broadphase1);
DEFINE_STAGE_VARS(broadphase2);
DEFINE_STAGE_VARS(allocScratch);
DEFINE_STAGE_VARS(prepSolver);
DEFINE_STAGE_VARS(contAccRef);
DEFINE_STAGE_VARS(eqAccRef);
DEFINE_STAGE_VARS(cg);
DEFINE_STAGE_VARS(lineSearch);
}
#endif

namespace madrona::phys::cv {

namespace tasks {
#ifdef CV_COUNT_GPU_CLOCKS
inline void reportPhysicsClocks(Context &ctx,
                                PhysicsSystemState &state)
{
    uint64_t total_clocks = 0;
    uint64_t count = 0;

    uint64_t cvNumFrames = state.cvNumFrames;

    #define CV_PREP_TOTAL(name) total_clocks += cv##name .load< sync::relaxed >(); \
                                count++;

    #define CV_REPORT_CLOCK(name) printf("\t- " #name " = %lu (%lf)\n", cv##name .load< sync::relaxed >(),\
                                        (double)(cv##name .load< sync::relaxed >()) / (double)total_clocks); \
                                  cv##name .store< sync::relaxed >(0);

    cvNumFrames++;

    #define CV_RUNNING_AVG(name) cv##name##_avg += (uint64_t) \
        ((double)(cv##name .load<sync::relaxed>() - cv##name##_avg) / (double)cvNumFrames); \
        total_clocks += cv##name##_avg; \
        count++;

    #define CV_REPORT_AVG_CLOCK(name) printf("\t- " #name " = %lu (%lf)\n", cv##name##_avg,\
                                        (double)(cv##name##_avg) / (double)total_clocks); \
                                  cv##name .store< sync::relaxed >(0);



    if (ctx.worldID().idx != 0) {
        return;
    }

    if (threadIdx.x == 0 && ctx.worldID().idx == 0) {
        printf("Reporting physics clocks:\n");

        CV_RUNNING_AVG(com);
        CV_RUNNING_AVG(inertias);
        CV_RUNNING_AVG(rne);
        CV_RUNNING_AVG(crb);
        CV_RUNNING_AVG(invMass);
        CV_RUNNING_AVG(processContacts);
        CV_RUNNING_AVG(convert);
        CV_RUNNING_AVG(destroy);
        CV_RUNNING_AVG(init);
        CV_RUNNING_AVG(damp);
        CV_RUNNING_AVG(intg);
        CV_RUNNING_AVG(fk);
        CV_RUNNING_AVG(narrowphase);
        CV_RUNNING_AVG(broadphase1);
        CV_RUNNING_AVG(broadphase2);
        CV_RUNNING_AVG(allocScratch);
        CV_RUNNING_AVG(prepSolver);
        CV_RUNNING_AVG(contAccRef);
        CV_RUNNING_AVG(eqAccRef);
        CV_RUNNING_AVG(cg);
        CV_RUNNING_AVG(lineSearch);

        CV_REPORT_AVG_CLOCK(com);
        CV_REPORT_AVG_CLOCK(inertias);
        CV_REPORT_AVG_CLOCK(rne);
        CV_REPORT_AVG_CLOCK(crb);
        CV_REPORT_AVG_CLOCK(invMass);
        CV_REPORT_AVG_CLOCK(processContacts);
        CV_REPORT_AVG_CLOCK(convert);
        CV_REPORT_AVG_CLOCK(destroy);
        CV_REPORT_AVG_CLOCK(init);
        CV_REPORT_AVG_CLOCK(damp);
        CV_REPORT_AVG_CLOCK(intg);
        CV_REPORT_AVG_CLOCK(fk);
        CV_REPORT_AVG_CLOCK(narrowphase);
        CV_REPORT_AVG_CLOCK(broadphase1);
        CV_REPORT_AVG_CLOCK(broadphase2);
        CV_REPORT_AVG_CLOCK(allocScratch);
        CV_REPORT_AVG_CLOCK(prepSolver);
        CV_REPORT_AVG_CLOCK(contAccRef);
        CV_REPORT_AVG_CLOCK(eqAccRef);
        CV_REPORT_AVG_CLOCK(cg);
        CV_REPORT_AVG_CLOCK(lineSearch);
    }
}
#endif

void refreshPointers(Context &ctx,
                     BodyGroupMemory &m)
{
    m.qVectorsPtr = ctx.memoryRangePointer<MRElement128b>(m.qVectors);
    m.tmpPtr = ctx.memoryRangePointer<SolverScratch256b>(m.tmp);

#if 0
    printf("tmp_ptr = %p; vectors_ptr = %p\n",
            m.tmpPtr, m.qVectorsPtr);
#endif
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
        } else {
            // Body groups should now all be set up at this point
            cur_node = builder.addToGraph<
                ClearTmpNode<InitBodyGroupArchetype>>({cur_node});
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

#ifdef CV_COUNT_GPU_CLOCKS
    cur_node = builder.addToGraph<ParallelForNode<Context,
             tasks::reportPhysicsClocks,
                PhysicsSystemState>>({cur_node});
#endif

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
    ctx.singleton<PhysicsSystemState>().cvNumFrames = 0;
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
