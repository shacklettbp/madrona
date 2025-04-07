#include "cv.hpp"
#include <madrona/sync.hpp>
#include "physics_impl.hpp"

#ifdef CV_COUNT_GPU_CLOCKS
extern "C" {
madrona::AtomicU64 cvinertias = 0;
madrona::AtomicU64 cvrne = 0;
madrona::AtomicU64 cvcrb = 0;
madrona::AtomicU64 cvinvMass = 0;
madrona::AtomicU64 cvprocessContacts = 0;
madrona::AtomicU64 cvconvert = 0;
madrona::AtomicU64 cvdestroy = 0;
madrona::AtomicU64 cvinit = 0;
madrona::AtomicU64 cvintg = 0;
madrona::AtomicU64 cvfk = 0;
madrona::AtomicU64 cvnarrowphase = 0;
madrona::AtomicU64 cvallocScratch = 0;
madrona::AtomicU64 cvprepSolver = 0;
madrona::AtomicU64 cvcontAccRef = 0;
madrona::AtomicU64 cveqAccRef = 0;
madrona::AtomicU64 cvcg = 0;
}
#endif

namespace madrona::phys::cv {

namespace tasks {
#ifdef CV_COUNT_GPU_CLOCKS
inline void reportPhysicsClocks(Context &ctx,
                                PhysicsSystemState &)
{
    uint64_t total_clocks = 0;
    uint64_t count = 0;

    #define CV_PREP_TOTAL(name) total_clocks += cv##name .load< sync::relaxed >(); \
                                count++;

    #define CV_REPORT_CLOCK(name) printf("\t- " #name " = %lu (%lf)\n", cv##name .load< sync::relaxed >(),\
                                        (double)(cv##name .load< sync::relaxed >()) / (double)total_clocks); \
                                  cv##name .store< sync::relaxed >(0);

    if (ctx.worldID().idx != 0) {
        return;
    }

    if (threadIdx.x == 0 && ctx.worldID().idx == 0) {
        printf("Reporting physics clocks:\n");

        CV_PREP_TOTAL(com);
        CV_PREP_TOTAL(inertias);
        CV_PREP_TOTAL(rne);
        CV_PREP_TOTAL(crb);
        CV_PREP_TOTAL(invMass);
        CV_PREP_TOTAL(processContacts);
        CV_PREP_TOTAL(convert);
        CV_PREP_TOTAL(destroy);
        CV_PREP_TOTAL(init);
        CV_PREP_TOTAL(intg);
        CV_PREP_TOTAL(fk);
        CV_PREP_TOTAL(narrowphase);
        CV_PREP_TOTAL(broadphase1);
        CV_PREP_TOTAL(broadphase2);
        CV_PREP_TOTAL(allocScratch);
        CV_PREP_TOTAL(prepSolver);
        CV_PREP_TOTAL(contAccRef);
        CV_PREP_TOTAL(eqAccRef);
        CV_PREP_TOTAL(cg);

        CV_REPORT_CLOCK(com);
        CV_REPORT_CLOCK(inertias);
        CV_REPORT_CLOCK(rne);
        CV_REPORT_CLOCK(crb);
        CV_REPORT_CLOCK(invMass);
        CV_REPORT_CLOCK(processContacts);
        CV_REPORT_CLOCK(convert);
        CV_REPORT_CLOCK(destroy);
        CV_REPORT_CLOCK(init);
        CV_REPORT_CLOCK(intg);
        CV_REPORT_CLOCK(fk);
        CV_REPORT_CLOCK(narrowphase);
        CV_REPORT_CLOCK(broadphase1);
        CV_REPORT_CLOCK(broadphase2);
        CV_REPORT_CLOCK(allocScratch);
        CV_REPORT_CLOCK(prepSolver);
        CV_REPORT_CLOCK(contAccRef);
        CV_REPORT_CLOCK(eqAccRef);
        CV_REPORT_CLOCK(cg);
    }
}
#endif

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
