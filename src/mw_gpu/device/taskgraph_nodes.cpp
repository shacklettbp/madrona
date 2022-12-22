#include <madrona/taskgraph.hpp>

namespace madrona {

void ClearTmpNode::run(int32_t)
{
    StateManager *state_mgr = mwGPU::getStateManager();
    state_mgr->clearTemporaries(archetypeID);
}

// Key question:
// Are we specializing too much on compute count => then
// fork? Is there a way to get this same functionality
// with a more generic mechanism to run multiple nodes
// over the same data?

void RecycleEntitiesNode::run(int32_t invocation_idx)
{
    mwGPU::getStateManager()->recycleEntities(
        invocation_idx, recycleBase);
}

void RecycleEntitiesNode::numInvocations()
{
    auto [recycle_base, num_deleted] =
        mwGPU::getStateManager()->fetchRecyclableEntities();

    if (num_deleted > 0) {
        recycleBase = recycle_base;
    }

    return num_deleted;
}

RecycleEntitiesNode NodeBuilder::recycleEntitiesNode()
{
    return RecycleEntitiesNode();
}

void CompactArchetypeEntry::run(EntryData &data, int32_t invocation_idx)
{
#if 0
    uint32_t archetype_id = data.compactArchetype.archetypeID;
    StateManager *state_mgr = mwGPU::getStateManager();
#endif

    // Actually compact
    assert(false);
}

void SortArchetypeEntry::Setup::run(EntryData &data,
                                           int32_t invocation_idx)
{
    StateManager *state_mgr = getStateManager();
    state_mgr->archetypeSetupSortState(
        data.sortArchetypeSetup.archetypeID,
        data.sortArchetypeSetup.columnIDX,
        data.sortArchetypeSetup.numPasses);
}

void SortArchetypeEntry::Run::run(EntryData &data,
                                               int32_t invocation_idx)
{
    uint32_t archetype_id = data.sortArchetype.archetypeID;
    StateManager *state_mgr = getStateManager();

    state_mgr->sortArchetype(archetype_id, invocation_idx);
}

void RecycleEntitiesEntry::run(EntryData &data, int32_t invocation_idx)
{
}

void ResetTmpAllocatorEntry::run(EntryData &, int32_t invocation_idx)
{
    TmpAllocator::get().reset();
}

}
