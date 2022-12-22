#pragma once

#include <madrona/taskgraph.hpp>

namespace madrona {

template <typename ContextT, typename WorldDataT,
          auto Fn, typename ...ComponentTs>
struct ParallelForNode : NodeBase<
        ParallelForNode<ContextT, WorldDataT, Fn, ComponentTs...>> {
    ParallelForNode(QueryRef *query_ref)
        : queryRef(query_ref)
    {}

    inline void run(int32_t invocation_idx)
    {
        StateManager *state_mgr = mwGPU::getStateManager();

        int32_t cumulative_num_rows = 0;
        state_mgr->iterateArchetypesRaw<sizeof...(ComponentTs)>(queryRef,
                [&](int32_t num_rows, WorldID *world_column,
                    auto ...raw_ptrs) {
            int32_t tbl_offset = invocation_idx - cumulative_num_rows;
            cumulative_num_rows += num_rows;
            if (tbl_offset >= num_rows) {
                return false;
            }

            WorldID world_id = world_column[tbl_offset];

            // This entity has been deleted but not actually removed from the
            // table yet
            if (world_id.idx == -1) {
                return true;
            }

            ContextT ctx = ParallelForEntry::makeContext(world_id);


            // The following should work, but doesn't in cuda 11.7 it seems
            // Need to put arguments in tuple for some reason instead
            //Fn(ctx, ((ComponentTs *)raw_ptrs)[tbl_offset] ...);

            cuda::std::tuple typed_ptrs {
                (ComponentTs *)raw_ptrs
                ...
            };

            std::apply([&](auto ...ptrs) {
                Fn(ctx, ptrs[tbl_offset] ...);
            }, typed_ptrs);

            return true;
        });
    }

    inline int32_t numInvocations()
    {
        StateManager *state_mgr = mwGPU::getStateManager();
        return state_mgr->numMatchingEntities(queryRef);
    }

    static inline ContextT makeContext(WorldID world_id)
    {
        auto world = TaskGraph::getWorld(world_id.idx);
        return ContextT((WorldDataT *)world, WorkerInit {
            world_id,
        });
    }

    QueryRef queryRef;
};

struct ClearTmpNode : NodeBase<ClearTmpNode, 1, 0> {
    ClearTmpNode(uint32_t archetype_id)
        : archetypeID(archetype_id)
    {}

    void run(int32_t);

    uint32_t archetypeID;
};

struct RecycleEntitiesNode : NodeBase<RecycleEntitiesNode> {
    RecycleEntitiesNode() 
        : recycleBase(0)
    {}

    void run(int32_t invocation_idx);
    int32_t numInvocations();

    int32_t recycleBase;
};

struct CompactArchetypeEntry {
    static void run(EntryData &data, int32_t invocation_idx);
};

struct SortArchetypeEntry {
    struct Setup {
        static void run(EntryData &data, int32_t invocation_idx);
    };

    struct Run {
        static void run(EntryData &data, int32_t invocation_idx);
    };
};

struct RecycleEntitiesEntry {
    static void run(EntryData &data, int32_t invocation_idx);
};

struct ResetTmpAllocatorEntry {
    static void run(EntryData &, int32_t invocation_idx);
};

class NodeBuilder {
public:
    NodeBuilder(StateManager &state_mgr)
        : state_mgr_(&state_mgr)
    {}

    template <typename ContextT, auto Fn, typename ...ComponentTs>
    ParallelForNode<ContextT, mwGPU::WorldTypeExtract<ContextT>,
                    Fn, ComponentTs...>
    parallelFor()
    {
        using WorldT = typename mwGPU::WorldTypeExtract<ContextT>::type;

        auto query = mwGPU::getStateManager()->query<ComponentTs...>();
        QueryRef *query_ref = query.getSharedRef();
        query_ref->numReferences.fetch_add(1, std::memory_order_relaxed);

        return ParallelForNode<ContextT, WorldT, Fn, ComponentTs...>(
            query_ref);
    }

    template <typename ArchetypeT>
    ClearTmpNode clearTemporariesNode()
    {
        uint32_t archetype_id = TypeTracker::typeID<ArchetypeT>();

        return ClearTmpNode(archetype_id);
    }

    RecycleEntitiesNode recycleEntitiesNode();

private:
    StateManager *state_mgr_;
};

}
