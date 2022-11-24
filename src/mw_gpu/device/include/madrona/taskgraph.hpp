#pragma once

#include <madrona/span.hpp>
#include <madrona/query.hpp>

#include <madrona/state.hpp>

#include "mw_gpu/const.hpp"
#include "mw_gpu/worker_init.hpp"

#include <cuda/barrier>
#include <cuda/std/tuple>

namespace madrona {

class TaskGraph;

namespace mwGPU {

struct EntryData {
    struct ParallelFor {
        QueryRef *query;
    };

    union {
        ParallelFor parallelFor;
    };
};

template <typename EntryT>
__attribute__((used, always_inline))
inline void userEntry(EntryData &entry_data, int32_t invocation_idx)
{
    EntryT::run(entry_data, invocation_idx);
}

template <typename EntryT>
struct UserFuncIDBase {
    static uint32_t id;
};

template <typename EntryT,
          decltype(userEntry<EntryT>) = userEntry<EntryT>>
struct UserFuncID : UserFuncIDBase<EntryT> {};

template <typename ContextT, typename WorldDataT>
struct EntryBase {
    static inline ContextT makeContext(WorldID world_id)
    {
        auto world = TaskGraph::getWorld(world_id.idx);
        return ContextT((WorldDataT *)world, WorkerInit {
            world_id,
        });
    }
};

template <typename ContextT, typename WorldDataT,
          auto Fn, typename ...ComponentTs>
struct ParallelForEntry : public EntryBase<ContextT, WorldDataT> {
    static inline void run(EntryData &data,
                           int32_t invocation_idx)
    {
        QueryRef *query_ref = data.parallelFor.query;
        StateManager *state_mgr = mwGPU::getStateManager();

        int32_t cumulative_num_rows = 0;
        state_mgr->iterateArchetypesRaw<sizeof...(ComponentTs)>(query_ref,
                [&](int32_t num_rows, WorldID *world_column,
                    auto ...raw_ptrs) {
            int32_t tbl_offset = invocation_idx - cumulative_num_rows;
            cumulative_num_rows += num_rows;
            if (tbl_offset >= num_rows) {
                return false;
            }

            WorldID world_id = world_column[tbl_offset];

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
};

}

struct NodeID {
    uint32_t id;
};

class TaskGraph {
private:
    enum class NodeType {
        ParallelFor,
    };

    struct NodeInfo {
        NodeType type;
        uint32_t funcID;
        mwGPU::EntryData data;
    };

    struct NodeState {
        NodeInfo info;
        std::atomic_uint32_t curOffset;
        std::atomic_uint32_t numRemaining;
        std::atomic_uint32_t totalNumInvocations;
    };

public:
    class Builder {
    public:
        Builder(uint32_t max_num_nodes,
                uint32_t max_num_dependencies);
        ~Builder();

        template <typename ContextT, auto Fn, typename ...ComponentTs>
        inline NodeID parallelForNode(Span<const NodeID> dependencies)
        {
            using WorldDataT = typename ContextT::WorldDataT;

            using Entry = typename mwGPU::ParallelForEntry<
                ContextT, WorldDataT, Fn, ComponentTs...>;
            uint32_t func_id = mwGPU::UserFuncID<Entry>::id;

            auto query = mwGPU::getStateManager()->query<ComponentTs...>();
            QueryRef *query_ref = query.getSharedRef();
            query_ref->numReferences.fetch_add(1, std::memory_order_relaxed);

            registerNode(NodeInfo {
                .type = NodeType::ParallelFor,
                .funcID = func_id,
                .data = {
                    .parallelFor = {
                        query_ref,
                    },
                },
            }, dependencies);
        }

        void build(TaskGraph *out);

    private:
        NodeID registerNode(const NodeInfo &node_info,
                            Span<const NodeID> dependencies);

        struct StagedNode {
            NodeInfo node;
            uint32_t dependencyOffset;
            uint32_t numDependencies;
        };

        StagedNode *nodes_;
        uint32_t num_nodes_;
        NodeID *all_dependencies_;
        uint32_t num_dependencies_;
        uint32_t num_worlds_;
    };

    enum class WorkerState {
        Run,
        PartialRun,
        Loop,
        Exit,
    };

    TaskGraph(const TaskGraph &) = delete;
    ~TaskGraph();

    void init();

    WorkerState getWork(mwGPU::EntryData **entry_data,
                        uint32_t *run_func_id,
                        int32_t *run_offset);

    void finishWork();

    static inline WorldBase * getWorld(int32_t world_idx)
    {
        const auto &consts = mwGPU::GPUImplConsts::get();
        auto world_ptr = (char *)consts.worldDataAddr +
            world_idx * (int32_t)consts.numWorldDataBytes;

        return (WorldBase *)world_ptr;
    }

    struct BlockState;
private:
    TaskGraph(NodeState *nodes, uint32_t num_nodes);

    inline void setBlockState();
    inline uint32_t computeNumInvocations(NodeState &node);

    NodeState *sorted_nodes_;
    uint32_t num_nodes_;
    std::atomic_uint32_t cur_node_idx_;
    cuda::barrier<cuda::thread_scope_device> init_barrier_;

friend class Builder;
};

template <typename ContextT, typename WorldDataT, typename InitT>
class TaskGraphEntryBase {
public:
    static void init(const InitT *inits, int32_t num_worlds)
    {
        StateManager *state_mgr = mwGPU::getStateManager();
        new (state_mgr) StateManager(0);

        TaskGraph::Builder builder(1024, 1024);

        WorldDataT::setup(*state_mgr, builder);

        builder.build((TaskGraph *)mwGPU::GPUImplConsts::get().taskGraph);

        for (int32_t world_idx = 0; world_idx < num_worlds; world_idx++) {
            const InitT &init = inits[world_idx];
            WorldBase *world = TaskGraph::getWorld(world_idx);

            ContextT ctx =
                mwGPU::EntryBase<ContextT, WorldDataT>::makeContext(WorldID {
                    world_idx,
                });

            new (world) WorldDataT(ctx, init);
        }
    }
};

template <typename ContextT, typename WorldDataT, typename InitT,
          decltype(TaskGraphEntryBase<ContextT, WorldDataT, InitT>::init) =
            TaskGraphEntryBase<ContextT, WorldDataT, InitT>::init>
class TaskGraphEntry :
    public TaskGraphEntryBase<ContextT, WorldDataT, InitT> {};

}
