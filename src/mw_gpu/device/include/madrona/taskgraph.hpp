#pragma once

#include <madrona/span.hpp>
#include <madrona/system.hpp>
#include <madrona/query.hpp>

#include <madrona/state.hpp>

#include "mw_gpu/const.hpp"

#include <cuda/barrier>

namespace madrona {

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
inline void userEntry(WorldBase *world_base, int32_t invocation_idx)
{
    EntryT::run(world_base, invocation_idx);
}

template <typename EntryT>
struct UserFuncIDBase {
    static uint32_t id;
};

template <typename EntryT,
          decltype(userEntry<EntryT>) = userEntry<EntryT>>
struct UserFuncID : UserFuncIDBase<EntryT> {};


template <typename Fn, typename ...ComponentTs>
struct ParallelForEntry {
    static inline void run(EntryData &data,
                           Context &ctx,
                           int32_t invocation_idx)
    {
        QueryRef *query_ref = data.parallelFor.query;
        StateManager *state_mgr = mwGPU::getStateManager();

        int32_t cumulative_num_rows = 0;
        state_mgr->iterateArchetypesRaw<sizeof...(ComponentTs)>(query_ref,
                [&](int32_t num_rows, auto ...raw_ptrs) {
            int32_t tbl_offset = invocation_idx - cumulative_num_rows;
            cumulative_num_rows += num_rows;
            if (tbl_offset >= num_rows) {
                return false;
            }

            Fn(ctx, ((ComponentTs *)raw_ptrs)[tbl_offset] ...); // FIXME

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

        template <typename ComponentT, typename Fn>
        NodeID parallelForNode(Span<const NodeID> dependencies)
        {
            using Entry = typename mwGPU::ParallelForEntry<Fn, ComponentT>;
            uint32_t func_id = mwGPU::UserFuncID<Entry>::id;

            auto query = mwGPU::getStateManager()->query<ComponentT>();
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
                        uint32_t *run_func_id, uint32_t *run_offset);

    void finishWork();

    struct BlockState;
private:
    TaskGraph(StateManager &state_mgr, NodeState *nodes, uint32_t num_nodes);

    inline void setBlockState();
    inline uint32_t computeNumInvocations(NodeState &node);

    NodeState *sorted_nodes_;
    uint32_t num_nodes_;
    std::atomic_uint32_t cur_node_idx_;
    cuda::barrier<cuda::thread_scope_device> init_barrier_;

friend class Builder;
};

template <typename MgrT, typename InitT>
class TaskGraphEntryBase {
public:
    static void init(const InitT *inits, uint32_t num_worlds)
    {
        MgrT *mgr = (MgrT *)mwGPU::GPUImplConsts::get().taskGraphUserData;
        new (mgr) MgrT(inits, num_worlds);
        TaskGraph::Builder builder(1024, 1024);
        mgr->taskgraphSetup(builder);
        builder.build((TaskGraph *)mwGPU::GPUImplConsts::get().taskGraph);
    }
};

template <typename MgrT, typename InitT,
          decltype(TaskGraphEntryBase<MgrT, InitT>::init) =
            TaskGraphEntryBase<MgrT, InitT>::init>
class TaskGraphEntry : public TaskGraphEntryBase<MgrT, InitT> {};

}
