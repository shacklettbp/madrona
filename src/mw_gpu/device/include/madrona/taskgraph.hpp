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

struct alignas(64) NodeData {
    char userData[48];
    uint32_t numRunInvocations;
    uint32_t numCountInvocations;
    uint32_t runID;
    uint32_t countID;
};

static_assert(sizeof(NodeData) == 64);

#if 0
    struct ParallelFor {
        QueryRef *query;
    };

    struct ClearTmp {
        uint32_t archetypeID;
    };

    struct CompactArchetype {
        uint32_t archetypeID;
    };
    
    struct SortArchetypeSetup {
        uint32_t archetypeID;
        int32_t columnIDX;
        int32_t numPasses;
    };

    struct SortArchetype {
        uint32_t archetypeID;
    };
    
    struct RecycleEntities {
        int32_t recycleBase;
    };

    struct CustomData {
        void *ptr;
    };

    union {
        ParallelFor parallelFor;
        ClearTmp clearTmp;
        CompactArchetype compactArchetype;
        SortArchetypeSetup sortArchetypeSetup;
        SortArchetype sortArchetype;
        RecycleEntities recycleEntities;
        CustomData custom;
    };
#endif

template <typename EntryT>
__attribute__((used, always_inline))
inline void userEntry(NodeData &node_data, int32_t invocation_idx)
{
    EntryT::exec(node_data, invocation_idx);
}

template <typename EntryT>
struct FuncIDBase {
    static uint32_t id;
};

template <typename EntryT,
          decltype(userEntry<EntryT>) = userEntry<EntryT>>
struct UserFuncID : FuncIDBase<EntryT> {};

template <typename ContextT, bool = false>
struct WorldTypeExtract {
    using type = typename ContextT::WorldDataT;
};

template <bool ignore>
struct WorldTypeExtract<Context, ignore> {
    using type = WorldBase;
};

}

template <typename NodeT,
         int32_t fixed_run_count = 0,
         int32_t num_count_invocations = 1>
struct NodeBase {
    struct RunEntry {
        static inline void exec(mwGPU::NodeData &storage,
                                int32_t invocation_idx)
        {
            auto ptr = (NodeT *)storage.userData;
            ptr->run(invocation_idx);
        }
    };

    struct CountEntry {
        static inline void exec(mwGPU::NodeData &storage,
                                int32_t)
        {
            auto ptr = (NodeT *)storage.userData;
            return ptr->numInvocations();
        }
    };
};

class TaskGraph {
private:
    struct NodeState {
        uint32_t dataIDX;
        bool countNode;
        std::atomic_uint32_t curOffset;
        std::atomic_uint32_t numRemaining;
        std::atomic_uint32_t totalNumInvocations;
    };

public:
    struct NodeID {
        uint32_t id;
    };

    class Builder {
    public:
        Builder(uint32_t max_num_nodes,
                uint32_t max_num_dependencies);
        ~Builder();

        template <typename NodeT,
                  int32_t fixed_run_count,
                  int32_t num_count_invokes>
        inline NodeID addNode(
            const NodeBase<NodeT, fixed_run_count, num_count_invokes> &node,
            Span<const NodeID> dependencies)
        {
            mwGPU::NodeData node_data;
            new (node_data.userData) NodeT(static_cast<NodeT &>(node));
            if constexpr (fixed_run_count == 0) {
                node_data.numRunInvocations = 0;
                node_data.numCountInvocations = num_count_invokes;
                static_assert(num_count_invokes != 0);

                node_data.countID =
                    mwGPU::UserFuncID<typename NodeT::CountEntry>::id;
            } else {
                node_data.numRunInvocations = fixed_run_count;
                node_data.numCountInvocations = 0;
            }

            node_data.runID = mwGPU::UserFuncID<typename NodeT::RunEntry>::id;

            return registerNode(node_data, dependencies, fixed_run_count == 0);
        }

        NodeID recycleEntitiesNode(Span<const NodeID> dependencies);

        NodeID resetTmpAllocatorNode(Span<const NodeID> dependencies);

        void build(TaskGraph *out);

    private:
        NodeID compactArchetypeNode(uint32_t archetype_id,
                                    Span<const NodeID> dependencies);

        NodeID sortArchetypeNode(uint32_t archetype_id,
                                 uint32_t component_id,
                                 Span<const NodeID> dependencies);

        NodeID registerNode(const mwGPU::NodeData &node_data,
                            Span<const NodeID> dependencies,
                            bool runtime_count);

        struct StagedNode {
            int32_t dataIDX;
            uint32_t dependencyOffset;
            uint32_t numDependencies;
        };

        StagedNode *nodes_;
        mwGPU::NodeData *node_datas_;
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

}
