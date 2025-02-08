#pragma once

#include <madrona/dyn_array.hpp>
#include <madrona/heap_array.hpp>
#include <madrona/span.hpp>
#include <madrona/state.hpp>
#include <madrona/fwd.hpp>
#include <madrona/context.hpp>

#include <functional>
#include <thread>

namespace madrona {
struct NodeBase {};

class TaskGraph {
private:
    static inline constexpr uint32_t maxNodeDataBytes = 256;
    static inline constexpr uint32_t maxNodeDataAlign = 64;
    struct alignas(maxNodeDataAlign) NodeData {
        char userData[maxNodeDataBytes];
    };
    static_assert(sizeof(NodeData) == 256);

    struct Node {
        void (*fn)(NodeBase *, Context *, TaskGraph *);
        uint32_t dataIDX;
        uint32_t numChildren;
    };

public:
    TaskGraph(StateManager *state_mgr,
              StateCache *state_cache,
              MADRONA_MW_COND(uint32_t world_id,) 
              HeapArray<Node> &&sorted_nodes,
              HeapArray<NodeData> &&node_datas);
    TaskGraph(const TaskGraph &) = delete;
    TaskGraph(TaskGraph &&) = default;

    TaskGraph & operator=(const TaskGraph &) = delete;
    TaskGraph & operator=(TaskGraph &&) = default;

    void run(Context *ctx);

    template <typename ArchetypeT>
    void clearTemporaries();
    void resetTmpAlloc();

    template <typename ContextT, typename Fn, typename ...ComponentTs>
    void iterateQuery(ContextT &ctx,
                      Query<ComponentTs...> &query,
                      Fn &&fn);

    inline StateManager & stateManager() const;

private:
    StateManager *state_mgr_;
    StateCache *state_cache_;
#ifdef MADRONA_MW_MODE
    uint32_t cur_world_id_;
#endif
    HeapArray<Node> sorted_nodes_;
    HeapArray<NodeData> node_datas_;

friend class TaskGraphBuilder;
};

}

#include "taskgraph.inl"
