#pragma once

#include <madrona/dyn_array.hpp>
#include <madrona/heap_array.hpp>
#include <madrona/span.hpp>
#include <madrona/state.hpp>
#include <madrona/context.hpp>

#include <functional>
#include <thread>

namespace madrona {
struct NodeBase {};

class TaskGraph {
private:
    static inline constexpr uint32_t maxNodeDataBytes = 128;
    struct alignas(maxNodeDataBytes) NodeData {
        char userData[maxNodeDataBytes];
    };
    static_assert(sizeof(NodeData) == 128);

    struct Node {
        void (*fn)(NodeBase *, Context *);
        uint32_t dataIDX;
        uint32_t numChildren;
    };

public:
    struct NodeID {
        uint32_t id;
    };

    struct DataID {
        int32_t id;
    };

    template <typename NodeT>
    struct TypedDataID : DataID {};

    class Builder {
    public:
        Builder(Context &ctx);

        template <typename NodeT, typename... Args>
        TypedDataID<NodeT> constructNodeData(Args &&...args);

        template <auto fn, typename NodeT>
        NodeID addNodeFn(TypedDataID<NodeT> data,
                         Span<const NodeID> dependencies,
                         Optional<NodeID> parent_node =
                             Optional<NodeID>::none());

        template <typename NodeT, typename... Args>
        TaskGraph::NodeID addDefaultNode(Span<const NodeID> dependencies,
                                         Args && ...args);

        template <typename NodeT>
        inline NodeID addToGraph(Span<const NodeID> dependencies);

        template <typename NodeT>
        NodeT & getDataRef(TypedDataID<NodeT> data_id);

        TaskGraph build();

    private:
        NodeID registerNode(uint32_t data_idx,
                            void (*fn)(NodeBase *, Context *),
                            Span<const NodeID> dependencies,
                            Optional<NodeID> parent_node);

        struct StagedNode {
            Node node;
            int32_t parentID;
            uint32_t dependencyOffset;
            uint32_t numDependencies;
        };

        Context *ctx_;
        DynArray<StagedNode> staged_;
        DynArray<NodeData> node_datas_;
        DynArray<NodeID> all_dependencies_;
    };

    TaskGraph(const TaskGraph &) = delete;

    void run(Context *ctx);

private:
    TaskGraph(HeapArray<Node> &&sorted_nodes,
              HeapArray<NodeData> &&node_datas);

    HeapArray<Node> sorted_nodes_;
    HeapArray<NodeData> node_datas_;

friend class Builder;
};

class ThreadPoolExecutor {
public:
    struct Job {
        void (*fn)(void *);
        void *data;
    };

    ThreadPoolExecutor(CountT num_worlds, CountT num_workers = 0);
    ~ThreadPoolExecutor();
    void run(Job *jobs, CountT num_jobs);

protected:
    void ctxInit(void (*init_fn)(void *, const WorkerInit &),
                 void *init_data, CountT world_idx);

    ECSRegistry getECSRegistry();

private:
    void workerThread();

    HeapArray<std::thread> workers_;
    alignas(MADRONA_CACHE_LINE) std::atomic_int32_t worker_wakeup_;
    alignas(MADRONA_CACHE_LINE) std::atomic_int32_t main_wakeup_;
    Job *current_jobs_;
    uint32_t num_jobs_;
    alignas(MADRONA_CACHE_LINE) std::atomic_uint32_t next_job_;
    alignas(MADRONA_CACHE_LINE) std::atomic_uint32_t num_finished_;
    StateManager state_mgr_;
    HeapArray<StateCache> state_caches_;
};

template <typename ContextT, typename WorldT, typename... InitTs>
class TaskGraphExecutor : private ThreadPoolExecutor {
public:
    struct Config {
        uint32_t numWorlds;
        uint32_t numWorkers = 0;
    };

    template <typename... Args>
    TaskGraphExecutor(const Config &cfg,
                      const Args * ... user_init_ptrs);

    inline void run();

private:
    struct WorldContext {
        ContextT ctx;
        WorldT worldData;
        TaskGraph taskgraph;

        inline WorldContext(const WorkerInit &worker_init,
                            const InitTs & ...world_inits);
                            
    };

    static inline void stepWorld(void *data_raw);

    HeapArray<WorldContext> world_contexts_;
    HeapArray<Job> jobs_;
};

template <typename ContextT, auto Fn, typename ...ComponentTs>
class ParallelForNode : public NodeBase {
public:
    ParallelForNode(Context &ctx);

    inline void run(Context *ctx_base);

    static TaskGraph::NodeID addToGraph(
            Context &ctx,
            TaskGraph::Builder &builder,
            Span<const TaskGraph::NodeID> dependencies);

private:
    Query<ComponentTs...> query_;
};

class ResetTmpAllocNode : public NodeBase {
public:
    inline void run(Context *ctx);

    static TaskGraph::NodeID addToGraph(
            Context &ctx,
            TaskGraph::Builder &builder,
            Span<const TaskGraph::NodeID> dependencies);
};

template <typename ArchetypeT>
class ClearTmpNode : public NodeBase {
public:
    inline void run(Context *ctx);

    static TaskGraph::NodeID addToGraph(
            Context &ctx,
            TaskGraph::Builder &builder,
            Span<const TaskGraph::NodeID> dependencies);
};

}

#include "taskgraph.inl"
