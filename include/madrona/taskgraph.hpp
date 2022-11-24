#pragma once

#include <madrona/dyn_array.hpp>
#include <madrona/heap_array.hpp>
#include <madrona/span.hpp>
#include <madrona/system.hpp>

#include <thread>

namespace madrona {

struct SystemID {
    uint32_t id;
};

class TaskGraph {
public:
    struct NodeID {
        uint32_t id;
    };

    class Builder {
    public:
        Builder();

        template <typename ContextT, auto Fn,
                  typename ...ComponentTs>
        NodeID parallelForNode(Span<const NodeID> deps)
        {
            // FIXME
            (void)deps;
            return NodeID { 0 };
        }

        TaskGraph build();
    private:
        struct StagedSystem {
            SystemBase *sys;
            uint32_t dependencyOffset;
            uint32_t numDependencies;
        };

        DynArray<StagedSystem> systems_;
        DynArray<SystemID> all_dependencies_;
    };

    TaskGraph(const TaskGraph &) = delete;
    ~TaskGraph();

    void run(void *data);

private:
    struct SystemInfo {
        SystemBase *sys;
        std::atomic_uint32_t curOffset;
        std::atomic_uint32_t numRemaining;
    };

    void workerThread();

    TaskGraph(HeapArray<SystemInfo> &&systems);
    HeapArray<std::thread> workers_;
    std::atomic_uint32_t num_sleeping_workers_;
    std::atomic_uint32_t worker_sleep_;
    std::atomic_uint32_t main_sleep_;

    std::atomic_uint32_t cur_sys_idx_;
    void *global_data_;

    HeapArray<SystemInfo> sorted_systems_;

friend class Builder;
};

template <typename Engine, typename WorldT, typename InitT>
class TaskGraphEntry {
public:
    inline TaskGraphEntry(const InitT *inits, uint32_t num_worlds)
        : mgr_(inits, num_worlds),
          taskgraph_([this]() {
              TaskGraph::Builder builder;
              mgr_.taskgraphSetup(builder);

              return builder.build();
          }())
    {}

    inline void run()
    {
        taskgraph_.run(&mgr_);
    }

private:
    WorldT mgr_;
    TaskGraph taskgraph_;
};

}
