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
    class Builder {
    public:
        Builder();

        SystemID registerSystem(SystemBase &sys,
                                Span<const SystemID> dependencies);

        TaskGraph * build();
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
    };

    void workerThread();

    TaskGraph(HeapArray<SystemInfo> &&systems);
    HeapArray<std::thread> workers_;
    std::atomic_uint32_t worker_sleep_;
    std::atomic_uint32_t main_sleep_;

    std::atomic_uint32_t cur_sys_;
    std::atomic_uint32_t cur_num_invocations_;
    std::atomic_uint32_t num_remaining_invocations_;
    std::atomic_uint32_t cur_invocation_offset_;
    void *global_data_;

    HeapArray<SystemInfo> sorted_systems_;

friend class Builder;
};

}
