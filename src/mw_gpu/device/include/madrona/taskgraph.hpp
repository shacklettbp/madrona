#pragma once

#include <madrona/span.hpp>
#include <madrona/system.hpp>

#include "mw_gpu/const.hpp"

#include <cuda/barrier>

namespace madrona {

struct SystemID {
    uint32_t id;
};

class TaskGraph {
public:
    class Builder {
    public:
        Builder(uint32_t max_num_systems, uint32_t max_num_dependencies);
        ~Builder();

        SystemID registerSystem(SystemBase &sys,
                                Span<const SystemID> dependencies);

        void build(TaskGraph *out);
    private:
        struct StagedSystem {
            SystemBase *sys;
            uint32_t dependencyOffset;
            uint32_t numDependencies;
        };

        StagedSystem *systems_;
        uint32_t num_systems_;
        SystemID *all_dependencies_;
        uint32_t num_dependencies_;
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

    WorkerState getWork(SystemBase **run_sys, uint32_t *run_func_id,
        uint32_t *run_offset);

    void finishWork();

    struct BlockState;
private:
    struct SystemInfo {
        SystemBase *sys;
        std::atomic_uint32_t curOffset;
        std::atomic_uint32_t numRemaining;
    };

    TaskGraph(SystemInfo *systems, uint32_t num_systems);

    inline void setBlockState();

    std::atomic_uint32_t cur_sys_idx_;
    SystemInfo *sorted_systems_;
    uint32_t num_systems_;
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
