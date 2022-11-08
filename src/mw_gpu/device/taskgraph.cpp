#include <madrona/taskgraph.hpp>
#include <madrona/crash.hpp>

namespace madrona {
namespace consts {
static constexpr uint32_t numMegakernelThreads = 256;
}

TaskGraph::Builder::Builder(uint32_t max_num_systems, uint32_t max_num_dependencies)
    : systems_((StagedSystem *)malloc(sizeof(StagedSystem) * max_num_systems)),
      num_systems_(0),
      all_dependencies_((SystemID *)malloc(sizeof(SystemID) * max_num_dependencies)),
      num_dependencies_(0)
{}

TaskGraph::Builder::~Builder()
{
    free(systems_);
    free(all_dependencies_);
}

SystemID TaskGraph::Builder::registerSystem(SystemBase &sys,
                        Span<const SystemID> dependencies)
{
    uint32_t offset = num_dependencies_;
    uint32_t num_deps = dependencies.size();

    num_dependencies_ += num_deps;

    for (int i = 0; i < (int)num_deps; i++) {
        all_dependencies_[offset + i] = dependencies[i];
    }

    systems_[num_systems_++] = StagedSystem {
        &sys,
        offset,
        num_deps,
    };

    return SystemID {
        num_systems_ - 1,
    };
}

void TaskGraph::Builder::build(TaskGraph *out)
{
    assert(systems_[0].numDependencies == 0);
    SystemInfo *sorted_systems = 
        (SystemInfo *)malloc(sizeof(SystemInfo) * num_systems_);
    bool *queued = (bool *)malloc(num_systems_ * sizeof(bool));
    new (&sorted_systems[0]) SystemInfo {
        systems_[0].sys,
        0,
        0,
    };
    queued[0] = true;

    uint32_t num_remaining_systems = num_systems_ - 1;
    uint32_t *remaining_systems =
        (uint32_t *)malloc(num_remaining_systems * sizeof(uint32_t));

    for (int64_t i = 1; i < (int64_t)num_systems_; i++) {
        queued[i]  = false;
        remaining_systems[i - 1] = i;
    }

    uint32_t sorted_idx = 1;

    while (num_remaining_systems > 0) {
        uint32_t cur_sys_idx = remaining_systems[0];
        StagedSystem &cur_sys = systems_[cur_sys_idx];

        bool dependencies_satisfied = true;
        for (uint32_t dep_offset = 0; dep_offset < cur_sys.numDependencies;
             dep_offset++) {
            uint32_t dep_system_idx =
                all_dependencies_[cur_sys.dependencyOffset + dep_offset].id;
            if (!queued[dep_system_idx]) {
                dependencies_satisfied = false;
                break;
            }
        }

        remaining_systems[0] =
            remaining_systems[num_remaining_systems - 1];
        if (!dependencies_satisfied) {
            remaining_systems[num_remaining_systems - 1] =
                cur_sys_idx;
        } else {
            queued[cur_sys_idx] = true;
            new (&sorted_systems[sorted_idx++]) SystemInfo {
                cur_sys.sys,
                0,
                0,
            };
            num_remaining_systems--;
        }
    }

    free(remaining_systems);
    free(queued);

    new (out) TaskGraph(sorted_systems, num_systems_);
}

TaskGraph::TaskGraph(SystemInfo *systems, uint32_t num_systems)
    : cur_sys_idx_(num_systems),
      sorted_systems_(systems),
      num_systems_(num_systems),
      init_barrier_(82 * 5)
{}

TaskGraph::~TaskGraph()
{
    free(sorted_systems_);
}

struct TaskGraph::BlockState {
    WorkerState state;
    uint32_t sysIdx;
    uint32_t numInvocations;
    uint32_t funcID;
    uint32_t runOffset;
};

static __shared__ TaskGraph::BlockState sharedBlockState;

void TaskGraph::init()
{
    int thread_idx = threadIdx.x;
    if (thread_idx != 0) {
        return;
    }

    int block_idx = blockIdx.x;

    if (block_idx == 0) {
        SystemInfo &first_sys = sorted_systems_[0];

        uint32_t num_invocations =
            first_sys.sys->numInvocations.load(std::memory_order_relaxed);

        first_sys.numRemaining.store(num_invocations,
            std::memory_order_relaxed);

        first_sys.curOffset.store(0, std::memory_order_relaxed);
        cur_sys_idx_.store(0, std::memory_order_release);
    } 

    init_barrier_.arrive_and_wait();
}

void TaskGraph::setBlockState()
{
    uint32_t sys_idx = cur_sys_idx_.load(std::memory_order_acquire);
    if (sys_idx == num_systems_) {
        sharedBlockState.state = WorkerState::Exit;
        return;
    }

    SystemInfo &cur_sys = sorted_systems_[sys_idx];

    uint32_t cur_offset = 
        cur_sys.curOffset.load(std::memory_order_relaxed);

    uint32_t total_invocations =
        cur_sys.sys->numInvocations.load(std::memory_order_relaxed);

    if (cur_offset >= total_invocations) {
        sharedBlockState.state = WorkerState::Loop;
        return;
    }

    cur_offset = cur_sys.curOffset.fetch_add(consts::numMegakernelThreads,
        std::memory_order_relaxed);

    if (cur_offset >= total_invocations) {
        sharedBlockState.state = WorkerState::Loop;
        return;
    }

    sharedBlockState.state = WorkerState::Run;
    sharedBlockState.sysIdx = sys_idx;
    sharedBlockState.numInvocations = total_invocations;
    sharedBlockState.funcID = cur_sys.sys->sys_id_;
    sharedBlockState.runOffset = cur_offset;
}

TaskGraph::WorkerState TaskGraph::getWork(SystemBase **run_sys,
                                          uint32_t *run_func_id,
                                          uint32_t *run_offset)
{
    int thread_idx = threadIdx.x;

    if (thread_idx == 0) {
        setBlockState();
    }

    __syncthreads();

    WorkerState worker_state = sharedBlockState.state;

    if (worker_state != WorkerState::Run) {
        return worker_state;
    }

    uint32_t num_invocations = sharedBlockState.numInvocations;
    uint32_t base_offset = sharedBlockState.runOffset;

    uint32_t thread_offset = base_offset + thread_idx;
    if (thread_offset >= num_invocations) {
        return WorkerState::PartialRun;
    }

    *run_sys = sorted_systems_[sharedBlockState.sysIdx].sys;
    *run_func_id = sharedBlockState.funcID;
    *run_offset = thread_offset;

    return WorkerState::Run;
}

void TaskGraph::finishWork()
{
    int thread_idx = threadIdx.x;
    __syncthreads();

    if (thread_idx != 0) return;

    uint32_t num_finished = std::min(
        sharedBlockState.numInvocations - sharedBlockState.runOffset,
        consts::numMegakernelThreads);

    uint32_t sys_idx = sharedBlockState.sysIdx;
    SystemInfo &cur_sys = sorted_systems_[sys_idx];

    uint32_t prev_remaining = cur_sys.numRemaining.fetch_sub(num_finished,
        std::memory_order_acq_rel);

    if (prev_remaining == num_finished) {
        uint32_t next_sys_idx = sys_idx + 1;

        while (true) {
            if (next_sys_idx < num_systems_) {
                uint32_t new_num_invocations =
                    sorted_systems_[next_sys_idx].sys->numInvocations.load(
                        std::memory_order_relaxed);

                if (new_num_invocations == 0) {
                    next_sys_idx++;
                    continue;
                }

                SystemInfo &next_sys = sorted_systems_[next_sys_idx];
                next_sys.curOffset.store(0, std::memory_order_relaxed);
                next_sys.numRemaining.store(new_num_invocations,
                                            std::memory_order_relaxed);
            }

            cur_sys_idx_.store(next_sys_idx, std::memory_order_release);
            break;
        }
    }
}

}

extern "C" __global__ void madronaMWGPUComputeConstants(
    uint32_t num_worlds,
    uint32_t num_world_data_bytes,
    uint32_t world_data_alignment,
    madrona::mwGPU::GPUImplConsts *out_constants,
    size_t *job_system_buffer_size)
{
    using namespace madrona;
    using namespace madrona::mwGPU;

    uint64_t total_bytes = sizeof(TaskGraph);

    uint64_t chunk_allocator_offset = utils::roundUp(total_bytes,
        (uint64_t)alignof(ChunkAllocator));

    total_bytes = chunk_allocator_offset + sizeof(ChunkAllocator);

    uint64_t world_data_offset =
        utils::roundUp(total_bytes, (uint64_t)world_data_alignment);

    total_bytes =
        world_data_offset + (uint64_t)num_world_data_bytes;

    *out_constants = GPUImplConsts {
        .jobSystemAddr = (void *)0ul,
        .stateManagerAddr = (void *)0ul,
        .chunkAllocatorAddr = (void *)chunk_allocator_offset,
        .chunkBaseAddr = (void *)0ul,
        .worldDataAddr = (void *)0ul,
        .numWorldDataBytes = 0ul,
        .numWorlds = num_worlds,
        .jobGridsOffset = (uint32_t)0,
        .jobListOffset = (uint32_t)0,
        .maxJobsPerGrid = 0,
        .sharedJobTrackerOffset = (uint32_t)0,
        .userJobTrackerOffset = (uint32_t)0,
        .taskGraph = (void *)0ul,
        .taskGraphUserData = (void *)world_data_offset,
    };

    *job_system_buffer_size = total_bytes;
}
