#include <madrona/taskgraph.hpp>
#include <madrona/crash.hpp>

#include <sched.h>
#include <unistd.h>


namespace madrona {
namespace {

int getNumCores()
{
    int os_num_threads = sysconf(_SC_NPROCESSORS_ONLN);

    if (os_num_threads == -1) {
        FATAL("Failed to get number of concurrent threads");
    }

    return os_num_threads;
}

}

TaskGraph::Builder::Builder()
    : systems_(0),
      all_dependencies_(0)
{}

SystemID TaskGraph::Builder::registerSystem(SystemBase &sys,
                        Span<const SystemID> dependencies)
{
    uint32_t offset = all_dependencies_.size();
    uint32_t num_deps = dependencies.size();

    all_dependencies_.resize(all_dependencies_.size() + num_deps,
                             [](auto) {});

    memcpy(&all_dependencies_[offset], dependencies.data(),
           sizeof(SystemID) * num_deps);

    systems_.push_back(StagedSystem {
        &sys,
        offset,
        num_deps,
    });

    return SystemID {
        (uint32_t)systems_.size() - 1,
    };
}

TaskGraph * TaskGraph::Builder::build()
{
    assert(systems_[0].numDependencies == 0);
    HeapArray<SystemInfo> sorted_systems(systems_.size());
    HeapArray<bool> queued(systems_.size());
    sorted_systems[0] = SystemInfo {
        systems_[0].sys,
    };
    queued[0] = true;

    HeapArray<uint32_t> remaining_systems(systems_.size() - 1);

    for (int64_t i = 1; i < (int64_t)systems_.size(); i++) {
        queued[i]  = false;
        remaining_systems[i - 1] = i;
    }

    uint32_t sorted_idx = 1;

    uint32_t num_remaining_systems = remaining_systems.size();
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
            sorted_systems[sorted_idx++] = SystemInfo {
                cur_sys.sys,
            };
            num_remaining_systems--;
        }
    }

    return new TaskGraph(std::move(sorted_systems));
}

TaskGraph::TaskGraph(HeapArray<SystemInfo> &&systems)
    : workers_(getNumCores()),
      worker_sleep_(0),
      cur_sys_(0),
      cur_num_invocations_(0),
      num_remaining_invocations_(0),
      cur_invocation_offset_(0),
      global_data_(nullptr),
      sorted_systems_(std::move(systems))
{
    for (int64_t i = 0; i < (int)workers_.size(); i++) {
        workers_.emplace(i, [this]() {
            workerThread();
        });
    }
}

TaskGraph::~TaskGraph()
{
    worker_sleep_.store(~0_u32, std::memory_order_release);
    worker_sleep_.notify_all();

    for (auto &thread : workers_) {
        thread.join();
    }
}

void TaskGraph::run(void *data)
{
    global_data_ = data;

    uint32_t num_invocations =
        sorted_systems_[0].sys->numInvocations.load(std::memory_order_relaxed);
    printf("Init invocations %u\n", num_invocations);
    cur_num_invocations_.store(num_invocations,
                               std::memory_order_relaxed);
    num_remaining_invocations_.store(num_invocations,
                                     std::memory_order_relaxed);
    cur_invocation_offset_.store(0, std::memory_order_relaxed);

    cur_sys_.store(0, std::memory_order_release);

    worker_sleep_.store(1, std::memory_order_release);
    worker_sleep_.notify_all();

    main_sleep_.store(0, std::memory_order_relaxed);
    main_sleep_.wait(0, std::memory_order_acquire);

    printf("Wakeup\n");
}

void TaskGraph::workerThread()
{
    worker_sleep_.wait(0, std::memory_order_acquire);
    uint32_t wakeup_val = worker_sleep_.load(std::memory_order_relaxed);
    if (wakeup_val == ~0_u32) {
        return;
    }


    while (true) {
        uint32_t sys_idx = cur_sys_.load(std::memory_order_acquire);

        if (sys_idx == sorted_systems_.size()) {
            printf("Sleeping\n");
            worker_sleep_.wait(0, std::memory_order_acquire);
            wakeup_val = worker_sleep_.load(std::memory_order_relaxed);

            if (wakeup_val == ~0_u32) {
                return;
            }

            printf("Wakeup\n");

            continue;
        }

        uint32_t cur_offset = cur_invocation_offset_.load(std::memory_order_relaxed); 
        uint32_t num_invocations = cur_num_invocations_.load(std::memory_order_relaxed); 

        if (cur_offset >= num_invocations) {
            sched_yield();
            continue;
        }

        cur_offset = cur_invocation_offset_.fetch_add(1, std::memory_order_relaxed);

        if (cur_offset >= num_invocations) {
            sched_yield();
            continue;
        }

        SystemInfo &sys_info = sorted_systems_[sys_idx];

        sys_info.sys->entry_fn_(sys_info.sys, global_data_, cur_offset);

        uint32_t prev_remaining = num_remaining_invocations_.fetch_sub(1,
            std::memory_order_acq_rel);

        if (prev_remaining == 1) {
            uint32_t next_sys = sys_idx + 1;

            while (true) {
                if (next_sys < sorted_systems_.size()) {
                    uint32_t new_num_invocations =
                        sorted_systems_[next_sys].sys->numInvocations.load(
                            std::memory_order_relaxed);

                    if (new_num_invocations == 0) {
                        next_sys++;
                        continue;
                    }

                    cur_invocation_offset_.store(0, std::memory_order_relaxed);
                    cur_num_invocations_.store(new_num_invocations,
                                               std::memory_order_relaxed);
                    num_remaining_invocations_.store(new_num_invocations,
                                                     std::memory_order_relaxed);
                } else {
                    worker_sleep_.store(0, std::memory_order_relaxed);
                    main_sleep_.store(1, std::memory_order_release);
                    main_sleep_.notify_one();
                    printf("End\n");
                }
            }

            cur_sys_.store(next_sys, std::memory_order_release);
            break;
        }
    }
}

}
