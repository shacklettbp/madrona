#pragma once
#include <madrona/job.hpp>
#include <madrona/macros.hpp>

namespace madrona {
namespace mwGPU {

#ifdef MADRONA_CLANG
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wundefined-internal"
#endif
static inline __attribute__((always_inline)) void dispatch(
        uint32_t func_id,
        madrona::JobContainerBase *data,
        uint32_t *data_indices,
        uint32_t *invocation_offsets,
        uint32_t num_launches, 
        uint32_t grid);
#ifdef MADRONA_CLANG
#pragma clang diagnostic pop
#endif

static inline __attribute__((always_inline)) void megakernelImpl()
{
    static __shared__ RunnableJob threadblock_job;
    static __shared__ bool job_found;

    while (true) {
        {
            JobManager *job_mgr = JobManager::get();
            if (job_mgr->numOutstandingInvocations.load(
                    std::memory_order_relaxed) == 0) {
                break;
            }

            if (threadIdx.x == 0) {
                job_found = job_mgr->startBlockIter(&threadblock_job);
            }
            __syncthreads();

            if (!job_found) {
                break;
            }
        }

        RunnableJob runnable = threadblock_job;

        dispatch(func_id, data, data_indices, invocation_offsets, num_launches,
                 grid);

        // Iterate through submitted jobs, merge job IDs, etc
        JobManager::get()->finishBlockIter();
    }
}

}
}

extern "C" void madronaMWGPUMegakernel()
{
    madrona::mwGPU::megakernelImpl();
}
