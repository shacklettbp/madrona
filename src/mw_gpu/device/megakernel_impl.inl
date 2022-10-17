#pragma once
#include <madrona/job.hpp>
#include <madrona/macros.hpp>

static_assert(true); // https://github.com/clangd/clangd/issues/1167
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

extern "C" void madronaMWGPUMegakernel(uint32_t func_id,
        madrona::JobContainerBase *data,
        uint32_t *data_indices,
        uint32_t *invocation_offsets,
        uint32_t num_launches,
        uint32_t grid)
{
    dispatch(func_id, data, data_indices, invocation_offsets, num_launches,
             grid);
}
