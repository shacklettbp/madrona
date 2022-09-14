#pragma once

#include <madrona/mw_gpu/worker_init.hpp>

namespace madrona {

template <typename ContextT, typename BaseT>
void TrainingEntry<ContextT, BaseT>::submitInit(uint32_t invocation_idx)
{
    ContextT ctx = makeContext(invocation_idx);

    ctx.submit([](ContextT &ctx) {
        BaseT::init(ctx);
    }, false);
}

template <typename ContextT, typename BaseT>
void TrainingEntry<ContextT, BaseT>::submitRun(uint32_t invocation_idx)
{
    ContextT ctx = makeContext(invocation_idx);

    ctx.submit([](ContextT &ctx) {
        BaseT::run(ctx);
    }, false);
}

template <typename ContextT, typename BaseT>
ContextT TrainingEntry<ContextT, BaseT>::makeContext(
    uint32_t invocation_idx)
{
    uint32_t lane_id =
        invocation_idx % madrona::mwGPU::ICfg::numWarpThreads;

    WorkerInit worker_init {
        .jobID = 0,
        .gridID = 0,
        .worldID = invocation_idx,
        .laneID = lane_id,
    };

    char *ctx_data_base = (char *)mwGPU::GPUImplConsts::get().ctxDataAddr;
    void *ctx_data = ctx_data_base +
        worker_init.worldID * mwGPU::GPUImplConsts::get().numCtxDataBytes;

    return ContextT(ctx_data, std::move(worker_init));
}

}
