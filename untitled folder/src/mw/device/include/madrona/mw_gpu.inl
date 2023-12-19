/*
 * Copyright 2021-2022 Brennan Shacklett and contributors
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 */
#pragma once

#include "mw_gpu/worker_init.hpp"

namespace madrona {

template <typename ContextT, typename InitT, typename BaseT>
void GPUEntry<ContextT, InitT, BaseT>::submitInit(uint32_t invocation_idx,
                                                  void *world_init_ptr)
{
    ContextT ctx = makeFakeContext(invocation_idx);

    InitT *base_init = (InitT *)world_init_ptr;
    InitT *init = base_init + ctx.worldID();

    ctx.submit([init](ContextT &ctx) {
        BaseT::init(ctx, *init);
    }, false);
}

template <typename ContextT, typename InitT, typename BaseT>
void GPUEntry<ContextT, InitT, BaseT>::submitRun(uint32_t invocation_idx)
{
    ContextT ctx = makeFakeContext(invocation_idx);

    ctx.submit([](ContextT &ctx) {
        BaseT::run(ctx);
    }, false);
}

template <typename ContextT, typename InitT, typename BaseT>
ContextT GPUEntry<ContextT, InitT, BaseT>::makeFakeContext(
    uint32_t invocation_idx)
{
    uint32_t lane_id =
        invocation_idx % mwGPU::consts::numWarpThreads;
    uint32_t world_id = lane_id;

    return JobManager::makeContext<ContextT>(JobID { 0, 0 }, 0,
                                             world_id, lane_id);
}

}
