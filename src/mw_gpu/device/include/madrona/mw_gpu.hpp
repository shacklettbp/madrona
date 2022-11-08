/*
 * Copyright 2021-2022 Brennan Shacklett and contributors
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 */
#pragma once

#include <cstdint>

#include "mw_gpu/entry.hpp"

namespace madrona {

template <typename ContextT, typename InitT, typename BaseT>
class GPUJobEntry : mwGPU::EntryBase<BaseT> {
public:
    static void submitInit(uint32_t invocation_idx, void *world_init_ptr);
    static void submitRun(uint32_t invocation_idx);

private:
    static ContextT makeFakeContext(uint32_t invocation_idx);
};

#include "mw_gpu.inl"
