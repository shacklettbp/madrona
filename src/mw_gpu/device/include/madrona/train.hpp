/*
 * Copyright 2021-2022 Brennan Shacklett and contributors
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 */
#pragma once

#include <cstdint>

#include <madrona/mw_gpu/entry.hpp>

namespace madrona {

template <typename ContextT, typename BaseT>
class TrainingEntry : mwGPU::EntryBase<BaseT> {
public:
    static void submitInit(uint32_t invocation_idx);
    static void submitRun(uint32_t invocation_idx);

private:
    static ContextT makeContext(uint32_t invocation_idx);
};

}

#include "train.inl"
