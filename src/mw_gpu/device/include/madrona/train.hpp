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
