#pragma once

#include <cstdint>

#include <madrona/gpu_train/entry.hpp>

namespace madrona {

template <typename ContextT, typename BaseT>
class TrainingEntry : gpuTrain::EntryBase<BaseT> {
public:
    static void submitInit(uint32_t invocation_idx);
    static void submitRun(uint32_t invocation_idx);

private:
    static ContextT makeContext(uint32_t invocation_idx);
};

}

#include "train.inl"
