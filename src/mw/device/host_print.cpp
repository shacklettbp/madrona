#pragma once

#if defined (MADRONA_MWGPU_BVH_MODULE)
#define MADRONA_MWGPU_MAX_BLOCKS_PER_SM 8
#endif

#include <madrona/mw_gpu/host_print.hpp>
#include <madrona/mw_gpu/const.hpp>

#if defined (MADRONA_MWGPU_BVH_MODULE)
#include <madrona/bvh.hpp>
extern "C" __constant__ madrona::BVHParams bvhParams;
#endif

namespace madrona {
namespace mwGPU {

HostPrint::HostPrint(void *channel_raw)
    : channel_((Channel *)channel_raw),
      device_lock_()
{}

void HostPrint::logSubmit(const char *str, void **ptrs, FmtType *types,
                          int32_t num_args)
{
#if defined (MADRONA_MWGPU_BVH_MODULE)
    HostPrint *host_print = (HostPrint *)bvhParams.hostPrintAddr;
#else
    auto host_print = (HostPrint *)GPUImplConsts::get().hostPrintAddr;
#endif
    host_print->logSubmitImpl(str, ptrs, types, num_args);
}

void HostPrint::logSubmitImpl(const char *str, void **ptrs, FmtType *types,
                              int32_t num_args)
{
    using cuda::std::memory_order_relaxed;
    using cuda::std::memory_order_release;

    device_lock_.lock();

    int32_t cur_offset = 0;
    do {
        channel_->buffer[cur_offset] = str[cur_offset];
    } while (str[cur_offset++] != '\0');

    for (int i = 0; i < num_args; i++) {
        FmtType type = types[i];

        int32_t arg_size;
        switch (type) {
        case FmtType::I32: {
            arg_size = sizeof(int32_t);
        }; break;
        case FmtType::U32: {
            arg_size = sizeof(uint32_t);
        }; break;
        case FmtType::I64: {
            arg_size = sizeof(int64_t);
        }; break;
        case FmtType::U64: {
            arg_size = sizeof(uint64_t);
        }; break;
        case FmtType::Float: {
            arg_size = sizeof(float);
        }; break;
        case FmtType::Ptr: {
            arg_size = sizeof(void *);
        }; break;
        default: 
            __builtin_unreachable();
        }

        memcpy(&channel_->buffer[cur_offset],
               ptrs[i], arg_size);
        cur_offset += arg_size;
        assert(cur_offset < max_bytes);

        channel_->args[i] = type;
    }
    channel_->numArgs = num_args;

    channel_->signal.store(1, memory_order_release);

    while (channel_->signal.load(memory_order_relaxed) == 1) {
        __nanosleep(0);
    }

    device_lock_.unlock();
}

}
}
