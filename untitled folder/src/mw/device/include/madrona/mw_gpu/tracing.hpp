#pragma once

#ifdef MADRONA_GPU_MODE

#include <stdint.h>
#include <atomic>
#include "const.hpp"

#endif

namespace madrona::mwGPU {
class DeviceTracingManager;

// todo: expand to log the activity of every block instead of only nodes
enum class DeviceEvent : uint32_t {
    calibration = 0,
    nodeStart = 1,
    nodeFinish = 2,
    blockStart = 3,
    blockWait = 4,
    blockExit = 5,
};

class DeviceTracing {
private:
    // collect 1M events for a single step, if overflow this step will be dropped
    // for hide & seek with 16384 worlds, roughly half million events will be generated
    static constexpr inline uint64_t maxLogSize = 1000000;

    struct DeviceLog {
        DeviceEvent event;
        uint32_t funcID;
        uint32_t numInvocations;
        uint32_t nodeID;
        uint32_t warpID;
        uint32_t blockID;
        uint32_t smID;
        // to align with memory address and future use
        uint32_t padding;
        uint64_t cycleCount;
    };

    AtomicI32 cur_index_;
    DeviceLog device_logs_[maxLogSize];

public:
    inline DeviceTracing()
        : cur_index_(0),
          device_logs_ {}
    {}

    inline int32_t getIndex()
    {
        return cur_index_.load_relaxed();
    }

#ifdef MADRONA_GPU_MODE

#ifdef MADRONA_TRACING
    static inline DeviceTracing & get()
    {
        return *(DeviceTracing *)mwGPU::GPUImplConsts::get().deviceTracingAddr;
    }
#endif

    static inline void resetIndex()
    {
#ifdef MADRONA_TRACING
        DeviceTracing::get().resetIndex_();
#endif
    }

    static inline void Log([[maybe_unused]] DeviceEvent event,
                           [[maybe_unused]] uint32_t func_id,
                           [[maybe_unused]] uint32_t num_invocations,
                           [[maybe_unused]] uint32_t node_id)
    {
#ifdef MADRONA_TRACING
        DeviceTracing::get().LogImpl(event, func_id, num_invocations, node_id, threadIdx.x == 0);
#endif
    }

    static inline void Log([[maybe_unused]] DeviceEvent event,
                           [[maybe_unused]] uint32_t func_id,
                           [[maybe_unused]] uint32_t num_invocations,
                           [[maybe_unused]] uint32_t node_id,
                           [[maybe_unused]] bool is_leader)
    {
#ifdef MADRONA_TRACING
        DeviceTracing::get().LogImpl(event, func_id, num_invocations, node_id, is_leader);
#endif
    }

private:
    __forceinline__ uint64_t globalTimer()
    {
        uint64_t timestamp;
        asm volatile("mov.u64 %0, %%globaltimer;"
                     : "=l"(timestamp));
        return timestamp;
    }

    inline void resetIndex_()
    {
        cur_index_.store_release(0);
    }


    inline void LogImpl(DeviceEvent event, uint32_t func_id,
                        uint32_t num_invocations, uint32_t node_id, bool is_leader)
    {
        if (is_leader) {
            if (getIndex() >= 0) {
                uint32_t sm_id;
                asm("mov.u32 %0, %smid;"
                    : "=r"(sm_id));
                uint32_t log_index = cur_index_.fetch_add_relaxed(1);
                if (log_index >= maxLogSize) {
                    // mark the current set of traces to be corrupted
                    cur_index_.store_release(-1);
                } else{
                    device_logs_[log_index] = {event, func_id, num_invocations, node_id, threadIdx.x / 32, blockIdx.x + blockIdx.y * gridDim.x, sm_id, log_index, globalTimer()};
                }
            }
        }
    }
#endif

    friend class DeviceTracingManager;
};

}
