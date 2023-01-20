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
    // todo: have a smaller value by enabling log transfering between steps
    // actually, for 8k worlds, even a single step might generate over 16 MB log data
    // instead of trying to collect all logs
    // it might be better to clean the data we can get
    // for this 10M logs, ~300MB GPU memory is needed
    static constexpr inline uint64_t maxLogSize = 10000000;

    struct DeviceLog {
        DeviceEvent event;
        uint32_t funcID;
        uint32_t numInvocations;
        uint32_t nodeID;
        uint32_t blockID;
        uint32_t smID;
        uint64_t cycleCount;
    };

    std::atomic_uint32_t cur_index_;
    DeviceLog device_logs_[maxLogSize];

public:
    inline DeviceTracing()
        : cur_index_(0),
          device_logs_ {}
    {}

    inline uint32_t getIndex()
    {
        return cur_index_.load(std::memory_order_relaxed);
    }

    inline void resetIndex()
    {
        cur_index_.store(0, std::memory_order_release);
    }

#ifdef MADRONA_GPU_MODE

#ifdef MADRONA_TRACING
    static inline DeviceTracing & get()
    {
        return *(DeviceTracing *)mwGPU::GPUImplConsts::get().deviceTracingAddr;
    }
#endif

    static inline void Log([[maybe_unused]] DeviceEvent event,
                           [[maybe_unused]] uint32_t func_id,
                           [[maybe_unused]] uint32_t num_invocations,
                           [[maybe_unused]] uint32_t node_id)
    {
#ifdef MADRONA_TRACING
        DeviceTracing::get().LogImpl(event, func_id, num_invocations, node_id);
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

    inline void LogImpl(DeviceEvent event, uint32_t func_id,
                        uint32_t num_invocations, uint32_t node_id)
    {
        if (threadIdx.x == 0)
        {
            uint32_t sm_id;
            asm("mov.u32 %0, %smid;"
                : "=r"(sm_id));
            uint32_t log_index = cur_index_.fetch_add(1, std::memory_order_relaxed);
            if (log_index >= maxLogSize)
            {
                log_index = 0;
                resetIndex();
            }
            device_logs_[log_index] = {event, func_id, num_invocations, node_id, blockIdx.x, sm_id, globalTimer()};
        }
    }
#endif

    friend class DeviceTracingManager;
};

}
