#pragma once

#include <madrona/mw_gpu/tracing.hpp>

namespace madrona
{
    namespace mwGPU
    {

        void DeviceTracing::DeviceEventLogging(DeviceEvent event, uint32_t func_id, uint32_t num_invocations, uint32_t node_id)
        {
            uint32_t sm_id;
            asm("mov.u32 %0, %smid;"
                : "=r"(sm_id));
            uint32_t log_index = cur_index_.fetch_add(1, std::memory_order_relaxed);
            if (log_index >= NUM_EVENT_LOG)
            {
                cur_index_.store(0, std::memory_order_relaxed);
                log_index = 0;
            }
            device_logs_[log_index] = {event, func_id, num_invocations, node_id, blockIdx.x, sm_id, clock64()};
        }

        void DeviceTracing::resetIndex()
        {
            cur_index_.store(0, std::memory_order_release);
        }
        uint32_t DeviceTracing::getIndex()
        {
            return cur_index_.load(std::memory_order_relaxed);
        }

    } // namespace mwGPU

}
