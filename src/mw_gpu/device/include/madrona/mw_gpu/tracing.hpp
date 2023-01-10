#pragma once

#include <stdint.h>
#include <atomic>

// for current setting, it is required to be larger than steps * (#blocks + #nodes)
#define NUM_EVENT_LOG 10000000

namespace madrona
{
    namespace mwGPU
    {

        class DeviceTracingAllocator;

        // todo: expand to log the activity of every block instead of only nodes
        enum class DeviceEvent : uint32_t
        {
            calibration = 0,
            nodeStart = 1,
            nodeFinish = 2,
        };

        class DeviceTracing
        {
        private:
            struct DeviceLog
            {
                DeviceEvent event;
                uint32_t funcID;
                uint32_t numInvocations;
                uint32_t nodeID;
                uint32_t blockID;
                uint32_t smID;
                int64_t cycleCount;
            };

            std::atomic_uint32_t cur_index_;
            DeviceLog device_logs_[NUM_EVENT_LOG];

        public:
            inline DeviceTracing()
            {
            }
            void DeviceEventLogging(DeviceEvent event, uint32_t func_id, uint32_t num_invocations, uint32_t node_id);

            void resetIndex();

            uint32_t getIndex();

            friend class DeviceTracingAllocator;
        };

    }

}