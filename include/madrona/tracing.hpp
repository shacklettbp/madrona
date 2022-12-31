#pragma once

#include <vector>
#include <iostream>

namespace madrona
{
    enum HostEvent
    {
        initStart = 0, // todo: may further split up if necessary
        initEnd = 1,
        megaKernelStart = 2,
        megaKernelEnd = 3,
        renderStart = 4, // todo: further split up after gaining a better understanding
        renderEnd = 5,
    };

    enum DeviceEvent
    {
        TBD,
    };

    struct HostTracing
    {
        // todo: replace vectors with pre-allocated memory pool for lover overhead
        std::vector<HostEvent> events;
        std::vector<int64_t> time_stamps;
    };

    struct DeviceTracing
    {
        std::vector<DeviceEvent> events;
    };

    // TLS is used for easy access from both MWCudaExecutor and applications such as hindseek
    extern thread_local HostTracing HOST_TRACING;

    inline int64_t GetTimeStamp() { return __builtin_ia32_rdtsc(); }

    inline void HostEventLogging(HostEvent event)
    {
#ifdef MADRONA_TRACING
        HOST_TRACING.events.push_back(event);
        HOST_TRACING.time_stamps.push_back(GetTimeStamp());
#endif
    }

    void HostEventLogging(HostEvent event);

    void FinalizeLogging(const std::string file_path);
} // namespace madrona