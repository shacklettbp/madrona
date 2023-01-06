#pragma once

#include <vector>
#include <string>
#include <stdint.h>

#include <madrona/macros.hpp>

namespace madrona
{
    enum class HostEvent : uint8_t
    {
        initStart = 0, // todo: may further split up if necessary
        initEnd = 1,
        megaKernelStart = 2,
        megaKernelEnd = 3,
        renderStart = 4, // todo: further split up after gaining a better understanding
        renderEnd = 5,
    };

    enum class DeviceEvent
    {
        TBD,
    };

    struct HostTracing
    {
        // todo: replace vectors with pre-allocated memory pool for lover overhead
        std::vector<HostEvent> events;
        std::vector<uint64_t> time_stamps;
    };

    struct DeviceTracing
    {
        std::vector<DeviceEvent> events;
    };

    // TLS is used for easy access from both MWCudaExecutor and applications such as hindseek
    extern thread_local HostTracing HOST_TRACING;

    // may replace this with chrono or clock_gettime for better portability
    inline uint64_t GetTimeStamp()
    {
#ifdef MADRONA_X64
        return __builtin_ia32_rdtsc();
#else
        // todo
        return 0;
#endif
    }

    inline void HostEventLogging([[maybe_unused]] HostEvent event)
    {
#ifdef MADRONA_TRACING
        HOST_TRACING.events.push_back(event);
        HOST_TRACING.time_stamps.push_back(GetTimeStamp());
#endif
    }

    void FinalizeLogging(const std::string file_path);
} // namespace madrona
