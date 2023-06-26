#pragma once

#include <vector>
#include <string>
#include <stdint.h>

#include <madrona/macros.hpp>

#ifdef MADRONA_MSVC
#include <intrin.h>
#endif

namespace madrona
{
    enum class HostEvent : uint32_t
    {
        initStart = 0, // todo: may further split up if necessary
        initEnd = 1,
        megaKernelStart = 2,
        megaKernelEnd = 3,
        renderStart = 4, // todo: further split up after gaining a better understanding
        renderEnd = 5,
    };

    struct HostTracing
    {
        // todo: replace vectors with pre-allocated memory pool for lover overhead
        std::vector<HostEvent> events;
        std::vector<uint64_t> time_stamps;
    };

    // TLS is used for easy access from both MWCudaExecutor and applications such as hindseek
    extern thread_local HostTracing HOST_TRACING;

    // may replace this with chrono or clock_gettime for better portability
    inline uint64_t GetTimeStamp()
    {
#if defined(MADRONA_X64)
#if defined(MADRONA_MSVC)
        return __rdtsc();
#else
        return __builtin_ia32_rdtsc();
#endif
#elif defined(MADRONA_ARM)
        uint64_t val;
        asm volatile("mrs %0, cntvct_el0" : "=r" (val));
        return val;
#else
        STATIC_UNIMPLEMENTED();
#endif
    }

    inline void HostEventLogging([[maybe_unused]] HostEvent event)
    {
#ifdef MADRONA_TRACING
        HOST_TRACING.events.push_back(event);
        HOST_TRACING.time_stamps.push_back(GetTimeStamp());
#endif
    }

    void WriteToFile(void *data, size_t num_bytes,
                     const std::string &file_path,
                     const std::string &name);

    // FIXME: WriteToFile not a good name for global madrona scope
    template <typename T>
    inline void WriteToFile(T *events, size_t size,
                            const std::string &file_path,
                            const std::string &name)
    {
        WriteToFile((void *)events, size * sizeof(T), file_path, name);
    }

    void FinalizeLogging(const std::string file_path);
} // namespace madrona
