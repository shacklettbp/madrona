#pragma once

#ifdef MADRONA_GPU_MODE

#include <madrona/sync.hpp>

#endif

namespace madrona {
namespace mwGPU {

class HostPrintCPU;

class HostPrint {
public:
    HostPrint(void *channel_raw);

    template <typename... Args>
    static void log(const char *str, Args && ...args);

private:
    static inline constexpr int32_t max_args = 64;
    static inline constexpr int32_t max_bytes = 4096;

    enum class FmtType : uint32_t {
        I32,
        U32,
        I64,
        U64,
        Float,
        Ptr,
    };

    struct Channel {
        char buffer[max_bytes];
        FmtType args[max_args];
        int32_t numArgs;
        cuda::atomic<int32_t, cuda::thread_scope_system> signal;
    };

    static void logSubmit(const char *str, void **ptrs, FmtType *types,
                          int32_t num_args);

    void logSubmitImpl(const char *str, void **ptrs, FmtType *types,
                       int32_t num_args);

    Channel *channel_;
    SpinLock device_lock_;
friend HostPrintCPU;
};

}
}

#ifdef MADRONA_GPU_MODE
#include "host_print.inl"
#endif
