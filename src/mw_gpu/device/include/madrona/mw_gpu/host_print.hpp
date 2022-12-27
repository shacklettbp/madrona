#pragma once

#include <madrona/sync.hpp>

namespace madrona {
namespace mwGPU {

class HostPrintCPU;

class HostPrint {
public:
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
        cuda::std::atomic_int32_t signal;
    };

    inline HostPrint(Channel *channel);

    static void logSubmit(const char *str, void **ptrs, FmtType *types,
                          int32_t num_args);

    void logSubmitImpl(const char *str, void **ptrs, FmtType *types,
                       int32_t num_args);

    Channel *channel_;
    utils::SpinLock device_lock_;
friend HostPrintCPU;
};

}
}

#include "host_print.inl"
