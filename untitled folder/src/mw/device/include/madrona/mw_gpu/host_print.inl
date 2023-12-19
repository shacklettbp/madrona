#pragma once

#include <array>
#include <type_traits>

namespace madrona {
namespace mwGPU {

template <typename... Args>
void HostPrint::log(const char *str, Args && ...args)
{
#ifdef MADRONA_GPU_MODE
    __threadfence_system();
#endif

    auto translate_type = [](auto *ptr) {
        using T = std::decay_t<decltype(*ptr)>;

        if constexpr (std::is_same_v<T, int32_t>) {
            return FmtType::I32;
        } else if constexpr (std::is_same_v<T, uint32_t>) {
            return FmtType::U32;
        } else if constexpr (std::is_same_v<T, int64_t>) {
            return FmtType::I64;
        } else if constexpr (std::is_same_v<T, uint64_t>) {
            return FmtType::U64;
        } else if constexpr (std::is_same_v<T, float>) {
            return FmtType::Float;
        } else if constexpr (std::is_pointer_v<T>) {
            return FmtType::Ptr;
        } else {
            static_assert(!std::is_same_v<T, T>);
        }
    };

    std::array<void *, sizeof...(Args)> ptrs {
        (void *)&args
        ...
    };

    std::array<FmtType, ptrs.size()> types {
        translate_type(&args)
        ...
    };

    HostPrint::logSubmit(str, ptrs.data(), types.data(),
                         (int32_t)sizeof...(Args));

#ifdef MADRONA_GPU_MODE
    __threadfence_system();
#endif
}

}
}
