#pragma once

#include <array>
#include <atomic>
#include <cstdint>

namespace madrona {

struct TypeInfo {
    uint32_t alignment;
    uint32_t numBytes;
};

struct Table {
    static constexpr uint32_t maxColumns = 128;

    std::array<void *, maxColumns> columns;

    std::atomic_int32_t numRows;
};

}
