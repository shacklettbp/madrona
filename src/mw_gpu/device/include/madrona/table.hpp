#pragma once

#include <array>
#include <cstdint>

namespace madrona {

struct TypeInfo {
    uint32_t alignment;
    uint32_t numBytes;
};

struct Table {
    static constexpr uint32_t maxColumns = 128;

    std::array<void *, maxColumns> columns;

    int32_t numRows;
};

}
