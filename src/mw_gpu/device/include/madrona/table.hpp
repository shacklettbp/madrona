#pragma once

#include <array>
#include <atomic>
#include <cstdint>

#include <madrona/sync.hpp>
#include <madrona/types.hpp>

namespace madrona {

struct TypeInfo {
    uint32_t alignment;
    uint32_t numBytes;
};

struct Table {
    static constexpr uint32_t maxColumns = 128;

    std::array<void *, maxColumns> columns;

    // FIXME: move a lot of this metadata out of the core table struct
    std::array<uint32_t, maxColumns> columnSizes;
    std::array<uint64_t, maxColumns> columnMappedBytes;
    int32_t numColumns;

    std::atomic_int32_t numRows;
    int32_t mappedRows;
    utils::SpinLock growLock;
     
    static inline constexpr uint32_t maxRowsPerTable = 1_u32 << 30;
};

}
