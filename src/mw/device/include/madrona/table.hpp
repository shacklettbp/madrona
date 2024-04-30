#pragma once

#include <array>
#include <atomic>
#include <cstdint>

#include <madrona/sync.hpp>
#include <madrona/types.hpp>
#include <madrona/ecs_flags.hpp>

namespace madrona {

struct TypeInfo {
    uint32_t alignment;
    uint32_t numBytes;
};

struct Table {
    Table();

    static constexpr uint32_t maxColumns = 128;

    std::array<void *, maxColumns> columns;

    // FIXME: move a lot of this metadata out of the core table struct
    std::array<uint32_t, maxColumns> columnSizes;
    std::array<uint64_t, maxColumns> columnMappedBytes;
    std::array<ComponentFlags, maxColumns> columnFlags;

    uint32_t maxColumnSize;
    int32_t numColumns;

    AtomicI32 numRows;
    int32_t reservedRows;
    int32_t mappedRows;
    SpinLock growLock;
     
    static inline constexpr uint64_t maxReservedBytesPerTable =
        128_u64 * 1024_u64 * 1024_u64 * 1024_u64;
};

}
