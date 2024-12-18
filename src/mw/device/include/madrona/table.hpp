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

template <uint32_t max_columns>
struct Table {
    inline Table()
        : columns(),
          columnSizes(),
          columnMappedBytes(),
          columnFlags(),
          maxColumnSize(),
          numColumns(),
          numRows(0),
          mappedRows(0),
          growLock()
    {}

    static constexpr uint32_t maxColumns = max_columns;

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

using ArchetypeTable = Table<128>;

// Range maps only need to keep track of 3 columns no matter the type.
// It's just the WorldID and the actual units themselves.
using RangeMapTable = Table<3>;

}
