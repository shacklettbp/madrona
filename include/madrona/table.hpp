#pragma once

#include <madrona/memory.hpp>
#include <madrona/heap_array.hpp>
#include <madrona/stack_array.hpp>
#include <madrona/virtual.hpp>
#include <madrona/ecs.hpp>

#include <array>

namespace madrona {

struct TypeInfo {
    uint32_t alignment;
    uint32_t numBytes;
};

class Table {
public:
    Table(const TypeInfo *component_types, uint32_t num_components);

    uint32_t addRow();
    bool removeRow(uint32_t row);

    inline void * getValue(uint32_t column_idx, uint32_t row);
    inline const void * getValue(uint32_t column_idx, uint32_t row) const;

    inline void * data(uint32_t col_idx);
    inline const void * data(uint32_t col_idx) const;

    inline uint32_t numRows() const { return num_rows_; }

    // Drops all rows in the table and frees memory
    void reset();

    static constexpr uint32_t maxColumns = 128;

private:
    uint32_t num_rows_;
    StackArray<VirtualStore, maxColumns> columns_;
};

}

#include "table.inl"
