/*
 * Copyright 2021-2022 Brennan Shacklett and contributors
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 */
#pragma once

#include <madrona/memory.hpp>
#include <madrona/heap_array.hpp>
#include <madrona/inline_array.hpp>
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
    Table(const TypeInfo *component_types, CountT num_components,
          CountT init_num_rows);

    uint32_t addRow();
    bool removeRow(uint32_t row);
    void copyRow(uint32_t dst, uint32_t src);

    inline void * getValue(uint32_t column_idx, uint32_t row);
    inline const void * getValue(uint32_t column_idx, uint32_t row) const;

    inline void * data(uint32_t col_idx);
    inline const void * data(uint32_t col_idx) const;

    inline uint32_t numRows() const;
    inline void setNumRows(uint32_t num_rows);

    // Drops all rows in the table and frees memory
    void clear();

    inline uint32_t columnNumBytes(uint32_t col_idx) const;

    static constexpr uint32_t maxColumns = 128;

private:
    uint32_t num_rows_;
    uint32_t num_allocated_rows_;
    uint32_t num_components_;
    InlineArray<void *, maxColumns> columns_;
    InlineArray<uint32_t, maxColumns> bytes_per_column_;
};

}

#include "table.inl"
