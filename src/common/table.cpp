/*
 * Copyright 2021-2022 Brennan Shacklett and contributors
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 */
#include <madrona/table.hpp>

#include <algorithm>
#include <cstring>
#include <type_traits>

namespace madrona {

namespace ICfg {
inline constexpr uint32_t maxRowsPerTable = 1u << 28u;
}

Table::Table(const TypeInfo *component_types, CountT num_components,
             CountT init_num_rows)
    : num_rows_(init_num_rows),
      num_allocated_rows_(std::max(uint32_t(init_num_rows), 1_u32)),
      num_components_(num_components),
      columns_(),
      bytes_per_column_()
{
    for (int i = 0; i < (int)num_components; i++) {
        const TypeInfo &type = component_types[i];

#if 0
        // 3rd argument is offsetting the start from the page aligned boundary
        // to avoid everything mapping to the same cache sets. Should revisit -
        // maybe add a random offset for each Table as well?
        columns_.emplace_back(type.numBytes, type.alignment,
            MADRONA_CACHE_LINE * (i + 1), ICfg::maxRowsPerTable);
#endif

        size_t column_bytes_per_row = (size_t)type.numBytes;
        columns_[i] = malloc(
            (size_t)column_bytes_per_row * (size_t)num_allocated_rows_);
        bytes_per_column_[i] = column_bytes_per_row;
    }
}

uint32_t Table::addRow()
{
    uint32_t idx = num_rows_++;

    if (idx >= num_allocated_rows_) {
        uint32_t new_num_rows =
            std::max(std::max(10_u32, uint32_t(num_allocated_rows_ * 2)), idx);

        for (int i = 0; i < (int)num_components_; i++) {
            columns_[i] = realloc(columns_[i],
                uint64_t(new_num_rows) * uint64_t(bytes_per_column_[i]));
        }

        num_allocated_rows_ = new_num_rows;
    }

    return idx;
}

bool Table::removeRow(uint32_t row)
{
    uint32_t from_idx = --num_rows_;
    uint32_t to_idx = row;

    bool need_move = from_idx != to_idx;
    if (need_move) {
        copyRow(to_idx, from_idx);
    }

    return need_move;
}

void Table::copyRow(uint32_t dst, uint32_t src)
{
    for (int i = 0; i < (int)num_components_; i++) {
        memcpy(getValue(i, dst), getValue(i, src), bytes_per_column_[i]);
    }
}

void Table::clear()
{
    num_rows_ = 0;
}

}
