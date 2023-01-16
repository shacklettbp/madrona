/*
 * Copyright 2021-2022 Brennan Shacklett and contributors
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 */
#include <madrona/table.hpp>

#include <cstring>
#include <type_traits>

namespace madrona {

namespace ICfg {
inline constexpr uint32_t maxRowsPerTable = 1u << 28u;
}

Table::Table(const TypeInfo *component_types, CountT num_components,
             CountT init_num_rows)
    : num_rows_(init_num_rows),
      columns_()
{
    for (int i = 0; i < (int)num_components; i++) {
        const TypeInfo &type = component_types[i];

        // 3rd argument is offsetting the start from the page aligned boundary
        // to avoid everything mapping to the same cache sets. Should revisit -
        // maybe add a random offset for each Table as well?
        columns_.emplace_back(type.numBytes, type.alignment,
            MADRONA_CACHE_LINE * (i + 1), ICfg::maxRowsPerTable);
    }

    if (num_rows_ > 0) {
        for (VirtualStore &col : columns_) {
            col.expand(num_rows_);
        }
    }
}

uint32_t Table::addRow()
{
    uint32_t idx = num_rows_++;

    for (VirtualStore &col : columns_) {
        col.expand(num_rows_);
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

    for (VirtualStore &col : columns_) {
        col.shrink(num_rows_);
    }

    return need_move;
}

void Table::copyRow(uint32_t dst, uint32_t src)
{
    for (VirtualStore &col : columns_) {
        memcpy(col[dst], col[src], col.numBytesPerItem());
    }
}

void Table::clear()
{
    num_rows_ = 0;

    for (VirtualStore &col : columns_) {
        col.shrink(num_rows_);
    }
}

}
