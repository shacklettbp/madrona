#include <madrona/table.hpp>

#include <cstring>
#include <type_traits>

namespace madrona {

namespace ICfg {
inline constexpr uint32_t maxRowsPerTable = 1u << 30u;
}

Table::Table(const TypeInfo *component_types, uint32_t num_components)
    : num_rows_(0),
      columns_()
{
    for (int i = 0; i < (int)num_components; i++) {
        const TypeInfo &type = component_types[i];

        // 3rd argument is offsetting the start from the page aligned boundary
        // to avoid everything mapping to the same cache sets. Should revisit -
        // maybe add a random offset for each Table as well?
        columns_.emplace_back(type.numBytes, type.alignment, 128 * (i + 1),
                              ICfg::maxRowsPerTable);
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
        for (VirtualStore &col : columns_) {
            memcpy(col[to_idx], col[from_idx], col.numBytesPerItem());
        }
    }

    for (VirtualStore &col : columns_) {
        col.shrink(num_rows_);
    }

    return need_move;
}

void Table::reset()
{
    num_rows_ = 0;

    for (VirtualStore &col : columns_) {
        col.shrink(num_rows_);
    }
}

}
