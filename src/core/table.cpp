#include <madrona/table.hpp>

#include <cstring>
#include <type_traits>

namespace madrona {

namespace ICfg {
inline constexpr uint32_t maxRowsPerTable = 1u << 30u;
}

Table::Table(const TypeInfo *component_types, uint32_t num_components, uint32_t table_id)
    : table_id_(table_id),
      num_rows_(0),
      free_id_head_(~0u),
      id_to_idx_(sizeof(GenIndex), alignof(GenIndex), 0, ICfg::maxRowsPerTable),
      num_ids_(0),
      idx_to_id_(sizeof(Entity), alignof(Entity), 128, ICfg::maxRowsPerTable),
      columns_()
{
    for (int i = 0; i < (int)num_components; i++) {
        const TypeInfo &type = component_types[i];

        // 3rd argument is offsetting the start from the page aligned boundary
        // to avoid everything mapping to the same cache sets. Should revisit -
        // maybe add a random offset for each Table as well?
        columns_.emplace_back(type.numBytes, type.alignment, 128 * (i + 2),
                              ICfg::maxRowsPerTable);
    }
}

Entity Table::addRow()
{
    uint32_t idx = num_rows_++;

    for (VirtualStore &col : columns_) {
        col.expand(num_rows_);
    }

    idx_to_id_.expand(num_rows_);

    Entity e = makeID(idx);

    *(Entity *)idx_to_id_[idx] = e;

    return e;
}

void Table::removeRow(Entity e)
{
    uint32_t delete_idx = idToIndex(e);
    if (delete_idx == ~0u) {
        return;
    }

    uint32_t from_idx = --num_rows_;
    uint32_t to_idx = delete_idx;

    if (from_idx != to_idx) {
        Entity update_entity = *(Entity *)idx_to_id_[from_idx];

        (*(GenIndex *)id_to_idx_[update_entity.id]).idx = to_idx;
        *(Entity *)idx_to_id_[to_idx] = update_entity;

        for (VirtualStore &col : columns_) {
            memcpy(col[to_idx], col[from_idx], col.numBytesPerItem());
        }
    }

    freeID(e.id);

    for (VirtualStore &col : columns_) {
        col.shrink(num_rows_);
    }

    idx_to_id_.shrink(num_rows_);
}

Entity Table::makeID(uint32_t idx)
{
    if (free_id_head_ != ~0u) {
        uint32_t new_id = free_id_head_;
        GenIndex &old = *(GenIndex *)id_to_idx_[new_id];
        free_id_head_ = old.idx;
        old.idx = idx;

        return Entity {
            old.gen,
            table_id_,
            new_id,
        };
    }

    uint32_t id = num_ids_++;
    id_to_idx_.expand(num_ids_);

    *(GenIndex *)id_to_idx_[id] = {
        idx,
        0,
        0,
    };

    return Entity {
        0,
        table_id_,
        id,
    };
}

void Table::freeID(uint32_t id)
{
    GenIndex &idx_map = *(GenIndex *)id_to_idx_[id];
    idx_map.idx = free_id_head_;
    idx_map.gen++;
    free_id_head_ = id;
}

}
