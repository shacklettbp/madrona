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
    Table(const TypeInfo *component_types, uint32_t num_components,
          uint32_t table_id);

    Entity addRow();
    void removeRow(Entity e);

    inline Loc getLoc(Entity e) const;
    inline void * getValue(uint32_t column_idx, Loc loc);
    inline const void * getValue(uint32_t column_idx, Loc loc) const;

    inline void * data(uint32_t col_idx);
    inline const void * data(uint32_t col_idx) const;

    inline uint32_t numRows() const { return num_rows_; }

    static constexpr uint32_t maxColumns = 128;

private:
    struct GenIndex {
        uint32_t idx;
        uint32_t gen : 18;
        uint32_t pad : 14;
    };

    inline uint32_t idToIndex(Entity e) const;

    inline Entity makeID(uint32_t idx);
    inline void freeID(uint32_t id);

    uint32_t table_id_;
    uint32_t num_rows_;
    uint32_t free_id_head_;

    VirtualStore id_to_idx_;
    uint32_t num_ids_;
    StackArray<VirtualStore, maxColumns> columns_;
};

}

#include "table.inl"
