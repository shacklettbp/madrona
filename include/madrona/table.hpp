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

    struct Index {
        uint32_t idx;

        inline bool valid() const { return idx != ~0u; }
    };

    inline Index getIndex(Entity e) const;
    inline void *getValue(uint32_t column_idx, Index idx);
    inline const void *getValue(uint32_t column_idx, Index idx) const;

    inline uint32_t numRows() const { return num_rows_; }

    static constexpr uint32_t maxComponents = 64;

private:
    struct GenIndex {
        uint32_t idx;
        uint32_t gen : 18;
        uint32_t pad : 14;
    };

    inline uint32_t idToIndex(Entity e) const;

    Entity makeID(uint32_t idx);
    void freeID(uint32_t id);

    uint32_t table_id_;
    uint32_t num_rows_;
    uint32_t free_id_head_;

    VirtualStore id_to_idx_;
    uint32_t num_ids_;
    VirtualStore idx_to_id_;
    StackArray<VirtualStore, maxComponents> columns_;
};

}

#include "table.inl"
