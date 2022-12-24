/*
 * Copyright 2021-2022 Brennan Shacklett and contributors
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 */
#include <madrona/state.hpp>

#include "megakernel_consts.hpp"

namespace madrona {

static MADRONA_NO_INLINE void growTable(Table &tbl, int32_t row)
{
    using namespace mwGPU;

    tbl.growLock.lock();

    if (tbl.mappedRows > row) {
        tbl.growLock.unlock();
        return;
    }

    HostAllocator *alloc = getHostAllocator();
    
    int32_t new_num_rows = tbl.mappedRows * 2;

    if (new_num_rows - tbl.mappedRows > 500'000) {
        new_num_rows = tbl.mappedRows + 500'000;
    }

    int32_t min_mapped_rows = Table::maxRowsPerTable;
    for (int32_t i = 0; i < tbl.numColumns; i++) {
        void *column_base = tbl.columns[i];
        uint64_t column_bytes_per_row = tbl.columnSizes[i];
        uint64_t cur_mapped_bytes = tbl.columnMappedBytes[i];

        int32_t cur_max_rows = cur_mapped_bytes / column_bytes_per_row;

        if (cur_max_rows >= new_num_rows) {
            min_mapped_rows = min(cur_max_rows, min_mapped_rows);
            continue;
        }

        uint64_t new_mapped_bytes = column_bytes_per_row * new_num_rows;
        new_mapped_bytes = alloc->roundUpAlloc(new_mapped_bytes);

        uint64_t mapped_bytes_diff = new_mapped_bytes - cur_mapped_bytes;
        void *grow_base = (char *)column_base + cur_mapped_bytes;
        alloc->mapMemory(grow_base, mapped_bytes_diff);

        int32_t new_max_rows = new_mapped_bytes / column_bytes_per_row;
        min_mapped_rows = min(new_max_rows, min_mapped_rows);

        tbl.columnMappedBytes[i] = new_mapped_bytes;
    }

    tbl.mappedRows = min_mapped_rows;
    
    tbl.growLock.unlock();
}

ECSRegistry::ECSRegistry(StateManager &state_mgr, void **export_ptr)
    : state_mgr_(&state_mgr),
      export_ptr_(export_ptr)
{}

StateManager::StateManager(uint32_t)
{
    using namespace mwGPU;

#pragma unroll(1)
    for (int32_t i = 0; i < (int32_t)max_components_; i++) {
        Optional<TypeInfo>::noneAt(&components_[i]);
    }

    // Without disabling unrolling, this loop 100x's compilation time for
    // this file.... Optional must be really slow.
#pragma unroll(1)
    for (int32_t i = 0; i < (int32_t)max_archetypes_; i++) {
        Optional<ArchetypeStore>::noneAt(&archetypes_[i]);
    }

    // Initialize entity store
    HostAllocator *alloc = getHostAllocator();

    uint32_t init_mapped_entities = 1'048'576;
    uint32_t max_mapped_entities = 2'147'483'648;

    uint64_t reserve_entity_bytes =
        (uint64_t)max_mapped_entities * sizeof(EntityStore::EntitySlot);
    uint64_t reserve_idx_bytes =
        (uint64_t)max_mapped_entities * sizeof(int32_t);

    reserve_entity_bytes = alloc->roundUpReservation(reserve_entity_bytes);
    reserve_idx_bytes = alloc->roundUpReservation(reserve_idx_bytes);

    uint64_t init_entity_bytes =
        (uint64_t)init_mapped_entities * sizeof(EntityStore::EntitySlot);
    uint64_t init_idx_bytes =
        (uint64_t)init_mapped_entities * sizeof(int32_t);

    entity_store_.numSlotGrowBytes = alloc->roundUpAlloc(init_entity_bytes);

    entity_store_.numIdxGrowBytes = alloc->roundUpAlloc(init_idx_bytes);

    // FIXME technically we should find some kind of common multiple here
    assert(entity_store_.numSlotGrowBytes / sizeof(EntityStore::EntitySlot) ==
           init_mapped_entities);
    assert(entity_store_.numIdxGrowBytes / sizeof(int32_t) ==
           init_mapped_entities);

    entity_store_.entities = (EntityStore::EntitySlot *)alloc->reserveMemory(
        reserve_entity_bytes, entity_store_.numSlotGrowBytes);

    entity_store_.availableEntities = (int32_t *)alloc->reserveMemory(
        reserve_entity_bytes, entity_store_.numIdxGrowBytes);
    entity_store_.deletedEntities = (int32_t *)alloc->reserveMemory(
        reserve_entity_bytes, entity_store_.numIdxGrowBytes);

    entity_store_.numGrowEntities = init_mapped_entities;
    entity_store_.numMappedEntities = init_mapped_entities;

    for (int32_t i = 0; i < init_mapped_entities; i++) {
        entity_store_.entities[i].gen = 0;
        entity_store_.availableEntities[i] = i;
    }

    registerComponent<Entity>();
    registerComponent<WorldID>();
}

void StateManager::registerComponent(uint32_t id, uint32_t alignment,
                                     uint32_t num_bytes)
{
    components_[id].emplace(TypeInfo {
        /* .alignment = */ alignment,
        /* .numBytes = */  num_bytes,
    });
}

StateManager::ArchetypeStore::ArchetypeStore(uint32_t offset,
                                             uint32_t num_user_components,
                                             uint32_t num_columns,
                                             TypeInfo *type_infos,
                                             IntegerMapPair *lookup_input)
    : componentOffset(offset),
      numUserComponents(num_user_components),
      tbl {},
      columnLookup(lookup_input, num_columns),
      needsSort(false)
{
    using namespace mwGPU;

    uint32_t num_worlds = GPUImplConsts::get().numWorlds;
    HostAllocator *alloc = getHostAllocator();

    tbl.numColumns = num_columns;
    tbl.numRows.store(0, std::memory_order_relaxed);

    int32_t min_mapped_rows = Table::maxRowsPerTable;

    for (int i = 0 ; i < (int)num_columns; i++) {
        uint64_t reserve_bytes = (uint64_t)type_infos[i].numBytes *
            (uint64_t)Table::maxRowsPerTable;
        reserve_bytes = alloc->roundUpReservation(reserve_bytes);

        uint64_t init_bytes =
            (uint64_t)type_infos[i].numBytes * (uint64_t)num_worlds;
        init_bytes = alloc->roundUpAlloc(init_bytes);

        tbl.columns[i] = alloc->reserveMemory(reserve_bytes, init_bytes);
        tbl.columnSizes[i] = type_infos[i].numBytes;
        tbl.columnMappedBytes[i] = init_bytes;

        int num_mapped_in_column = init_bytes / tbl.columnSizes[i];

        min_mapped_rows = min(num_mapped_in_column, min_mapped_rows);
    }
    tbl.mappedRows = min_mapped_rows;
}

void StateManager::registerArchetype(uint32_t id, ComponentID *components,
                                     uint32_t num_user_components)
{
    uint32_t offset = archetype_component_offset_;
    archetype_component_offset_ += num_user_components;

    uint32_t num_total_components = num_user_components + 2;

    std::array<TypeInfo, max_archetype_components_> type_infos;
    std::array<IntegerMapPair, max_archetype_components_> lookup_input;

    TypeInfo *type_ptr = type_infos.data();
    IntegerMapPair *lookup_input_ptr = lookup_input.data();

    // Add entity column as first column of every table
    *type_ptr = *components_[0];
    type_ptr++;

    // Add world ID column as second column of every table
    *type_ptr = *components_[1];
    type_ptr++;

    *lookup_input_ptr = {
        TypeTracker::typeID<Entity>(),
        0,
    };
    lookup_input_ptr++;

    *lookup_input_ptr = {
        TypeTracker::typeID<WorldID>(),
        1,
    };
    lookup_input_ptr++;

    for (int i = 0; i < (int)num_user_components; i++) {
        ComponentID component_id = components[i];
        assert(component_id.id != TypeTracker::unassignedTypeID);
        archetype_components_[offset + i] = component_id.id;

        type_ptr[i] = *components_[component_id.id];

        lookup_input_ptr[i] = IntegerMapPair {
            /* .key = */   component_id.id,
            /* .value = */ (uint32_t)i + user_component_offset_,
        };
    }

    archetypes_[id].emplace(offset, num_user_components,
                            num_total_components,
                            type_infos.data(),
                            lookup_input.data());
}

void StateManager::makeQuery(const uint32_t *components,
                             uint32_t num_components,
                             QueryRef *query_ref)
{
    query_data_lock_.lock();

    if (query_ref->numMatchingArchetypes != 0xFFFF'FFFF) {
        return;
    }

    uint32_t query_offset = query_data_offset_;

    uint32_t num_matching_archetypes = 0;
    for (int32_t archetype_idx = 0; archetype_idx < (int32_t)num_archetypes_;
         archetype_idx++) {
        auto &archetype = *archetypes_[archetype_idx];

        bool has_components = true;
        for (int component_idx = 0; component_idx < (int)num_components; 
             component_idx++) {
            uint32_t component = components[component_idx];
            if (component == TypeTracker::typeID<Entity>()) {
                continue;
            }

            if (!archetype.columnLookup.exists(component)) {
                has_components = false;
                break;
            }
        }

        if (!has_components) {
            continue;
        }


        num_matching_archetypes += 1;
        query_data_[query_data_offset_++] = uint32_t(archetype_idx);

        for (int32_t component_idx = 0;
             component_idx < (int32_t)num_components; component_idx++) {
            uint32_t component = components[component_idx];
            assert(component != TypeTracker::unassignedTypeID);
            if (component == TypeTracker::typeID<Entity>()) {
                query_data_[query_data_offset_++] = 0;
            } else if (component == TypeTracker::typeID<WorldID>()) {
                query_data_[query_data_offset_++] = 1;
            } else {
                query_data_[query_data_offset_++] = 
                    archetype.columnLookup[component];
            }
        }
    }

    query_ref->offset = query_offset;
    query_ref->numMatchingArchetypes = num_matching_archetypes;
    query_ref->numComponents = num_components;

    query_data_lock_.unlock();
}

static inline int32_t getEntitySlot(EntityStore &entity_store)
{
    int32_t available_idx =
        entity_store.availableOffset.fetch_add(1, std::memory_order_relaxed);

    if (available_idx < entity_store.numMappedEntities) [[likely]] {
        return available_idx;
    }

    entity_store.growLock.lock();

    if (available_idx < entity_store.numMappedEntities) {
        entity_store.growLock.unlock();

        return available_idx;
    }

    void *entities_grow_base = (char *)entity_store.entities +
        (uint64_t)entity_store.numMappedEntities *
            sizeof(EntityStore::EntitySlot);

    void *available_grow_base =
        (char *)entity_store.availableEntities +
            (uint64_t)entity_store.numMappedEntities * sizeof(int32_t);

    void *deleted_grow_base =
        (char *)entity_store.deletedEntities +
            (uint64_t)entity_store.numMappedEntities * sizeof(int32_t);

    auto *alloc = mwGPU::getHostAllocator();
    alloc->mapMemory(entities_grow_base, entity_store.numSlotGrowBytes);
    alloc->mapMemory(available_grow_base, entity_store.numIdxGrowBytes);
    alloc->mapMemory(deleted_grow_base, entity_store.numIdxGrowBytes);

    for (int32_t i = 0; i < entity_store.numGrowEntities; i++) {
        int32_t idx = i + entity_store.numMappedEntities;
        entity_store.entities[idx].gen = 0;
        entity_store.availableEntities[idx] = idx;
    }

    entity_store.numMappedEntities += entity_store.numGrowEntities;

    entity_store.growLock.unlock();

    return available_idx;
}

Entity StateManager::makeEntityNow(WorldID world_id, uint32_t archetype_id)
{
    auto &archetype = *archetypes_[archetype_id];
    archetype.needsSort = true;
    Table &tbl = archetype.tbl;

    int32_t row = tbl.numRows.fetch_add(1, std::memory_order_relaxed);

    if (row >= tbl.mappedRows) {
        growTable(tbl, row);
    }

    Loc loc {
        archetype_id,
        row,
    };

    int32_t entity_slot_idx = getEntitySlot(entity_store_);

    EntityStore::EntitySlot &entity_slot =
        entity_store_.entities[entity_slot_idx];

    entity_slot.loc = loc;
    entity_slot.gen = 0;

    // FIXME: proper entity mapping on GPU
    Entity e {
        entity_slot.gen,
        entity_slot_idx,
    };

    Entity *entity_column = (Entity *)tbl.columns[0];
    WorldID *world_column = (WorldID *)tbl.columns[1];

    entity_column[row] = e;
    world_column[row] = world_id;

    return e;
}

Loc StateManager::makeTemporary(WorldID world_id,
                                uint32_t archetype_id)
{
    Table &tbl = archetypes_[archetype_id]->tbl;

    int32_t row = tbl.numRows.fetch_add(1, std::memory_order_relaxed);

    if (row >= tbl.mappedRows) {
        growTable(tbl, row);
    }

    Loc loc {
        archetype_id,
        row,
    };

    WorldID *world_column = (WorldID *)tbl.columns[1];
    world_column[row] = world_id;

    return loc;
}

void StateManager::destroyEntityNow(Entity e)
{
    EntityStore::EntitySlot &entity_slot =
        entity_store_.entities[e.id];

    entity_slot.gen++;
    Loc loc = entity_slot.loc;

    auto &archetype = *archetypes_[loc.archetype];
    archetype.needsSort = true;
    Table &tbl = archetype.tbl;
    WorldID *world_column = (WorldID *)tbl.columns[1];
    world_column[loc.row] = WorldID { -1 };

    int32_t deleted_offset =
        entity_store_.deletedOffset.fetch_add(1, std::memory_order_relaxed);

    entity_store_.deletedEntities[deleted_offset] = e.id;
}

void StateManager::clearTemporaries(uint32_t archetype_id)
{
    Table &tbl = archetypes_[archetype_id]->tbl;
    tbl.numRows = 0;
}

int32_t StateManager::numArchetypeRows(uint32_t archetype_id) const
{
    archetypes_[archetype_id]->tbl.numRows.load(std::memory_order_relaxed);
}

std::pair<int32_t, int32_t> StateManager::fetchRecyclableEntities()
{
    int32_t num_deleted =
        entity_store_.deletedOffset.load(std::memory_order_relaxed);

    int32_t available_end =
        entity_store_.availableOffset.load(std::memory_order_relaxed);

    int32_t recycle_base = available_end - num_deleted;

    if (num_deleted > 0) {
        entity_store_.deletedOffset.store(0, std::memory_order_relaxed);
        entity_store_.availableOffset.store(recycle_base,
                                            std::memory_order_relaxed);
    }

    return {
        recycle_base,
        num_deleted,
    };
}

void StateManager::recycleEntities(int32_t thread_offset,
                                   int32_t recycle_base)
{
    entity_store_.availableEntities[recycle_base + thread_offset] =
        entity_store_.deletedEntities[thread_offset];
}

#if 0
void StateManager::sortArchetypeBlock(uint32_t archetype_id,
                                        int32_t column_idx,
                                        int32_t invocation_idx)
{
    using BlockRadixSortT = cub::BlockRadixSort<uint32_t, 
                                                consts::numMegakernelThreads,
                                                numElementsPerSortThread,
                                                int32_t>;

    using TempStorageT = typename BlockRadixSortT::TempStorage;

    static_assert(sizeof(TempStorageT) <=
                  mwGPU::SharedMemStorage::numSMemBytes);

    auto &archetype = *archetypes_[archetype_id];

    int32_t num_rows = archetype.tbl.numRows.load(std::memory_order_relaxed);

    uint32_t keys[numElementsPerSortThread];
    int32_t vals[numElementsPerSortThread];

    uint32_t *col = (uint32_t *)archetype.tbl.columns[column_idx];

#pragma unroll
    for (int32_t i = 0; i < numElementsPerSortThread; i++) {
        // Each threadblock is responsible for loading numMegakernelThreads * 2
        // sequential items. Stride these accesses for warp coalescing
        // FIXME: There's an issue here with the indexing since the count
        // from prior threadblocks isn't accounted for
        int32_t row_idx = invocation_idx + i * consts::numMegakernelThreads;

        if (row_idx < num_rows) {
            keys[i] = col[row_idx];
            vals[i] = row_idx;
        } else {
            keys[i] = 0xFFFF'FFFF;
            vals[i] = -1;
        }
    }

    auto *sort_smem = (TempStorageT *)mwGPU::SharedMemStorage::buffer;

    BlockRadixSortT(*sort_smem).SortBlockedToStriped(keys, vals);

    auto *raw_smem = (char *)mwGPU::SharedMemStorage::buffer;

    // FIXME: can directly use keys to populate worldID column

    {
        Entity *entities = (Entity *)archetype.tbl.columns[0];
        Entity *entities_smem = (Entity *)raw_smem;

        for (int32_t i = 0; i < numElementsPerSortThread; i++) {
            int32_t orig_idx = vals[i];
            int32_t new_idx =
                invocation_idx + i * consts::numMegakernelThreads;

            bool valid = orig_idx != -1;

            __syncthreads();

            if (valid) {
                entities_smem[new_idx] = entities[new_idx];
            }

            __syncthreads();

            if (valid) {
                Entity e = entities_smem[orig_idx];
                entities[new_idx] = e;
                entity_store_.entities[e.id].loc.row = new_idx;
            }
        }
    }

    {
        WorldID *world_ids = (WorldID *)archetype.tbl.columns[1];

        for (int32_t i = 0; i < numElementsPerSortThread; i++) {
            int32_t new_idx =
                invocation_idx + i * consts::numMegakernelThreads;
            
            if (vals[i] != -1) {
                world_ids[new_idx].idx = keys[i];
            }
        }
    }

    for (int32_t col_idx = 2; col_idx < archetype.tbl.numColumns; col_idx++) {
        uint32_t elem_bytes = archetype.tbl.columnSizes[col_idx];
        char *col_base = (char *)archetype.tbl.columns[col_idx];

        assert(elem_bytes * consts::numMegakernelThreads < 
               mwGPU::SharedMemStorage::numSMemBytes);

#pragma unroll
        for (int32_t i = 0; i < numElementsPerSortThread; i++) {
            int32_t orig_idx = vals[i];
            int32_t new_idx = invocation_idx + i * consts::numMegakernelThreads;

            bool valid = orig_idx != -1;

            __syncthreads();

            if (valid) {
                memcpy(raw_smem + int64_t(elem_bytes) * int64_t(new_idx),
                       col_base + int64_t(elem_bytes) * int64_t(new_idx),
                       elem_bytes);
            } 

            __syncthreads();

            if (valid) {
                memcpy(col_base + int64_t(elem_bytes) * int64_t(new_idx),
                       raw_smem + int64_t(elem_bytes) * int64_t(orig_idx),
                       elem_bytes);
            }
        }
    }
}
#endif

}
