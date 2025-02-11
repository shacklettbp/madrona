/*
 * Copyright 2021-2022 Brennan Shacklett and contributors
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 */
#include <madrona/state.hpp>
#include <madrona/registry.hpp>
#include <madrona/mw_gpu/megakernel_consts.hpp>
#include <madrona/mw_gpu/host_print.hpp>

#include <algorithm>

namespace madrona {

Table::Table()
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

    assert(new_num_rows < tbl.reservedRows);

    int32_t min_mapped_rows = tbl.reservedRows;

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

ECSRegistry::ECSRegistry(StateManager *state_mgr, void **export_ptr)
    : state_mgr_(state_mgr),
      export_ptrs_(export_ptr)
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

    entity_store_.numGrows = 1;

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

StateManager::ArchetypeStore::ArchetypeStore(
        uint32_t offset,
        ArchetypeFlags archetype_flags,
        uint32_t max_num_entities_per_world,
        uint32_t num_user_components,
        uint32_t num_columns,
        TypeInfo *type_infos,
        IntegerMapPair *lookup_input,
        ComponentFlags *component_flags)
    : componentOffset(offset),
      numUserComponents(num_user_components),
      tbl(),
      columnLookup(lookup_input, num_user_components),
      worldOffsets(nullptr),
      worldCounts(nullptr),
      needsSort(false)
{
    using namespace mwGPU;

    uint32_t num_worlds = GPUImplConsts::get().numWorlds;
    uint32_t max_num_entities = max_num_entities_per_world * num_worlds;
    HostAllocator *alloc = getHostAllocator();

    tbl.numColumns = num_columns;
    tbl.numRows.store_relaxed(0);

    uint64_t bytes_per_row = 0;
    for (int32_t i = 0; i < (int32_t)num_columns; i++) {
        uint64_t col_row_bytes = type_infos[i].numBytes;
        bytes_per_row += col_row_bytes;
    }
    
    uint64_t num_reserved_rows = Table::maxReservedBytesPerTable /
        bytes_per_row;
    num_reserved_rows = min(num_reserved_rows, 0x7FFF'FFFF_u64);
    
    tbl.reservedRows = (int32_t)num_reserved_rows;

    int32_t min_mapped_rows = (int32_t)num_reserved_rows;

    uint32_t max_column_size = 0;

    tbl.columnFlags[0] = ComponentFlags::None;
    tbl.columnFlags[1] = ComponentFlags::None;
    for (int32_t i = 0; i < (int32_t)num_user_components; i++) {
        tbl.columnFlags[num_columns - num_user_components + i] =
            component_flags[i];
    }

    for (int32_t i = 0; i < (int32_t)num_columns; i++) {
        uint64_t col_row_bytes = type_infos[i].numBytes;

        uint64_t reserve_bytes = col_row_bytes * num_reserved_rows;
        reserve_bytes = alloc->roundUpReservation(reserve_bytes);

        uint64_t init_bytes = col_row_bytes * (uint64_t)num_worlds;
        init_bytes = alloc->roundUpAlloc(init_bytes);

        ComponentFlags component_flag = tbl.columnFlags[i];
        if ((component_flag & ComponentFlags::ImportMemory) !=
                ComponentFlags::ImportMemory) {
            if (max_num_entities == 0) {
                tbl.columns[i] =
                    alloc->reserveMemory(reserve_bytes, init_bytes);
                tbl.columnFlags[i] |= ComponentFlags::CudaReserveMemory;
            } else {
                uint64_t alloc_bytes = col_row_bytes * max_num_entities;
                tbl.columns[i] = alloc->allocMemory(alloc_bytes);
                tbl.columnFlags[i] |= ComponentFlags::CudaAllocMemory;
            }
        }

        tbl.columnSizes[i] = col_row_bytes;
        tbl.columnMappedBytes[i] = init_bytes;

        max_column_size = std::max((uint32_t)col_row_bytes, max_column_size);

        int32_t num_mapped_in_column = init_bytes / col_row_bytes;

        min_mapped_rows = min(num_mapped_in_column, min_mapped_rows);
    }

    if ((archetype_flags & ArchetypeFlags::ImportOffsets) !=
            ArchetypeFlags::ImportOffsets) {
        // Allocate space for the sorting offsets (one offset per world)
        uint64_t bytes = (uint64_t)sizeof(int32_t) * (uint64_t)num_worlds;
        bytes = alloc->roundUpReservation(bytes);
        worldOffsets = (int32_t *)alloc->allocMemory(bytes);
    }

    flags = archetype_flags;

    // Allocate space for the counts (one count per world)
    uint64_t bytes = (uint64_t)sizeof(int32_t) * (uint64_t)num_worlds;
    bytes = alloc->roundUpReservation(bytes);
    worldCounts = (int32_t *)alloc->allocMemory(bytes);

    tbl.maxColumnSize = max_column_size;

    if (max_num_entities == 0) {
        tbl.mappedRows = min_mapped_rows;
    } else {
        tbl.mappedRows = max_num_entities;
    }
}

void StateManager::registerArchetype(uint32_t id,
                                     ArchetypeFlags archetype_flags,
                                     uint32_t max_num_entities_per_world,
                                     ComponentID *components,
                                     ComponentFlags *component_flags,
                                     uint32_t num_user_components)
{
    std::array<TypeInfo, max_archetype_components_> type_infos;
    std::array<IntegerMapPair, max_archetype_components_> lookup_input;
    std::array<ComponentFlags, max_archetype_components_> flattened_flags;

    TypeInfo *type_ptr = type_infos.data();
    IntegerMapPair *lookup_input_ptr = lookup_input.data();

    // Add entity column as first column of every table
    *type_ptr = *components_[0];
    type_ptr++;

    // Add world ID column as second column of every table
    *type_ptr = *components_[1];
    type_ptr++;

    uint32_t flattened_user_component_start = archetype_component_offset_;
    uint32_t flag_offset = 0;
    for (int i = 0; i < (int)num_user_components; i++) {
        uint32_t component_id = components[i].id;
        assert(component_id != TypeTracker::unassignedTypeID);

        if ((component_id & bundle_typeid_mask_) != 0) {
            uint32_t bundle_id = component_id & ~bundle_typeid_mask_;
            BundleInfo bundle_info = bundle_infos_[bundle_id];

            for (CountT j = 0; j < (CountT)bundle_info.numComponents; j++) {
                uint32_t bundle_component_id =
                    bundle_components_[bundle_info.componentOffset + j];

                archetype_components_[archetype_component_offset_++] =
                    bundle_component_id;

                flattened_flags[flag_offset++] = component_flags[i];
            }
        } else {
            archetype_components_[archetype_component_offset_++] = component_id;
            flattened_flags[flag_offset++] = component_flags[i];
        }
    }
    uint32_t num_flattened_user_components =
        archetype_component_offset_ - flattened_user_component_start;

    for (int i = 0; i < (int)num_flattened_user_components; i++) {
        uint32_t component_id = archetype_components_[
            flattened_user_component_start + i];

        type_ptr[i] = *components_[component_id];

        lookup_input_ptr[i] = IntegerMapPair {
            /* .key = */   component_id,
            /* .value = */ (uint32_t)i + user_component_offset_,
        };
    }

    uint32_t num_total_components =
        num_flattened_user_components + user_component_offset_;

    archetypes_[id].emplace(flattened_user_component_start,
                            archetype_flags,
                            max_num_entities_per_world,
                            num_flattened_user_components,
                            num_total_components,
                            type_infos.data(),
                            lookup_input.data(),
                            flattened_flags.data());
}


void StateManager::registerBundle(uint32_t id,
                                  const uint32_t *components,
                                  CountT num_components)
{
    id &= ~bundle_typeid_mask_;

    uint32_t bundle_offset = bundle_component_offset_;

    for (CountT i = 0; i < num_components; i++) {
        uint32_t component_type_id = components[i];
        assert(component_type_id != TypeTracker::unassignedTypeID);

        if ((component_type_id & bundle_typeid_mask_) != 0) {
            uint32_t sub_bundle_id = component_type_id & ~bundle_typeid_mask_;
            BundleInfo sub_bundle_info = bundle_infos_[sub_bundle_id];

            for (CountT j = 0; j < (CountT)sub_bundle_info.numComponents; j++) {
                uint32_t bundle_component_id = 
                    bundle_components_[sub_bundle_info.componentOffset + j];
                bundle_components_[bundle_component_offset_++] = 
                    bundle_component_id;
            }
        } else {
            bundle_components_[bundle_component_offset_++] = component_type_id;
        }
    }

    uint32_t num_flattened_components =
        bundle_component_offset_ - bundle_offset;

    bundle_infos_[id] = BundleInfo {
        .componentOffset = bundle_offset,
        .numComponents = num_flattened_components,
    };
}

void StateManager::makeQuery(const uint32_t *components,
                             uint32_t num_components,
                             QueryRef *query_ref)
{
    query_data_lock_.lock();

    if (query_ref->numMatchingArchetypes != 0xFFFF'FFFF) {
        query_data_lock_.unlock();
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
        entity_store.availableOffset.fetch_add_relaxed(1);

    if (available_idx < entity_store.numMappedEntities) [[likely]] {
        return entity_store.availableEntities[available_idx];
    }

    entity_store.growLock.lock();

    if (available_idx < entity_store.numMappedEntities) {
        entity_store.growLock.unlock();

        return entity_store.availableEntities[available_idx];
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

    ++entity_store.numGrows;

    for (int32_t i = 0; i < entity_store.numGrowEntities; i++) {
        int32_t idx = i + entity_store.numMappedEntities;
        entity_store.entities[idx].gen = 0;
        entity_store.availableEntities[idx] = idx;
    }

    entity_store.numMappedEntities += entity_store.numGrowEntities;

    entity_store.growLock.unlock();

    return entity_store.availableEntities[available_idx];
}

Entity StateManager::makeEntityNow(WorldID world_id, uint32_t archetype_id)
{
    auto &archetype = *archetypes_[archetype_id];
    archetype.needsSort = true;
    Table &tbl = archetype.tbl;

    int32_t row = tbl.numRows.fetch_add_relaxed(1);

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
    auto &archetype = *archetypes_[archetype_id];

    Table &tbl = archetype.tbl;
    archetype.needsSort = true;

    int32_t row = tbl.numRows.fetch_add_relaxed(1);

    if (row >= tbl.mappedRows) {
        growTable(tbl, row);
    }

    Loc loc {
        archetype_id,
        row,
    };

    // FIXME: Shouldn't even need to write entity IDs
    // for tmp entities (or have the column), but the sort
    // needs a sentinel value written here so it doesn't try to do
    // invalid entity ID remapping
    Entity *entity_column = (Entity *)tbl.columns[0];
    entity_column[row] = Entity::none();

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
        entity_store_.deletedOffset.fetch_add_relaxed(1);

    entity_store_.deletedEntities[deleted_offset] = e.id;
}

void StateManager::clearTemporaries(uint32_t archetype_id)
{
    auto &archetype = *archetypes_[archetype_id];

    Table &tbl = archetype.tbl;
    auto num_rows_bef = 
        tbl.numRows.exchange<sync::memory_order::relaxed>(0);

    if (num_rows_bef != 0) {
        archetype.needsSort = true;
    }
}

void StateManager::resizeArchetype(uint32_t archetype_id, int32_t num_rows)
{
    archetypes_[archetype_id]->tbl.numRows.store_relaxed(num_rows);
}

int32_t StateManager::numArchetypeRows(uint32_t archetype_id) const
{
    return archetypes_[archetype_id]->tbl.numRows.load_relaxed();
}

std::pair<int32_t, int32_t> StateManager::fetchRecyclableEntities()
{
    int32_t num_deleted = entity_store_.deletedOffset.load_relaxed();

    int32_t available_end = entity_store_.availableOffset.load_relaxed();

    int32_t recycle_base = available_end - num_deleted;

    if (num_deleted > 0) {
        entity_store_.deletedOffset.store_relaxed(0);
        entity_store_.availableOffset.store_relaxed(recycle_base);
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

void StateManager::freeTables()
{
    mwGPU::HostAllocator *host_alloc = mwGPU::getHostAllocator();

    for (int i = 0; i < archetypes_.size(); ++i) {
        if (archetypes_[i].has_value()) {
            auto &arch_store = *(archetypes_[i]);

            if ((arch_store.flags & ArchetypeFlags::ImportOffsets) !=
                ArchetypeFlags::ImportOffsets) {
                host_alloc->allocFree(arch_store.worldOffsets);
                host_alloc->allocFree(arch_store.worldCounts);
            }

            uint64_t bytes_per_row = 0;
            for (int col = 0; col < (int)arch_store.tbl.numColumns; col++) {
                uint64_t col_row_bytes = arch_store.tbl.columnSizes[col];
                bytes_per_row += col_row_bytes;
            }

            for (int col = 0; col < (int32_t)arch_store.tbl.numColumns; ++col) {
                if ((arch_store.tbl.columnFlags[col] & 
                            ComponentFlags::ImportMemory) !=
                    ComponentFlags::ImportMemory) {
                    if ((arch_store.tbl.columnFlags[col] & 
                                ComponentFlags::CudaReserveMemory) ==
                        ComponentFlags::CudaReserveMemory) {
                        uint64_t num_reserved_rows = Table::maxReservedBytesPerTable /
                            bytes_per_row;
                        num_reserved_rows = min(num_reserved_rows, 0x7FFF'FFFF_u64);

                        uint64_t col_row_bytes = arch_store.tbl.columnSizes[col];

                        uint64_t reserve_bytes = col_row_bytes * num_reserved_rows;
                        reserve_bytes = host_alloc->roundUpReservation(reserve_bytes);

                        host_alloc->reserveFree(
                                arch_store.tbl.columns[col],
                                arch_store.tbl.columnMappedBytes[col],
                                reserve_bytes);
                    } else if ((arch_store.tbl.columnFlags[col] & 
                                ComponentFlags::CudaAllocMemory) ==
                        ComponentFlags::CudaAllocMemory) {
                        host_alloc->allocFree(arch_store.tbl.columns[col]);
                    }
                }
            }
        }
    }

    { // Free the entity store
        uint64_t num_slot_bytes = entity_store_.numGrows *
            entity_store_.numSlotGrowBytes;
        uint64_t num_idx_bytes = entity_store_.numGrows *
            entity_store_.numIdxGrowBytes;

        uint32_t max_mapped_entities = 2'147'483'648;

        uint64_t reserve_entity_bytes =
            (uint64_t)max_mapped_entities * sizeof(EntityStore::EntitySlot);
        uint64_t reserve_idx_bytes =
            (uint64_t)max_mapped_entities * sizeof(int32_t);

        reserve_entity_bytes = host_alloc->roundUpReservation(reserve_entity_bytes);
        reserve_idx_bytes = host_alloc->roundUpReservation(reserve_idx_bytes);

        host_alloc->reserveFree(
                entity_store_.entities, num_slot_bytes, reserve_entity_bytes);
        host_alloc->reserveFree(
                entity_store_.availableEntities, num_idx_bytes, reserve_entity_bytes);
        host_alloc->reserveFree(
                entity_store_.deletedEntities, num_idx_bytes, reserve_entity_bytes);
    }
}

}

extern "C" __global__ void freeECSTables()
{
    using namespace madrona;
    StateManager *mgr = mwGPU::getStateManager();
    mgr->freeTables();
}
