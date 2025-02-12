/*
 * Copyright 2021-2022 Brennan Shacklett and contributors
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 */
#include <madrona/state.hpp>
#include <madrona/registry.hpp>
#include <madrona/utils.hpp>
#include <madrona/dyn_array.hpp>

#include <cassert>
#include <functional>
#include <mutex>
#include <string_view>

#ifndef MADRONA_WINDOWS
#include <sys/mman.h>
#include <unistd.h>
#endif

#include <madrona/impl/id_map_impl.inl>

#include <madrona/math.hpp>

namespace madrona {

namespace ICfg {
static constexpr uint32_t maxQueryOffsets = 100'000;
}

ECSRegistry::ECSRegistry(StateManager *state_mgr, void **export_ptrs)
    : state_mgr_(state_mgr),
      export_ptrs_(export_ptrs)
{}

template <typename T>
EntityStore::LockedMapStore<T>::LockedMapStore(CountT init_capacity)
    : store(sizeof(T), alignof(T), 0, ~0u),
      numIDs(init_capacity),
      expandLock()
{
    if (init_capacity > 0) {
        store.expand(init_capacity);
    }
}

template <typename T>
CountT EntityStore::LockedMapStore<T>::expand(CountT num_new_elems)
{
    std::lock_guard lock(expandLock);

    CountT offset = numIDs;

    numIDs += num_new_elems;
    store.expand(numIDs);

    return offset;
}

EntityStore::EntityStore()
    : map_(0)
{}

Entity EntityStore::newEntity(Cache &cache)
{
    return map_.acquireID(cache);
}

void EntityStore::freeEntity(Cache &cache, Entity e)
{
    map_.releaseID(cache, e);
}

void EntityStore::bulkFree(Cache &cache, Entity *entities,
                           uint32_t num_entities)
{
    map_.bulkRelease(cache, entities, num_entities);
}

StateCache::StateCache()
    : entity_cache_()
{}

StateManager::TmpAllocator::TmpAllocator()
    : cur_block_((Block *)rawAllocAligned(sizeof(Block), 256))
{
    cur_block_->metadata.next = nullptr;
    cur_block_->metadata.offset = 0;
}

StateManager::TmpAllocator::~TmpAllocator()
{
    reset();
}

void * StateManager::TmpAllocator::alloc(uint64_t num_bytes)
{
    num_bytes = utils::roundUpPow2(num_bytes, 256);

    assert(num_bytes <= numFreeBlockBytes);

    CountT cur_offset = cur_block_->metadata.offset;
    if (num_bytes > numFreeBlockBytes - cur_offset) {
        Block *new_block = (Block *)rawAllocAligned(sizeof(Block), 256);
        new_block->metadata.next = cur_block_;
        cur_block_ = new_block;
        cur_offset = 0;
    }

    void *ptr = &cur_block_->data[0] + cur_offset;

    cur_block_->metadata.offset = cur_offset + num_bytes;

    return ptr;
}

void StateManager::TmpAllocator::reset()
{
    Block *cur_block = cur_block_;
    Block *next_block;
    while ((next_block = cur_block->metadata.next) != nullptr) {
        rawDeallocAligned(cur_block);
        cur_block = next_block;
    }

    cur_block->metadata.offset = 0;
    cur_block_ = cur_block;
}

#ifdef MADRONA_MW_MODE
StateManager::StateManager(CountT num_worlds)
    : init_state_cache_(),
      entity_store_(),
      component_infos_(0),
      archetype_components_(0),
      archetype_stores_(0),
      bundle_components_(0),
      bundle_infos_(0),
      export_jobs_(0),
      tmp_allocators_(num_worlds),
      num_worlds_(num_worlds),
      register_lock_()
{
    registerComponent<Entity>();
    registerComponent<WorldID>();

    for (CountT i = 0; i < num_worlds; i++) {
        tmp_allocators_.emplace(i);
    }
}
#else
StateManager::StateManager()
    : entity_store_(),
      component_infos_(0),
      archetype_components_(0),
      archetype_stores_(0),
      bundle_components_(0),
      bundle_infos_(0),
      tmp_allocator_()
{
    registerComponent<Entity>();
}
#endif

Transaction StateManager::makeTransaction()
{
    return Transaction();
}

void StateManager::commitTransaction(Transaction &&txn)
{
    (void)txn;
}

void StateManager::destroyEntity(MADRONA_MW_COND(uint32_t world_id,)
                                 Transaction &txn, StateCache &cache, Entity e)
{
#ifdef MADRONA_MW_MODE
    (void)world_id;
#endif
    (void)txn.head;
    (void)cache;
    (void)e;
}

void StateManager::destroyEntityNow(MADRONA_MW_COND(uint32_t world_id,)
                                    StateCache &cache, Entity e)
{
    Loc loc = entity_store_.getLoc(e);
    
    if (!loc.valid()) {
        return;
    }

    ArchetypeStore &archetype = *archetype_stores_[loc.archetype];

    archetype.tblStorage.column<Entity>(
        MADRONA_MW_COND(world_id,) 0)[loc.row] = Entity::none();

    entity_store_.freeEntity(cache.entity_cache_, e);
}

#ifdef MADRONA_MW_MODE
StateManager::TableStorage::TableStorage(Span<TypeInfo> types,
                                         CountT num_worlds,
                                         CountT max_num_per_world)
{
    maxNumPerWorld = max_num_per_world;

    if (max_num_per_world == 0) {
        new (&tbls) HeapArray<Table>(num_worlds);

        for (CountT i = 0; i < num_worlds; i++) {
            tbls.emplace(i, types.data(), types.size(), 0);
        }
    } else {
        new (&fixed) Fixed {
            Table(types.data(), types.size(),
                  max_num_per_world * num_worlds),
            HeapArray<int32_t>(num_worlds),
        };

        for (CountT i = 0; i < num_worlds; i++) {
            fixed.activeRows[i] = 0;
        }
    }
}

StateManager::TableStorage::~TableStorage()
{
    if (maxNumPerWorld == 0) {
        tbls.~HeapArray<Table>();
    } else {
        fixed.~Fixed();
    }
}

#else
StateManager::TableStorage::TableStorage(Span<TypeInfo> types)
    : tbl(types.data(), types.size(), 0)
{}
#endif

struct StateManager::ArchetypeStore::Init {
    uint32_t componentOffset;
    uint32_t numComponents;
    uint32_t id;
    Span<TypeInfo> types;
    Span<IntegerMapPair> lookupInputs;
    CountT maxNumEntitiesPerWorld;
#ifdef MADRONA_MW_MODE
    CountT numWorlds;
#endif
};

StateManager::ArchetypeStore::ArchetypeStore(Init &&init)
    : componentOffset(init.componentOffset),
      numComponents(init.numComponents),
      tblStorage(init.types
          MADRONA_MW_COND(, init.numWorlds, init.maxNumEntitiesPerWorld)),
      columnLookup(init.lookupInputs.data(), init.lookupInputs.size())
{}

StateManager::QueryState::QueryState()
    : lock(),
      queryData(0, ICfg::maxQueryOffsets)
{}

void StateManager::makeQuery(const ComponentID *components,
                             uint32_t num_components,
                             QueryRef *query_ref)
{
    std::lock_guard lock(query_state_.lock);

    if (query_ref->numMatchingArchetypes != 0xFFFF'FFFF) {
        return;
    }

    // FIXME: should reclaim dropped queries in this function by 
    // recording query_ref and finding unused ranges (numReferences == 0)

    // Enough tmp space for 1 archetype with the maximum number of components
    InlineArray<uint32_t, 1 + max_archetype_components_> tmp_query_indices;

    auto saveTmpIndices = [&](uint32_t cur_offset) {
        assert(query_state_.queryData.size() + tmp_query_indices.size() <
               ICfg::maxQueryOffsets);

        query_state_.queryData.resize(
            cur_offset + (uint32_t)tmp_query_indices.size(), [](auto) {});
        memcpy(&query_state_.queryData[cur_offset], tmp_query_indices.data(),
               sizeof(uint32_t) * tmp_query_indices.size());
    };

    const uint32_t query_offset = query_state_.queryData.size();
    uint32_t cur_offset = query_offset;

    uint32_t matching_archetypes = 0;
    for (int archetype_idx = 0, num_archetypes = archetype_stores_.size();
         archetype_idx < num_archetypes; archetype_idx++) {
        if (!archetype_stores_[archetype_idx].has_value()) {
            continue;
        }

        auto &archetype = *archetype_stores_[archetype_idx];

        bool has_components = true;
        for (int component_idx = 0; component_idx < (int)num_components; 
             component_idx++) {
            ComponentID component = components[component_idx];
            if (component.id == componentID<Entity>().id) {
                continue;
            }

            if (!archetype.columnLookup.exists(component.id)) {
                has_components = false;
                break;
            }
        }

        if (!has_components) {
            continue;
        }

        if (tmp_query_indices.size() + 1 + num_components >
                tmp_query_indices.capacity()) {
            assert(tmp_query_indices.size() > 0);

            saveTmpIndices(cur_offset);
            cur_offset += tmp_query_indices.size();
            tmp_query_indices.clear();
        }

        matching_archetypes += 1;
        tmp_query_indices.push_back(uint32_t(archetype_idx));

        int component_idx;
        for (component_idx = 0; component_idx < (int)num_components;
             component_idx++) {
            ComponentID component = components[component_idx];
            assert(component.id != TypeTracker::unassignedTypeID);
            if (component.id == componentID<Entity>().id) {
                tmp_query_indices.push_back(0);
            } 
#ifdef MADRONA_MW_MODE
            else if (component.id == componentID<WorldID>().id) {
                tmp_query_indices.push_back(1);
            }
#endif
            else {
                tmp_query_indices.push_back(archetype.columnLookup[component.id]);
            }
        }
    }

    saveTmpIndices(cur_offset);

    query_ref->offset = query_offset;
    query_ref->numMatchingArchetypes = matching_archetypes;
    query_ref->numComponents = num_components;
}

void StateManager::registerComponent(uint32_t id,
                                     uint32_t alignment,
                                     uint32_t num_bytes)
{
    // IDs are globally assigned, technically there is an edge case where
    // there are gaps in the IDs assigned to a specific StateManager
    if (id >= component_infos_.size()) {
        component_infos_.resize(id + 1, [](auto ptr) {
            Optional<TypeInfo>::noneAt(ptr);
        });
    }

    component_infos_[id].emplace(TypeInfo {
        .alignment = alignment,
        .numBytes = num_bytes,
    });
}

void StateManager::registerArchetype(uint32_t id,
                                     ArchetypeFlags archetype_flags,
                                     CountT max_num_entities_per_world,
                                     CountT num_user_components,
                                     const ComponentID *components,
                                     const ComponentFlags *component_flags)
{
    (void)archetype_flags, (void)component_flags;

    std::array<TypeInfo, max_archetype_components_> type_infos;
    std::array<IntegerMapPair, max_archetype_components_> lookup_input;

    TypeInfo *type_ptr = type_infos.data();

    // Add entity column as first column of every table
    *type_ptr = *component_infos_[0];
    type_ptr++;

#ifdef MADRONA_MW_MODE
    *type_ptr = *component_infos_[componentID<WorldID>().id];
    type_ptr++;
#endif

    CountT user_component_start = archetype_components_.size();

    for (CountT i = 0; i < (CountT)num_user_components; i++) {
        uint32_t component_id = components[i].id;
        assert(component_id != TypeTracker::unassignedTypeID);

        if ((component_id & bundle_typeid_mask_) != 0) {
            uint32_t bundle_id = component_id & ~bundle_typeid_mask_;
            BundleInfo bundle_info = *bundle_infos_[bundle_id];

            for (CountT j = 0; j < (CountT)bundle_info.numComponents; j++) {
                uint32_t bundle_component_id =
                    bundle_components_[bundle_info.componentOffset + j];

                archetype_components_.push_back(
                    ComponentID { bundle_component_id });
            }
        } else {
            archetype_components_.push_back(ComponentID {component_id});
        }
    }

    CountT num_total_user_components =
        archetype_components_.size() - user_component_start;

    CountT num_total_components =
        user_component_offset_ + num_total_user_components;

    for (CountT i = 0; i < num_total_user_components; i++) {
        ComponentID component_id =
            archetype_components_[i + user_component_start];

        type_ptr[i] = *component_infos_[component_id.id];

        lookup_input[i] = IntegerMapPair {
            .key = component_id.id,
            .value = (uint32_t)i + user_component_offset_,
        };
    }

    // IDs are globally assigned, technically there is an edge case where
    // there are gaps in the IDs assigned to a specific StateManager
    if (archetype_stores_.size() <= id) {
        archetype_stores_.resize(id + 1, [](auto ptr) {
            Optional<ArchetypeStore>::noneAt(ptr);
        });
    }

    archetype_stores_[id].emplace(ArchetypeStore::Init {
        uint32_t(user_component_start),
        uint32_t(num_total_user_components),
        id,
        Span(type_infos.data(), num_total_components),
        Span(lookup_input.data(), num_total_user_components),
        max_num_entities_per_world,
        MADRONA_MW_COND(num_worlds_,)
    });
}

void StateManager::registerBundle(uint32_t id,
                                  const uint32_t *components,
                                  CountT num_components)
{
    id &= ~bundle_typeid_mask_;

    uint32_t bundle_offset = bundle_components_.size();

    for (CountT i = 0; i < num_components; i++) {
        uint32_t component_type_id = components[i];
        assert(component_type_id != TypeTracker::unassignedTypeID);

        if ((component_type_id & bundle_typeid_mask_) != 0) {
            uint32_t sub_bundle_id = component_type_id & ~bundle_typeid_mask_;
            BundleInfo sub_bundle_info = *bundle_infos_[sub_bundle_id];

            for (CountT j = 0; j < (CountT)sub_bundle_info.numComponents; j++) {
                uint32_t bundle_component_id = 
                    bundle_components_[sub_bundle_info.componentOffset + j];
                bundle_components_.push_back(bundle_component_id);
            }
        } else {
            bundle_components_.push_back(component_type_id);
        }
    }

    if (bundle_infos_.size() <= (CountT)id) {
        bundle_infos_.resize(id + 1, [](auto ptr) {
            Optional<BundleInfo>::noneAt(ptr);
        });
    }

    uint32_t num_flattened_components =
        (uint32_t)bundle_components_.size() - bundle_offset;

    bundle_infos_[id] = BundleInfo {
        .componentOffset = bundle_offset,
        .numComponents = num_flattened_components,
    };
}

void * StateManager::exportColumn(uint32_t archetype_id, uint32_t component_id)
{
    auto &archetype = *archetype_stores_[archetype_id];
    uint32_t col_idx;

    if (component_id == componentID<Entity>().id) {
        col_idx = 0;
    }
#ifdef MADRONA_MW_MODE
    else if (component_id == componentID<WorldID>().id) {
        col_idx = 1;
    }
#endif
    else {
        col_idx = *archetype.columnLookup.lookup(component_id);
    }

#ifdef MADRONA_MW_MODE
    if (archetype.tblStorage.maxNumPerWorld == 0) {

        uint32_t num_bytes_per_row = component_infos_[component_id]->numBytes;
        uint64_t map_size = 1'000'000'000 * num_bytes_per_row;

        VirtualRegion mem(map_size, 0, 1);
        void *export_buffer = mem.ptr();

        export_jobs_.push_back(ExportJob {
            .archetypeIdx = archetype_id,
            .columnIdx = col_idx,
            .numBytesPerRow = num_bytes_per_row,
            .numMappedChunks = 0,
            .mem = std::move(mem),
        });

        return export_buffer;
    } else {
        return archetype.tblStorage.fixed.tbl.data(col_idx);
    }
#else
    return archetype.tblStorage.tbl.data(col_idx);
#endif
}

void StateManager::copyInExportedColumns()
{
#ifdef MADRONA_MW_MODE
    for (ExportJob &export_job : export_jobs_) {
        auto &archetype = *archetype_stores_[export_job.archetypeIdx];

        CountT cumulative_copied_rows = 0;
        for (Table &tbl : archetype.tblStorage.tbls) {
            CountT num_rows = tbl.numRows();

            if (num_rows == 0) {
                continue;
            }

            CountT tbl_start = cumulative_copied_rows;

            cumulative_copied_rows += num_rows;

            memcpy(tbl.data(export_job.columnIdx),
                   (char *)export_job.mem.ptr() +
                       tbl_start * export_job.numBytesPerRow,
                   export_job.numBytesPerRow * num_rows);
        }
    }
#endif
}

void StateManager::copyOutExportedColumns()
{
#ifdef MADRONA_MW_MODE
    for (ExportJob &export_job : export_jobs_) {
        auto &archetype = *archetype_stores_[export_job.archetypeIdx];

        CountT cumulative_copied_rows = 0;
        for (Table &tbl : archetype.tblStorage.tbls) {
            CountT num_rows = tbl.numRows();

            if (num_rows == 0) {
                continue;
            }

            CountT tbl_start = cumulative_copied_rows;
            cumulative_copied_rows += num_rows;

            uint64_t num_mapped_chunks = export_job.numMappedChunks;
            uint64_t num_mapped_bytes =
                 num_mapped_chunks * export_job.mem.chunkSize();

            uint64_t num_needed_bytes = (uint64_t)cumulative_copied_rows *
                (uint64_t)export_job.numBytesPerRow;

            if (num_needed_bytes > num_mapped_bytes) {
                uint64_t new_num_mapped_bytes =
                    std::max(num_mapped_bytes * 2, num_needed_bytes);

                uint64_t new_num_chunks = utils::divideRoundUp(
                    new_num_mapped_bytes, export_job.mem.chunkSize());

                export_job.mem.commitChunks(num_mapped_chunks,
                    new_num_chunks - num_mapped_chunks);
                export_job.numMappedChunks = new_num_chunks;
            }

            memcpy((char *)export_job.mem.ptr() +
                       tbl_start * export_job.numBytesPerRow,
                   tbl.data(export_job.columnIdx),
                   export_job.numBytesPerRow * num_rows);
        }
    }
#endif
}

void StateManager::clear(MADRONA_MW_COND(uint32_t world_id,)
                         StateCache &cache, uint32_t archetype_id,
                         bool is_temporary)
{
    ArchetypeStore &archetype = *archetype_stores_[archetype_id];

    // Free all IDs before deleting the table
    if (!is_temporary) {
        Entity *entities = archetype.tblStorage.column<Entity>(
            MADRONA_MW_COND(world_id,) 0);
        uint32_t num_entities = archetype.tblStorage.numRows(
            MADRONA_MW_COND(world_id));
        entity_store_.bulkFree(cache.entity_cache_, entities, num_entities);
    }

    archetype.tblStorage.clear(MADRONA_MW_COND(world_id));
}


void * StateManager::tmpAlloc(MADRONA_MW_COND(uint32_t world_id,)
                              uint64_t num_bytes)
{
#ifdef MADRONA_MW_MODE
    return tmp_allocators_[world_id].alloc(num_bytes);
#else
    return tmp_allocator_.alloc(num_bytes);
#endif
}

void StateManager::resetTmpAlloc(MADRONA_MW_COND(uint32_t world_id))
{
#ifdef MADRONA_MW_MODE
    tmp_allocators_[world_id].reset();
#else
    tmp_allocator_.reset();
#endif
}

void StateManager::sortArchetype(
    MADRONA_MW_COND(uint32_t world_id,)
    uint32_t archetype_id,
    uint32_t component_id)
{
    ArchetypeStore &archetype_store = *archetype_stores_[archetype_id];

    Table &tbl =
#ifndef MADRONA_MW_MODE
        archetype_store.tblStorage.tbl;
#else
      archetype_store.tblStorage.tbls[world_id];
    assert(archetype_store.tblStorage.maxNumPerWorld == 0);
#endif

    uint32_t num_rows = tbl.numRows();
    uint32_t sort_col_idx = archetype_store.columnLookup[component_id];
    uint32_t *sort_keys = (uint32_t *)tbl.data(sort_col_idx);

    uint32_t *idxs = (uint32_t *)malloc(sizeof(uint32_t) * num_rows);

    for (uint32_t i = 0; i < num_rows; i++) {
        idxs[i] = i;
    }

    std::stable_sort(idxs, idxs + num_rows, [&](uint32_t a_idx, uint32_t b_idx) {
        return sort_keys[a_idx] < sort_keys[b_idx];
    });

    uint32_t num_cols = archetype_store.numComponents + user_component_offset_;
    uint32_t max_column_num_bytes = 0;

    for (uint32_t col_idx = 0; col_idx < num_cols; col_idx++) {
        uint32_t num_bytes = tbl.columnNumBytes(col_idx);
        if (num_bytes > max_column_num_bytes) {
            max_column_num_bytes = num_bytes;
        }
    }

    void *column_staging = malloc((uint64_t)max_column_num_bytes * num_rows);

    for (uint32_t orig_idx = 0; orig_idx < num_rows; orig_idx++) {
        uint32_t new_idx = idxs[orig_idx];
        Entity e = ((Entity *)tbl.data(0))[orig_idx];
        entity_store_.setRow(e, new_idx);
    }

    for (uint32_t col_idx = 0; col_idx < num_cols; col_idx++) {
      void *col_dst = tbl.data(col_idx);
      uint32_t col_num_bytes = tbl.columnNumBytes(col_idx);
      memcpy(column_staging, col_dst,
             (uint64_t)col_num_bytes * num_rows);
      for (uint32_t orig_idx = 0; orig_idx < num_rows; orig_idx++) {
          uint32_t new_idx = idxs[orig_idx];

          memcpy((char *)col_dst + (uint64_t)new_idx * col_num_bytes,
                 (char *)column_staging + (uint64_t)orig_idx * col_num_bytes,
                 col_num_bytes);
      }
    }

    free(column_staging);
    free(idxs);
}

void StateManager::compactArchetype(
    MADRONA_MW_COND(uint32_t world_id,)
    uint32_t archetype_id)
{
    ArchetypeStore &archetype_store = *archetype_stores_[archetype_id];

    Table &tbl =
#ifndef MADRONA_MW_MODE
        archetype_store.tblStorage.tbl;
#else
      archetype_store.tblStorage.tbls[world_id];
    assert(archetype_store.tblStorage.maxNumPerWorld == 0);
#endif

    Entity *entity_col = (Entity *)tbl.data(0);

    uint32_t num_rows = tbl.numRows();
    uint32_t num_cols = archetype_store.numComponents + user_component_offset_;
    uint32_t num_compacted_rows = 0;

    for (uint32_t orig_row_idx = 0; orig_row_idx < num_rows; orig_row_idx++) {
        Entity e = entity_col[orig_row_idx];

        if (e == Entity::none()) {
            continue;
        }

        uint32_t new_row_idx = num_compacted_rows++;

        if (new_row_idx == orig_row_idx) {
            // Avoid memcpy on same / overlapping data.
            continue;
        }

        entity_store_.setRow(e, new_row_idx);

        for (uint32_t col_idx = 0; col_idx < num_cols; col_idx++) {
            uint32_t num_bytes = tbl.columnNumBytes(col_idx);
            memcpy(tbl.getValue(col_idx, new_row_idx),
                   tbl.getValue(col_idx, orig_row_idx),
                   num_bytes);
        }
    }

    tbl.setNumRows(num_compacted_rows);
}

StateManager::QueryState StateManager::query_state_ = StateManager::QueryState();

uint32_t StateManager::next_component_id_ = 0;
uint32_t StateManager::next_archetype_id_ = 0;
uint32_t StateManager::next_bundle_id_ = StateManager::bundle_typeid_mask_;

}
