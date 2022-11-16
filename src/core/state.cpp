/*
 * Copyright 2021-2022 Brennan Shacklett and contributors
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 */
#include <madrona/state.hpp>
#include <madrona/utils.hpp>
#include <madrona/dyn_array.hpp>

#include <cassert>
#include <functional>
#include <mutex>
#include <string_view>

#include <madrona/impl/id_map_impl.inl>

namespace madrona {

namespace ICfg {
static constexpr uint32_t maxQueryOffsets = 100'000;
}

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

#ifdef MADRONA_MW_MODE
StateManager::StateManager(int num_worlds)
    : entity_store_(),
      component_infos_(0),
      archetype_components_(0),
      archetype_stores_(0),
      num_worlds_(num_worlds),
      register_lock_()
{
    registerComponent<Entity>();
    registerComponent<WorldID>();
}
#else
StateManager::StateManager()
    : entity_store_(),
      component_infos_(0),
      archetype_components_(0),
      archetype_stores_(0)
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
    Table &tbl =
#ifdef MADRONA_MW_MODE
        archetype.tbls[world_id];
#else
        archetype.tbl;
#endif

    bool row_moved = tbl.removeRow(loc.row);

    if (row_moved) {
        Entity moved_entity = ((Entity *)tbl.data(0))[loc.row];
        entity_store_.setRow(moved_entity, loc.row);
    }

    entity_store_.freeEntity(cache.entity_cache_, e);
}

struct StateManager::ArchetypeStore::Init {
    uint32_t componentOffset;
    uint32_t numComponents;
    uint32_t id;
    Span<TypeInfo> types;
    Span<IntegerMapPair> lookupInputs;
#ifdef MADRONA_MW_MODE
    uint32_t numTables;
#endif
};

StateManager::ArchetypeStore::ArchetypeStore(Init &&init)
    : componentOffset(init.componentOffset),
      numComponents(init.numComponents),
#ifdef MADRONA_MW_MODE
      tbls(init.numTables),
#else
      tbl(init.types.data(), init.types.size()),
#endif
      columnLookup(init.lookupInputs.data(), init.lookupInputs.size())
{
#ifdef MADRONA_MW_MODE
    for (int i = 0; i < (int)init.numTables; i++) {
        tbls.emplace(i, init.types.data(), init.types.size());
    }
#endif
}

StateManager::QueryState::QueryState()
    : lock(),
      queryData(0, ICfg::maxQueryOffsets)
{}

void StateManager::makeQuery(const ComponentID *components,
                             uint32_t num_components,
                             QueryRef *query_ref)
{
    std::lock_guard lock(query_state_.lock);

    if (query_ref->numMatchingArchetypes != ~0u) {
        return;
    }

    // FIXME: should reclaim dropped queries in this function by 
    // recording query_ref and finding unused ranges (numReferences == 0)

    // Enough tmp space for 1 archetype with the maximum number of components
    StackArray<uint32_t, 1 + max_archetype_components_> tmp_query_indices;

    auto saveTmpIndices = [&](uint32_t cur_offset) {
        assert(query_state_.queryData.size() + tmp_query_indices.size() <
               ICfg::maxQueryOffsets);

        query_state_.queryData.resize(cur_offset + tmp_query_indices.size(), [](auto) {});
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
            } else {
                tmp_query_indices.push_back(archetype.columnLookup[component.id]);
            }
        }
    }

    saveTmpIndices(cur_offset);

    query_ref->offset = query_offset;
    query_ref->numMatchingArchetypes = matching_archetypes;
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

void StateManager::registerArchetype(uint32_t id, Span<ComponentID> components)
{
    uint32_t offset = archetype_components_.size();
    uint32_t num_user_components = components.size();

    uint32_t num_total_components = num_user_components + 1;
#ifdef MADRONA_MW_MODE
    num_total_components += 1;
#endif

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

    for (int i = 0; i < (int)num_user_components; i++) {
        ComponentID component_id = components[i];

        archetype_components_.push_back(component_id);
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
        offset,
        uint32_t(components.size()),
        id,
        Span(type_infos.data(), num_total_components),
        Span(lookup_input.data(), num_user_components),
        MADRONA_MW_COND(num_worlds_,)
    });
}

void StateManager::clear(MADRONA_MW_COND(uint32_t world_id,)
                         StateCache &cache, uint32_t archetype_id)
{
    ArchetypeStore &archetype = *archetype_stores_[archetype_id];
    Table &tbl =
#ifdef MADRONA_MW_MODE
        archetype.tbls[world_id];
#else
        archetype.tbl;
#endif

    // Free all IDs before deleting the table
    Entity *entities = (Entity *)tbl.data(0);
    uint32_t num_entities = tbl.numRows();
    entity_store_.bulkFree(cache.entity_cache_, entities, num_entities);

    tbl.clear();
}

StateManager::QueryState StateManager::query_state_ = StateManager::QueryState();

uint32_t StateManager::next_component_id_ = 0;
uint32_t StateManager::next_archetype_id_ = 0;

}
