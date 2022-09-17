#include <madrona/state.hpp>
#include <madrona/utils.hpp>
#include <madrona/dyn_array.hpp>

#include <cassert>
#include <functional>
#include <mutex>
#include <string_view>

namespace madrona {

namespace ICfg {
static constexpr uint32_t maxQueryOffsets = 100'000;
}

IDManager::IDManager()
    : store_(sizeof(GenLoc), alignof(GenLoc), 0, ~0u),
      num_ids_(0),
      free_id_head_(~0u)
{}

Entity IDManager::newEntity(uint32_t archetype, uint32_t row)
{
    if (free_id_head_ != ~0u) {
        uint32_t new_id = free_id_head_;
        GenLoc *new_loc = getGenLoc(new_id);
        free_id_head_ = new_loc->loc.row;

        new_loc->loc = Loc {
            .archetype = archetype,
            .row = row,
        };

        return Entity {
            .gen = new_loc->gen,
            .id = new_id,
        };
    }

    uint32_t id = num_ids_++;
    store_.expand(num_ids_);

    GenLoc *new_loc = getGenLoc(id);

    *new_loc = {
        .loc = Loc {
            .archetype = archetype,
            .row = row,
        },
        .gen = 0,
    };

    return Entity {
        .gen = 0,
        .id = id,
    };
}

void IDManager::freeEntity(Entity e)
{
    GenLoc *gen_loc = getGenLoc(e.id);
    gen_loc->gen++;

    gen_loc->loc.row = free_id_head_;
    free_id_head_ = e.id;
}

void IDManager::bulkFree(Entity *entities, uint32_t num_entities)
{
    uint32_t prev = free_id_head_;
    for (int idx = 0, n = num_entities; idx < n; idx++) {
        Entity e = entities[idx];
        GenLoc *gen_loc = getGenLoc(e.id);
        gen_loc->gen++;
        gen_loc->loc.row = prev;
 
        prev = e.id;
    }
    free_id_head_ = prev;
}

#ifdef MADRONA_MW_MODE
StateManager::StateManager(int num_worlds)
    : id_mgr_(),
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
    : id_mgr_(),
      component_infos_(0),
      archetype_components_(0),
      archetype_stores_(0)
{
    registerComponent<Entity>();
}
#endif

void StateManager::destroyEntity(Entity e)
{
    Loc loc = id_mgr_.getLoc(e);
    
    if (!loc.valid()) {
        return;
    }

    ArchetypeStore &archetype = *archetype_stores_[loc.archetype];
    bool row_moved = archetype.tbl.removeRow(loc.row);

    if (row_moved) {
        Entity moved_entity = ((Entity *)archetype.tbl.data(0))[loc.row];
        id_mgr_.updateLoc(moved_entity, loc.row);
    }

    id_mgr_.freeEntity(e);
}

struct StateManager::ArchetypeStore::Init {
    uint32_t componentOffset;
    uint32_t numComponents;
    uint32_t id;
    Span<TypeInfo> types;
    Span<IntegerMapPair> lookupInputs;
};

StateManager::ArchetypeStore::ArchetypeStore(Init &&init)
    : componentOffset(init.componentOffset),
      numComponents(init.numComponents),
      tbl(init.types.data(), init.types.size()),
      columnLookup(init.lookupInputs.data(), init.lookupInputs.size())
{
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
        components.size(),
        id,
        Span(type_infos.data(), num_total_components),
        Span(lookup_input.data(), num_user_components),
    });
}

void StateManager::reset(uint32_t archetype_id)
{
    auto &archetype = *archetype_stores_[archetype_id];

    // Free all IDs before deleting the table
    Entity *entities = (Entity *)archetype.tbl.data(0);
    uint32_t num_entities = archetype.tbl.numRows();
    id_mgr_.bulkFree(entities, num_entities);

    archetype.tbl.reset();
}

StateManager::QueryState StateManager::query_state_ = StateManager::QueryState();

uint32_t StateManager::next_component_id_ = 0;
uint32_t StateManager::next_archetype_id_ = 0;

}
