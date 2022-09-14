#include <madrona/state.hpp>
#include <madrona/utils.hpp>
#include <madrona/dyn_array.hpp>

#include <cassert>
#include <functional>
#include <mutex>
#include <string_view>

namespace madrona {

#ifdef MADRONA_MW_MODE
StateManager::StateManager(int num_worlds)
    : query_data_(0),
      component_infos_(0),
      archetype_components_(0),
      archetype_stores_(0),
      num_worlds_(num_worlds),
      world_indices_(0),
      register_lock_()
{
    registerComponent<Entity>();
    registerComponent<WorldIndex>();
    registerComponent<WorldID>();
}
#else
StateManager::StateManager()
    : query_data_(0),
      component_infos_(0),
      archetype_components_(0),
      archetype_stores_(0)
{
    registerComponent<Entity>();
}
#endif

struct StateManager::ArchetypeStore::Init {
    uint32_t componentOffset;
    uint32_t numComponents;
    uint32_t id;
    HeapArray<TypeInfo> types;
    HeapArray<IntegerMapPair> lookupInputs;
};

StateManager::ArchetypeStore::ArchetypeStore(Init &&init)
    : componentOffset(init.componentOffset),
      numComponents(init.numComponents),
      tbl(init.types.data(), init.types.size(), init.id),
      columnLookup(init.lookupInputs.data(), init.lookupInputs.size())
{
}

uint32_t StateManager::makeQuery(const ComponentID *components,
                                 uint32_t num_components,
                                 uint32_t *offset)
{
    DynArray<uint32_t, TmpAlloc> query_indices(0);

    uint32_t matching_archetypes = 0;
    for (int archetype_idx = 0, num_archetypes = archetype_stores_.size();
         archetype_idx < num_archetypes; archetype_idx++) {
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

        matching_archetypes += 1;

        query_indices.push_back(uint32_t(archetype_idx));

        for (int component_idx = 0; component_idx < (int)num_components;
             component_idx++) {
            ComponentID component = components[component_idx];
            if (component.id == componentID<Entity>().id) {
                query_indices.push_back(0);
            } else {
                query_indices.push_back(archetype.columnLookup[component.id]);
            }
        }
    }

    *offset = query_data_.size();
    query_data_.resize(*offset + query_indices.size(), [](auto) {});
    memcpy(&query_data_[*offset], query_indices.data(),
           sizeof(uint32_t) * query_indices.size());

    return matching_archetypes;
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

    uint32_t num_total_components = num_user_components;
#ifdef MADRONA_MW_MODE
    num_total_components += 2;
#endif

    HeapArray<TypeInfo, TmpAlloc> type_infos(num_total_components);
    HeapArray<IntegerMapPair, TmpAlloc> lookup_input(num_user_components);

    TypeInfo *type_ptr = type_infos.data();
#ifdef MADRONA_MW_MODE
    type_ptr[0] = *component_infos_[componentID<WorldIndex>().id];
    type_ptr[1] = *component_infos_[componentID<WorldID>().id];
    type_ptr += 2;

    if (world_indices_.size() <= id * num_worlds_) {
        world_indices_.resize((id + 1) * num_worlds_, [](auto) {});
    }

    for (int i = 0; i < (int)num_worlds_; i++) {
        int world_offset = id * num_worlds_;
        world_indices_[world_offset + i] = WorldIndex {
            ~0u,
            ~0u,
        };
    }
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
        std::move(type_infos),
        std::move(lookup_input),
    });
}

uint32_t StateManager::next_component_id_ = 0;
uint32_t StateManager::next_archetype_id_ = 0;

}
