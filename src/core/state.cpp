#include <madrona/state.hpp>
#include <madrona/utils.hpp>
#include <madrona/dyn_array.hpp>

#include <cassert>
#include <functional>
#include <mutex>
#include <string_view>

namespace madrona {

StateManager::StateManager()
    : component_infos_(0),
      archetype_components_(0),
      archetype_infos_(0),
      query_data_(0)
{}

void StateManager::saveComponentInfo(uint32_t id,
                                     uint32_t alignment,
                                     uint32_t num_bytes)
{
    // IDs are globally assigned, technically there is an edge case where
    // there are gaps in the IDs assigned to a specific StateManager
    // for component_infos_ just use default initialization of the
    // unregistered components
    if (id >= component_infos_.size()) {
        component_infos_.resize(id + 1, [](auto) {});
    }

    component_infos_[id] = TypeInfo {
        .alignment = alignment,
        .numBytes = num_bytes,
    };
}

void StateManager::saveArchetypeInfo(uint32_t id, Span<ComponentID> components)
{
    uint32_t offset = archetype_components_.size();

    // FIXME Candidates for tmp allocator
    HeapArray<TypeInfo> type_infos(components.size());
    HeapArray<IntegerMapPair> lookup_input(components.size());

    for (int i = 0; i < (int)components.size(); i++) {
        ComponentID component_id = components[i];

        archetype_components_.push_back(component_id);
        type_infos[i] = component_infos_[component_id.id];

        lookup_input[i] = IntegerMapPair {
            .key = component_id.id,
            .value = (uint32_t)i,
        };
    }

    Table archetype_tbl(type_infos.data(), type_infos.size(), id);

    ColumnMap column_lookup(lookup_input.data(), lookup_input.size());

    // IDs are globally assigned, technically there is an edge case where
    // there are gaps in the IDs assigned to a specific StateManager
    if (id >= archetype_infos_.size()) {
        archetype_infos_.resize(id + 1, [](auto ptr) {
            Optional<ArchetypeInfo>::noneAt(ptr);
        });
    }

    archetype_infos_[id].emplace(ArchetypeInfo {
        .componentOffset = offset,
        .numComponents = components.size(),
        .tbl = std::move(archetype_tbl),
        .columnLookup = std::move(column_lookup),
    });
}

uint32_t StateManager::makeQuery(const ComponentID *components,
                                 uint32_t num_components,
                                 const uint32_t **indices_out)
{
    DynArray<uint32_t, TmpAlloc> query_indices(0);

    uint32_t matching_archetypes = 0;
    for (int archetype_idx = 0; archetype_idx < (int)archetype_infos_.size();
         archetype_idx++) {
        auto &archetype = *archetype_infos_[archetype_idx];

        bool has_components = true;
        for (int component_idx = 0; component_idx < (int)num_components; 
             component_idx++) {
            ComponentID component = components[component_idx];
            if (!archetype.columnLookup.exists(component.id)) {
                has_components = false;
                break;
            }
        }

        if (!has_components) {
            continue;
        }

        query_indices.push_back(uint32_t(archetype_idx));
        matching_archetypes += 1;

        for (int component_idx = 0; component_idx < (int)num_components;
             component_idx++) {
            ComponentID component = components[component_idx];
            query_indices.push_back(archetype.columnLookup[component.id]);
        }
    }

    *indices_out = nullptr;
    return matching_archetypes;
}

uint32_t StateManager::next_component_id_ = 0;
uint32_t StateManager::next_archetype_id_ = 0;

}
