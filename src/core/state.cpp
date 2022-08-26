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
      archetype_infos_(0)
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
        component_infos_.resize(id + 1);
    }

    component_infos_[id] = TypeInfo {
        .alignment = alignment,
        .numBytes = num_bytes,
    };
}

void StateManager::saveArchetypeInfo(uint32_t id, Span<ComponentID> components)
{
    uint32_t offset = archetype_components_.size();
    HeapArray<TypeInfo> type_infos(components.size());
    for (int i = 0; i < (int)components.size(); i++) {
        ComponentID component_id = components[i];

        archetype_components_.push_back(component_id);
        type_infos[i] = component_infos_[component_id.id];
    }

    Table archetype_tbl(type_infos.data(), type_infos.size(), id);

    // IDs are globally assigned, technically there is an edge case where
    // there are gaps in the IDs assigned to a specific StateManager
    if (id >= archetype_infos_.size()) {
        archetype_infos_.resize(id + 1);
    }

    archetype_infos_[id].emplace(ArchetypeInfo {
        .componentOffset = offset,
        .numComponents = components.size(),
        .tbl = std::move(archetype_tbl),
    });
}

uint32_t StateManager::next_component_id_ = 0;
uint32_t StateManager::next_archetype_id_ = 0;

}
