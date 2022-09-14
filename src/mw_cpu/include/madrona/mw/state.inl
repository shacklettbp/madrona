#pragma once

namespace madrona {
namespace mw {

template <typename ComponentT>
ComponentID StateManager::registerComponent()
{
    std::lock_guard lock(register_lock_);

    ComponentID id = componentID<ComponentT>();

    if (id.id < component_infos_.size() &&
        component_infos_[id.id].has_value()) {
        return id;
    }

    return StateManagerBase::registerComponent<ComponentT>();
}

template <typename ArchetypeT>
ArchetypeID StateManager::registerArchetype()
{
    std::lock_guard lock(register_lock_);

    ArchetypeID id = archetypeID<ArchetypeT>();

    if (id.id < archetype_stores_.size() &&
        archetype_stores_[id.id].has_value()) {
        return id;
    }

    return StateManagerBase::registerArchetype<ArchetypeT>();
}

}
}
