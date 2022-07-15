#pragma once

namespace madrona {

namespace ComponentID {
template <typename T> Entity id;
}

template <typename T>
Entity StateManager::registerComponent()
{
    uint32_t component_id = num_components_++;
    Entity entity { component_id };

    ComponentID::id<T> = entity;

    return entity;
}

template <typename T>
Entity StateManager::componentID()
{
    return ComponentID::id<T>;
}

}
