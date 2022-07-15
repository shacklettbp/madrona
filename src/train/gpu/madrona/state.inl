#pragma once

namespace madrona {

namespace ComponentID {
template <typename T> Entity id = Entity::none();
}

template <typename T>
Entity StateManager::registerComponent()
{
    // Get lock
    while (register_lock_.exchange(1, std::memory_order_acq_rel)) {}
    std::atomic_thread_fence(std::memory_order_acquire);

    Entity component_id = ComponentID::id<T>;

    if (component_id == Entity::none()) {
        uint32_t new_component_id = num_components_++;
        component_id.id = new_component_id;

        ComponentID::id<T> = component_id;
    }

    register_lock_.store(0, std::memory_order_relaxed);
    std::atomic_thread_fence(std::memory_order_release);

    return component_id;
}

template <typename T>
Entity StateManager::componentID()
{
    return ComponentID::id<T>;
}

}
