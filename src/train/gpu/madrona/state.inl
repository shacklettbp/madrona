#pragma once

#include <madrona/utils.hpp>

namespace madrona {

template <typename ComponentT>
Entity Component<ComponentT>::_id = Entity::none();

template <typename ComponentT>
Entity StateManager::registerComponent()
{
    // Get lock
    while (register_lock_.exchange(1, std::memory_order_acq_rel)) {}
    std::atomic_thread_fence(std::memory_order_acquire);

    Entity component_id = ComponentT::_id;

    if (component_id == Entity::none()) {
        uint32_t new_component_id = num_components_++;
        component_id.id = new_component_id;

        ComponentT::_id = component_id;
    }

    register_lock_.store(0, std::memory_order_relaxed);
    std::atomic_thread_fence(std::memory_order_release);

    return component_id;
}

template <typename ComponentT>
Entity StateManager::componentID()
{
    return ComponentT::_id;
}

template <typename ArchetypeT>
void StateManager::registerArchetype()
{
    using Delegator = utils::PackDelegator<ArchetypeT>;
    uint32_t num_components = Delegator::call([]<typename... Args>() {
        static_assert(std::is_same_v<ArchetypeT, Archetype<Args...>>);

        uint32_t num_components = sizeof...(Args);

        // do actual stuff

        return num_components;
    });

    printf("%u\n", num_components);

    // Get lock
    while (register_lock_.exchange(1, std::memory_order_acq_rel)) {}
    std::atomic_thread_fence(std::memory_order_acquire);



    register_lock_.store(0, std::memory_order_relaxed);
    std::atomic_thread_fence(std::memory_order_release);
}

}
