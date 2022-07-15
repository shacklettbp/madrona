#pragma once

#include <madrona/utils.hpp>

namespace madrona {

namespace ComponentID {
template <typename T> Entity id;
}

namespace ArchetypeID {
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

template <template<typename ...> class A, typename... Components>
struct ComponentExtractor {
    static constexpr uint32_t numComponents = sizeof...(Components);
};

template <typename T>
void StateManager::registerArchetype()
{
    using Delegator = utils::PackDelegator<T>;

    uint32_t num_components = Delegator::call([]<typename... Args>() {
        static_assert(std::is_same_v<T, Archetype<Args...>>);

        uint32_t num_components = sizeof...(Args);

        // do actual stuff

        return num_components;
    });
}

}
