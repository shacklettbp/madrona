#pragma once

#include <madrona/utils.hpp>

#include <array>

namespace madrona {

template <typename T>
struct StateManager::TypeID {
    static Entity id;
};

template <typename T>
Entity StateManager::TypeID<T>::id =
    StateManager::trackType<T>(&StateManager::TypeID<T>::id);

template <typename ComponentT>
Entity StateManager::registerComponent()
{
    StateManager::registerType(&TypeID<ComponentT>::id);

    Entity component_entity = TypeID<ComponentT>::id;

    saveComponentInfo(component_entity, std::alignment_of_v<ComponentT>,
                      sizeof(ComponentT));

    return component_entity;
}

template <typename ArchetypeT>
Entity StateManager::registerArchetype()
{
    StateManager::registerType(&TypeID<ArchetypeT>::id);

    using Delegator = utils::PackDelegator<ArchetypeT>;

    auto archetype_components = Delegator::call([]<typename... Args>() {
        static_assert(std::is_same_v<ArchetypeT, Archetype<Args...>>);
        std::array archetype_components {
            TypeID<Args>::id
            ...
        };

        return archetype_components;
    });
    
    Entity archetype_entity = TypeID<ArchetypeT>::id;

    saveArchetypeInfo(archetype_entity,
        Span(archetype_components.data(), archetype_components.size()));

    return archetype_entity;
}

template <typename ComponentT>
Entity StateManager::componentID()
{
    return TypeID<ComponentT>::id;
}

template <typename ArchetypeT>
Entity StateManager::archetypeID()
{
    return TypeID<ArchetypeT>::id;
}

template <typename T>
Entity StateManager::trackType(Entity *ptr)
{
    return StateManager::trackByName(ptr,
#ifdef _MSC_VER
        __FUNCDNAME__
#else
        __PRETTY_FUNCTION__
#endif
        );
}

}
