#pragma once

#include <madrona/utils.hpp>

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

    // FIXME: register size & alignment
    return TypeID<ComponentT>::id;
}

template <typename ArchetypeT>
Entity StateManager::registerArchetype()
{
    StateManager::registerType(&TypeID<ArchetypeT>::id);

    using Delegator = utils::PackDelegator<ArchetypeT>;

    Delegator::call([]<typename... Args>() {
        static_assert(std::is_same_v<ArchetypeT, Archetype<Args...>>);
        // do actual stuff
    });

    return TypeID<ArchetypeT>::id;
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
