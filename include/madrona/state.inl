#pragma once

#include <madrona/utils.hpp>

namespace madrona {

template <typename T>
Entity IDManager::registerType()
{
    uint32_t id = IDManager::assignID(
#ifdef _MSC_VER
        __FUNCDNAME__
#else
        __PRETTY_FUNCTION__
#endif
        );

    return Entity {
        id,
    };
}

// Defining this static member in here rather than ecs.inl is a little
// weird, but the GPU implementation needs to be different due to
// the lack of static initialization support in nvcc, and ecs.inl is
// currently shared.
template <typename ComponentT>
Entity Component<ComponentT>::_id = IDManager::registerType<ComponentT>();

template <typename ComponentT>
Entity StateManager::registerComponent()
{
    // FIXME: register size & alignment
    return ComponentT::_id;
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

    Delegator::call([]<typename... Args>() {
        static_assert(std::is_same_v<ArchetypeT, Archetype<Args...>>);
        // do actual stuff
    });
}

}
