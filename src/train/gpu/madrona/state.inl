#pragma once

namespace madrona {

template <typename T>
struct StateManager::TypeID {
    static Entity id;
};

template <typename T>
Entity StateManager::TypeID<T>::id = Entity::none();

template <typename ComponentT>
Entity StateManager::registerComponent()
{
    return StateManager::registerType<ComponentT>();
}

template <typename ArchetypeT>
Entity StateManager::registerArchetype()
{
    Entity archetype_id = StateManager::registerType<ArchetypeT>();

    using Delegator = utils::PackDelegator<ArchetypeT>;
    uint32_t num_components = Delegator::call([]<typename... Args>() {
        static_assert(std::is_same_v<ArchetypeT, Archetype<Args...>>);

        uint32_t num_components = sizeof...(Args);

        // do actual stuff

        return num_components;
    });

    printf("%u\n", num_components);

    return archetype_id;
}

template <typename ComponentT>
Entity StateManager::componentID()
{
    return TypeID<ComponentT>::id;
}

template <typename T>
Entity StateManager::registerType()
{
    register_lock_.lock();

    Entity type_id = TypeID<T>::id;

    if (type_id == Entity::none()) {
        uint32_t new_id = StateManager::num_components_++;

        type_id = Entity {
            .id = new_id,
        };

        TypeID<T>::id = type_id;
    }

    register_lock_.unlock();

    return type_id;
}

}
