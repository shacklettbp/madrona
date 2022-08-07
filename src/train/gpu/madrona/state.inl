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
    auto archetype_components = Delegator::call([]<typename... Args>() {
        static_assert(std::is_same_v<ArchetypeT, Archetype<Args...>>);

        std::array archetype_components {
            TypeID<Args>::id
            ...
        };

        // do actual stuff

        return archetype_components;
    });

    saveArchetypeInfo(archetype_id, archetype_components);

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
