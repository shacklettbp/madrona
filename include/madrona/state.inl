#pragma once

#include <madrona/utils.hpp>

#include <array>

namespace madrona {

template <typename T>
struct StateManager::TypeID {
    static uint32_t id;
};

template <typename T>
uint32_t StateManager::TypeID<T>::id =
    StateManager::trackType<T>(&StateManager::TypeID<T>::id);

template <typename ComponentT>
ComponentID StateManager::registerComponent()
{
    StateManager::registerType(&TypeID<ComponentT>::id, true);

    uint32_t id = TypeID<ComponentT>::id;

    saveComponentInfo(id, std::alignment_of_v<ComponentT>,
                      sizeof(ComponentT));

    return ComponentID {
        id,
    };
}

template <typename ArchetypeT>
ArchetypeID StateManager::registerArchetype()
{
    StateManager::registerType(&TypeID<ArchetypeT>::id, false);

    using Delegator = utils::PackDelegator<ArchetypeT>;

    auto archetype_components = Delegator::call([]<typename... Args>() {
        static_assert(std::is_same_v<ArchetypeT, Archetype<Args...>>);
        std::array archetype_components {
            ComponentID { TypeID<Args>::id }
            ...
        };

        return archetype_components;
    });
    
    uint32_t id = TypeID<ArchetypeT>::id;

    saveArchetypeInfo(id,
        Span(archetype_components.data(), archetype_components.size()));

    return ArchetypeID {
        id,
    };
}

template <typename ComponentT>
ComponentID StateManager::componentID() const
{
    return ComponentID {
        TypeID<ComponentT>::id,
    };
}

template <typename ArchetypeT>
ArchetypeID StateManager::archetypeID() const
{
    return ArchetypeID {
        TypeID<ArchetypeT>::id,
    };
}

template <typename ComponentT>
ComponentT & StateManager::getComponent(Entity entity)
{
}

template <typename ...ComponentTs>
Query<ComponentTs...> StateManager::query()
{
    auto getContainingArchetypes = [this](Entity component_id) {
        
    };

    ( getContainingArchetypes(componentID<ComponentTs>()), ... );
}

template <typename ArchetypeT, typename ...Args>
Entity StateManager::makeEntity(Args && ...args)
{
    ArchetypeID archetype_id = archetypeID<ArchetypeT>();

    ArchetypeInfo &archetype = *archetype_infos_[archetype_id.id];

    assert(sizeof...(Args) == archetype.numComponents &&
           "Trying to construct entity with wrong number of arguments");


    Entity new_row = archetype.tbl.makeRow();

    auto tbl_index = archetype.tbl.getIndex(new_row);

    int column_idx = 0;

    auto constructNextComponent = [this, &column_idx, &archetype, &tbl_index](
            auto &&arg) {
        using ComponentT = std::remove_reference_t<decltype(arg)>;

        assert(componentID<ComponentT>().id ==
               archetype_components_[archetype.componentOffset +
                   column_idx].id);

        new (archetype.tbl.getValue(column_idx, tbl_index)) 
            ComponentT(std::forward<ComponentT>(arg));

        column_idx++;
    };

    ( constructNextComponent(std::forward<Args>(args)), ... );

    return new_row;
}

void StateManager::destroyEntity(Entity e)
{
    auto &archetype = *archetype_infos_[e.archetype];
    archetype.tbl.destroyRow(e);
}

template <typename T>
uint32_t StateManager::trackType(uint32_t *ptr)
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
