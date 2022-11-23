#pragma once

#include <madrona/utils.hpp>

namespace madrona {

template <typename ComponentT>
ComponentID StateManager::registerComponent()
{
    uint32_t id = TypeTracker::registerType<ComponentT>(
        &StateManager::num_components_);

    registerComponent(id, alignof(ComponentT), sizeof(ComponentT));

    return ComponentID {
        id,
    };
}

template <typename ArchetypeT>
ArchetypeID StateManager::registerArchetype()
{
    uint32_t archetype_id = TypeTracker::registerType<ArchetypeT>(
        &StateManager::num_archetypes_);

    using Base = typename ArchetypeT::Base;

    using Delegator = utils::PackDelegator<Base>;

    auto archetype_components = Delegator::call([]<typename... Args>() {
        static_assert(std::is_same_v<Base, Archetype<Args...>>);
        uint32_t column_idx = user_component_offset_;

        auto registerColumnIndex =
                [&column_idx]<typename ComponentT>() {
            using LookupT = typename ArchetypeRef<ArchetypeT>::
                template ComponentLookup<ComponentT>;

            TypeTracker::registerType<LookupT>(&column_idx);
        };

        ( registerColumnIndex.template operator()<Args>(), ... );

        std::array archetype_components {
            ComponentID { TypeTracker::typeID<Args>() }
            ...
        };

        return archetype_components;
    });

    registerArchetype(archetype_id, archetype_components.data(),
                      archetype_components.size());

    return ArchetypeID {
        archetype_id,
    };
}

template <typename... ComponentTs>
Query<ComponentTs...> StateManager::query()
{
    std::array component_ids {
        TypeTracker::typeID<ComponentTs>()
        ...
    };

    QueryRef *ref = &Query<ComponentTs...>::ref_;

    if (ref->numReferences.load(std::memory_order_acquire) == 0) {
        makeQuery(component_ids.data(), component_ids.size(), ref);
    }

    return Query<ComponentTs...>(true);
}

template <typename Fn, int32_t... Indices>
void StateManager::iterateArchetypesRawImpl(QueryRef *query_ref, Fn &&fn,
        std::integer_sequence<int32_t, Indices...>)
{

    uint32_t *query_values = &query_data_[query_ref->offset];
    int32_t num_archetypes = query_ref->numMatchingArchetypes;

    for (int i = 0; i < num_archetypes; i++) {
        uint32_t archetype_idx = *query_values;
        query_values += 1;

        Table &tbl = archetypes_[archetype_idx]->tbl;

        bool early_out =
            fn(tbl.numRows, (WorldID *)(tbl.columns[1]),
               tbl.columns[query_values[Indices]] ...);
        if (early_out) {
            return;
        }

        query_values += sizeof...(Indices);
    }
}

template <int32_t num_components, typename Fn>
void StateManager::iterateArchetypesRaw(QueryRef *query_ref, Fn &&fn)
{
    using IndicesWrapper =
        std::make_integer_sequence<int32_t, num_components>;

    iterateArchetypesRawImpl(query_ref, std::forward<Fn>(fn),
                             IndicesWrapper());
}

uint32_t StateManager::numMatchingEntities(QueryRef *query_ref)
{
    uint32_t *query_values = &query_data_[query_ref->offset];
    int32_t num_archetypes = query_ref->numMatchingArchetypes;
    int32_t num_components = query_ref->numComponents;

    uint32_t total_rows = 0;
    for (int i = 0; i < num_archetypes; i++) {
        uint32_t archetype_idx = *query_values;

        Table &tbl = archetypes_[archetype_idx]->tbl;

        total_rows += tbl.numRows;

        query_values += 1 + num_components;
    }

    return total_rows;
}

template <typename ArchetypeT>
Entity StateManager::makeEntityNow(WorldID world_id)
{
    uint32_t archetype_id = TypeTracker::typeID<ArchetypeT>();
    Table &tbl = archetypes_[archetype_id]->tbl;

    int32_t row = tbl.numRows++;

    // FIXME: proper entity mapping on GPU
    Entity e {
        0,
        row,
    };

    Entity *entity_column = (Entity *)tbl.columns[0];
    WorldID *world_column = (WorldID *)tbl.columns[1];

    entity_column[row] = e;
    world_column[row] = world_id;

    return e;
}

template <typename ArchetypeT>
void StateManager::clear()
{
    uint32_t archetype_id = TypeTracker::typeID<ArchetypeT>();
    Table &tbl = archetypes_[archetype_id]->tbl;

    tbl.numRows = 0;
}

template <typename ArchetypeT, typename ComponentT>
ComponentT * StateManager::getArchetypeColumn()
{
    uint32_t archetype_id = TypeTracker::typeID<ArchetypeT>();
    uint32_t component_id = TypeTracker::typeID<ComponentT>();
    auto &archetype = *archetypes_[archetype_id];
    int32_t col_idx = *archetype.columnLookup.lookup(component_id);

    Table &tbl = archetype.tbl;

    return (ComponentT *)(tbl.columns[col_idx]);
}

}
