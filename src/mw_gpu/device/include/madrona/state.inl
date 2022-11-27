#pragma once

#include <madrona/utils.hpp>

namespace madrona {

template <typename ComponentT>
void ECSRegistry::registerComponent()
{
    state_mgr_->registerComponent<ComponentT>();
}

template <typename ArchetypeT>
void ECSRegistry::registerArchetype()
{
    state_mgr_->registerArchetype<ArchetypeT>();
}

template <typename SingletonT>
void ECSRegistry::registerSingleton()
{
    state_mgr_->registerSingleton<SingletonT>();
}

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

template <typename SingletonT>
void StateManager::registerSingleton()
{
    using ArchetypeT = SingletonArchetype<SingletonT>;

    registerComponent<SingletonT>();
    registerArchetype<ArchetypeT>();

    uint32_t num_worlds = mwGPU::GPUImplConsts::get().numWorlds;

    for (uint32_t i = 0; i < num_worlds; i++) {
        makeEntityNow<ArchetypeT>(WorldID { int32_t(i) });
    }
}

template <typename SingletonT>
SingletonT & StateManager::getSingleton(WorldID world_id)
{
    using ArchetypeT = SingletonArchetype<SingletonT>;
    uint32_t archetype_id = TypeTracker::typeID<ArchetypeT>();

    // Abuse the fact that the singleton only has one component that is going
    // to be in column 2
    
    Table &tbl = archetypes_[archetype_id]->tbl;
    SingletonT *col = (SingletonT *)tbl.columns[2];

    return col[world_id.idx];
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

    int32_t row = tbl.numRows.fetch_add(1, std::memory_order_relaxed);

    Loc loc {
        archetype_id,
        row,
    };

    int32_t entity_slot_idx =
        entity_store_.numEntities.fetch_add(1, std::memory_order_relaxed);
    assert(entity_slot_idx < entity_store_.maxEntities);

    EntityStore::EntitySlot &entity_slot =
        entity_store_.entities[entity_slot_idx];

    entity_slot.loc = loc;
    entity_slot.gen = 0;

    // FIXME: proper entity mapping on GPU
    Entity e {
        entity_slot.gen,
        entity_slot_idx,
    };

    Entity *entity_column = (Entity *)tbl.columns[0];
    WorldID *world_column = (WorldID *)tbl.columns[1];

    entity_column[row] = e;
    world_column[row] = world_id;

    return e;
}

template <typename ArchetypeT>
Loc StateManager::makeTemporary(WorldID world_id)
{
    uint32_t archetype_id = TypeTracker::typeID<ArchetypeT>();
    Table &tbl = archetypes_[archetype_id]->tbl;

    int32_t row = tbl.numRows.fetch_add(1, std::memory_order_relaxed);

    Loc loc {
        archetype_id,
        row,
    };

    WorldID *world_column = (WorldID *)tbl.columns[1];
    world_column[row] = world_id;

    return loc;
}

template <typename ArchetypeT>
void StateManager::clear()
{
    uint32_t archetype_id = TypeTracker::typeID<ArchetypeT>();
    Table &tbl = archetypes_[archetype_id]->tbl;

    tbl.numRows = 0;
}

template <typename ComponentT>
ComponentT & StateManager::getUnsafe(Entity e)
{
    const EntityStore::EntitySlot &slot = entity_store_.entities[e.id];
    return getUnsafe<ComponentT>(slot.loc);
}

template <typename ComponentT>
ComponentT & StateManager::getUnsafe(Loc loc)
{
    auto &archetype = *archetypes_[loc.archetype];
    uint32_t component_id = TypeTracker::typeID<ComponentT>();
    int32_t col_idx = *archetype.columnLookup.lookup(component_id);

    Table &tbl = archetype.tbl;

    return ((ComponentT *)(tbl.columns[col_idx]))[loc.row];
}

}
