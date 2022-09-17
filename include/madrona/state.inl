#pragma once

#include <madrona/utils.hpp>

#include <array>
#include <mutex>

namespace madrona {

template <typename ComponentT>
ComponentID StateManager::registerComponent()
{
#ifdef MADRONA_MW_MODE
    std::lock_guard lock(register_lock_);

    uint32_t check_id = TypeTracker::typeID<ComponentT>();

    if (check_id < component_infos_.size() &&
        component_infos_[check_id].has_value()) {
        return ComponentID {
            check_id,
        };
    }
#endif

    TypeTracker::registerType<ComponentT>(&next_component_id_);

    uint32_t id = TypeTracker::typeID<ComponentT>();

    registerComponent(id, std::alignment_of_v<ComponentT>,
                      sizeof(ComponentT));

    return ComponentID {
        id,
    };
}

template <typename ArchetypeT>
ArchetypeID StateManager::registerArchetype()
{
#ifdef MADRONA_MW_MODE
    std::lock_guard lock(register_lock_);

    uint32_t check_id = TypeTracker::typeID<ArchetypeT>();

    if (check_id < component_infos_.size() &&
        archetype_stores_[check_id].has_value()) {
        return ArchetypeID {
            check_id,
        };
    }
#endif

    TypeTracker::registerType<ArchetypeT>(&next_archetype_id_);

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
    
    uint32_t id = TypeTracker::typeID<ArchetypeT>();

    registerArchetype(id,
        Span(archetype_components.data(), archetype_components.size()));

    return ArchetypeID {
        id,
    };
}

template <typename ComponentT>
ComponentID StateManager::componentID() const
{
    return ComponentID {
        TypeTracker::typeID<ComponentT>(),
    };
}

template <typename ArchetypeT>
ArchetypeID StateManager::archetypeID() const
{
    return ArchetypeID {
        TypeTracker::typeID<ArchetypeT>(),
    };
}

template <typename ComponentT>
ResultRef<ComponentT> StateManager::get(Entity entity)
{
    uint32_t archetype_idx = entity.archetype;

    ArchetypeStore &archetype = *archetype_stores_[archetype_idx];

    auto col_idx = archetype.columnLookup.lookup(componentID<ComponentT>().id);

    if (!col_idx.has_value()) {
        return ResultRef<ComponentT>(nullptr);
    }

    auto loc = archetype.tbl.getLoc(entity);
    if (!loc.valid()) {
        return ResultRef<ComponentT>(nullptr);
    }

    return ResultRef<ComponentT>(
        (ComponentT *)archetype.tbl.getValue(*col_idx, loc));
}

template <typename ArchetypeT>
ArchetypeRef<ArchetypeT> StateManager::archetype()
{
    auto archetype_id = archetypeID<ArchetypeT>();
    return ArchetypeRef<ArchetypeT>(&archetype_stores_[archetype_id.id]->tbl);
}

template <typename... ComponentTs>
Query<ComponentTs...> StateManager::query()
{
    std::array component_ids {
        componentID<ComponentTs>()
        ...
    };

    QueryRef *ref = &Query<ComponentTs...>::ref_;

    if (ref->numReferences.load(std::memory_order_acquire) == 0) {
        makeQuery(component_ids.data(), component_ids.size(), ref);
    }

    return Query<ComponentTs...>(true);
}

template <typename... ComponentTs, typename Fn>
void StateManager::iterateArchetypes(const Query<ComponentTs...> &query,
                                     Fn &&fn)
{
    using IndicesWrapper =
        std::make_integer_sequence<uint32_t, sizeof...(ComponentTs)>;

    iterateArchetypesImpl(query, std::forward<Fn>(fn), IndicesWrapper());
}

template <typename... ComponentTs, typename Fn, uint32_t... Indices>
void StateManager::iterateArchetypesImpl(const Query<ComponentTs...> &query,
    Fn &&fn, std::integer_sequence<uint32_t, Indices...>)
{
    assert(query.initialized_);

    uint32_t *cur_query_ptr = &query_state_.queryData[query.ref_.offset];
    const int num_archetypes = query.ref_.numMatchingArchetypes;

    for (int query_archetype_idx = 0; query_archetype_idx < num_archetypes;
         query_archetype_idx++) {
        uint32_t archetype_idx = *(cur_query_ptr++);

        auto &archetype = *archetype_stores_[archetype_idx];

        int num_rows = archetype.tbl.numRows();

        fn(num_rows, (ComponentTs *)archetype.tbl.data(
            cur_query_ptr[Indices]) ...);

        cur_query_ptr += sizeof...(ComponentTs);
    }
}

template <typename... ComponentTs, typename Fn>
void StateManager::iterateEntities(const Query<ComponentTs...> &query, Fn &&fn)
{
    iterateArchetypes(query, [&fn](int num_rows, auto ...ptrs) {
        for (int i = 0; i < num_rows; i++) {
            fn(ptrs[i] ...);
        }
    });
}

template <typename ArchetypeT, typename... Args>
Entity StateManager::makeEntity(Args && ...args)
{
    ArchetypeID archetype_id = archetypeID<ArchetypeT>();

    ArchetypeStore &archetype = *archetype_stores_[archetype_id.id];

    constexpr uint32_t num_args = sizeof...(Args);

    assert((num_args == 0 || num_args == archetype.numComponents) &&
           "Trying to construct entity with wrong number of arguments");

    Entity new_row = archetype.tbl.addRow();

    auto tbl_loc = archetype.tbl.getLoc(new_row);

    int component_idx = 0;

    auto constructNextComponent = [this, &component_idx, &archetype, &tbl_loc](
            auto &&arg) {
        using ArgT = decltype(arg);
        using ComponentT = std::remove_reference_t<ArgT>;

        assert(componentID<ComponentT>().id ==
               archetype_components_[archetype.componentOffset +
                   component_idx].id);

        new (archetype.tbl.getValue(
                component_idx + user_component_offset_, tbl_loc))
            ComponentT(std::forward<ArgT>(arg));

        component_idx++;
    };

    ( constructNextComponent(std::forward<Args>(args)), ... );

    return new_row;
}

void StateManager::destroyEntity(Entity e)
{
    ArchetypeStore &archetype = *archetype_stores_[e.archetype];
    archetype.tbl.removeRow(e);
}

#ifdef MADRONA_MW_MODE
uint32_t StateManager::numWorlds() const
{
    return num_worlds_;
}
#endif

}
