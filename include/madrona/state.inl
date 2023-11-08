/*
 * Copyright 2021-2022 Brennan Shacklett and contributors
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 */
#pragma once

#include <madrona/utils.hpp>

#include <array>
#include <mutex>

namespace madrona {

template <typename T>
T & EntityStore::LockedMapStore<T>::operator[](int32_t idx)
{
    return ((T *)store.data())[idx];
}

template <typename T>
const T & EntityStore::LockedMapStore<T>::operator[](int32_t idx) const
{
    return ((const T *)store.data())[idx];
}

Loc EntityStore::getLoc(Entity e) const
{
    return map_.lookup(e);
}

Loc EntityStore::getLocUnsafe(int32_t e_id) const
{
    return map_.getRef(e_id);
}

void EntityStore::setLoc(Entity e, Loc loc)
{
    map_.getRef(e) = loc;
}

void EntityStore::setRow(Entity e, uint32_t row)
{
    Loc &loc = map_.getRef(e);
    loc.row = row;
}

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

template <typename ArchetypeT, typename... MetadataComponentTs>
ArchetypeID StateManager::registerArchetype(
        ComponentMetadataSelector<MetadataComponentTs...> component_metadata,
        ArchetypeFlags archetype_flags,
        CountT max_num_entities_per_world)
{
#ifdef MADRONA_MW_MODE
    std::lock_guard lock(register_lock_);

    uint32_t check_id = TypeTracker::typeID<ArchetypeT>();

    if (check_id < archetype_stores_.size() &&
        archetype_stores_[check_id].has_value()) {
        return ArchetypeID {
            check_id,
        };
    }
#endif

    TypeTracker::registerType<ArchetypeT>(&next_archetype_id_);

    using Base = typename ArchetypeT::Base;

    using Delegator = utils::PackDelegator<Base>;

    auto [archetype_components, archetype_component_flags] = Delegator::call(
        [&component_metadata]<typename... Args>()
    {
        static_assert(std::is_same_v<Base, Archetype<Args...>>);
        uint32_t column_idx = user_component_offset_;

        auto registerColumnIndex =
                [&column_idx]<typename ComponentT>() {
            using LookupT = typename ArchetypeRef<ArchetypeT>::
                template ComponentLookup<ComponentT>;

            TypeTracker::registerType<LookupT>(&column_idx);
        };

        ( registerColumnIndex.template operator()<Args>(), ... );

        std::array components {
            ComponentID { TypeTracker::typeID<Args>() }
            ...
        };

        std::array<ComponentFlags, components.size()> component_flags;
        component_flags.fill(ComponentFlags::None);

        int32_t cur_metadata_idx = 0;
        auto setFlags = [&]<typename ComponentT>() {
            ComponentFlags cur_flags =
                component_metadata.flags[cur_metadata_idx++];

            using LookupT = typename ArchetypeRef<ArchetypeT>::
                template ComponentLookup<ComponentT>;

            uint32_t flag_out_idx =
                TypeTracker::typeID<LookupT>() - user_component_offset_;

            component_flags[flag_out_idx] = cur_flags;
        };

        ( setFlags.template operator()<MetadataComponentTs>(), ... );

        return std::make_pair(components, component_flags);
    });
    
    uint32_t id = TypeTracker::typeID<ArchetypeT>();

    registerArchetype(id,
                      archetype_flags,
                      max_num_entities_per_world,
                      (CountT)archetype_components.size(),
                      archetype_components.data(),
                      archetype_component_flags.data());

    return ArchetypeID {
        id,
    };
}

template <typename SingletonT>
void StateManager::registerSingleton()
{
    using ArchetypeT = SingletonArchetype<SingletonT>;

    registerComponent<SingletonT>();
    registerArchetype<ArchetypeT>(
        ComponentMetadataSelector<> {}, ArchetypeFlags::None, 1);

#ifdef MADRONA_MW_MODE
    for (CountT i = 0; i < (CountT)num_worlds_; i++) {
        makeEntityNow<ArchetypeT>(uint32_t(i), init_state_cache_);
    }
#else
    makeEntityNow<ArchetypeT>(init_state_cache_);
#endif
}

template <typename ArchetypeT, typename ComponentT>
ComponentT * StateManager::exportColumn()
{
    return (ComponentT *)exportColumn(
        archetypeID<ArchetypeT>().id,
        componentID<ComponentT>().id);
}

template <typename SingletonT>
SingletonT * StateManager::exportSingleton()
{
    using ArchetypeT = SingletonArchetype<SingletonT>;

    return exportColumn<ArchetypeT, SingletonT>();
}

template <typename SingletonT>
SingletonT & StateManager::getSingleton(MADRONA_MW_COND(uint32_t world_id))
{
    using ArchetypeT = SingletonArchetype<SingletonT>;
    uint32_t archetype_id = TypeTracker::typeID<ArchetypeT>();
    auto &archetype = *archetype_stores_[archetype_id];

    return *archetype.tblStorage.column<SingletonT>(
        MADRONA_MW_COND(world_id,)
        user_component_offset_);
}

template <typename ComponentT>
ComponentID StateManager::componentID() const
{
    static_assert(!std::is_reference_v<ComponentT> &&
                  !std::is_pointer_v<ComponentT> &&
                  !std::is_const_v<ComponentT>);
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

Loc StateManager::getLoc(Entity e) const
{
    return entity_store_.getLoc(e);
}

template <typename ComponentT>
inline ResultRef<ComponentT> StateManager::get(
    MADRONA_MW_COND(uint32_t world_id,) Loc loc)
{
    ArchetypeStore &archetype = *archetype_stores_[loc.archetype];
    auto col_idx = archetype.columnLookup.lookup(componentID<ComponentT>().id);

    if (!col_idx.has_value()) {
        return ResultRef<ComponentT>(nullptr);
    }

    auto col = archetype.tblStorage.column<ComponentT>(
        MADRONA_MW_COND(world_id,) *col_idx);

    return ResultRef<ComponentT>(col + loc.row);
}

template <typename ComponentT>
ResultRef<ComponentT> StateManager::get(
    MADRONA_MW_COND(uint32_t world_id,) Entity entity)
{
    Loc loc = entity_store_.getLoc(entity);
    if (!loc.valid()) {
        return ResultRef<ComponentT>(nullptr);
    }

    return get<ComponentT>(MADRONA_MW_COND(world_id,) loc);
}

template <typename ComponentT>
ComponentT & StateManager::getUnsafe(
    MADRONA_MW_COND(uint32_t world_id,) int32_t entity_id)
{
    Loc loc = entity_store_.getLocUnsafe(entity_id);
    return getUnsafe<ComponentT>(MADRONA_MW_COND(world_id,) loc);
}

template <typename ComponentT>
ComponentT & StateManager::getUnsafe(
    MADRONA_MW_COND(uint32_t world_id,) Loc loc)
{
    ArchetypeStore &archetype = *archetype_stores_[loc.archetype];
    auto col_idx =
        *archetype.columnLookup.lookup(componentID<ComponentT>().id);

    auto col = archetype.tblStorage.column<ComponentT>(
        MADRONA_MW_COND(world_id,) col_idx);

    return col[loc.row];
}

template <typename ComponentT>
inline ComponentT & StateManager::getDirect(MADRONA_MW_COND(uint32_t world_id,)
                                            CountT col_idx,
                                            Loc loc)
{
    ArchetypeStore &archetype = *archetype_stores_[loc.archetype];

    auto col = archetype.tblStorage.column<ComponentT>(
        MADRONA_MW_COND(world_id,) col_idx);

    return col[loc.row];
}

template <typename ArchetypeT>
ArchetypeRef<ArchetypeT> StateManager::archetype(
    MADRONA_MW_COND(uint32_t world_id))
{
#ifdef MADRONA_MW_MODE
    (void)world_id;
#endif
    assert(false);
#if 0
    auto archetype_id = archetypeID<ArchetypeT>();

    ArchetypeStore &archetype = *archetype_stores_[archetype_id.id];

    Table &tbl = 
#ifdef MADRONA_MW_MODE
        archetype.tbls[world_id];
#else
        archetype.tbl;
#endif

    return ArchetypeRef<ArchetypeT>(&tbl);
#endif
}

template <typename... ComponentTs>
Query<ComponentTs...> StateManager::query()
{
    std::array component_ids {
        componentID<std::remove_const_t<ComponentTs>>()
        ...
    };

    QueryRef *ref = &Query<ComponentTs...>::ref_;

    // If necessary, create the query templated on the passed in ComponentTs.
    if (ref->numReferences.load_acquire() == 0) {
        makeQuery(component_ids.data(), component_ids.size(), ref);
    }

    return Query<ComponentTs...>(true);
}

template <typename... ComponentTs, typename Fn>
void StateManager::iterateArchetypes(MADRONA_MW_COND(uint32_t world_id,)
                                     const Query<ComponentTs...> &query,
                                     Fn &&fn)
{
    using IndicesWrapper =
        std::make_integer_sequence<uint32_t, sizeof...(ComponentTs)>;

    iterateArchetypesImpl(MADRONA_MW_COND(world_id,)
                          query, std::forward<Fn>(fn), IndicesWrapper());
}

template <typename... ComponentTs, typename Fn, uint32_t... Indices>
void StateManager::iterateArchetypesImpl(MADRONA_MW_COND(uint32_t world_id,)
    const Query<ComponentTs...> &query, Fn &&fn,
    std::integer_sequence<uint32_t, Indices...>)
{
    assert(query.initialized_);

    uint32_t *cur_query_ptr = &query_state_.queryData[query.ref_.offset];
    const int num_archetypes = query.ref_.numMatchingArchetypes;

    for (int query_archetype_idx = 0; query_archetype_idx < num_archetypes;
         query_archetype_idx++) {
        uint32_t archetype_idx = *(cur_query_ptr++);

        ArchetypeStore &archetype = *archetype_stores_[archetype_idx];

        CountT num_rows =
            archetype.tblStorage.numRows(MADRONA_MW_COND(world_id));

        // FIXME: column API sucks here, hopefully the compiler can
        // do common subexpression elimination on the world_id index...
        fn(num_rows, archetype.tblStorage.column<ComponentTs>(
            MADRONA_MW_COND(world_id,) cur_query_ptr[Indices]) ...);

        cur_query_ptr += sizeof...(ComponentTs);
    }
}

template <typename... ComponentTs, typename Fn>
void StateManager::iterateQuery(MADRONA_MW_COND(uint32_t world_id,)
                                   const Query<ComponentTs...> &query, Fn &&fn)
{
    iterateArchetypes(MADRONA_MW_COND(world_id,) query, 
            [&fn](int num_rows, auto ...ptrs) {
        for (int i = 0; i < num_rows; i++) {
            fn(ptrs[i] ...);
        }
    });
}

template <typename ArchetypeT, typename... Args>
Entity StateManager::makeEntityNow(MADRONA_MW_COND(uint32_t world_id,)
                                   StateCache &cache, Args && ...args)
{
    ArchetypeID archetype_id = archetypeID<ArchetypeT>();

    ArchetypeStore &archetype = *archetype_stores_[archetype_id.id];

    constexpr uint32_t num_args = sizeof...(Args);

    assert((num_args == 0 || num_args == archetype.numComponents) &&
           "Trying to construct entity with wrong number of arguments");

    Entity e = entity_store_.newEntity(cache.entity_cache_);

    CountT new_row = archetype.tblStorage.addRow(MADRONA_MW_COND(world_id));

    archetype.tblStorage.column<Entity>(
        MADRONA_MW_COND(world_id,) 0)[new_row] = e;

#ifdef MADRONA_MW_MODE
    archetype.tblStorage.column<WorldID>(world_id, 1)[new_row] =
        WorldID { (int32_t)world_id };
#endif

    int component_idx = 0;

    auto constructNextComponent = [&](auto &&arg) {
        using ArgT = decltype(arg);
        using ComponentT = std::remove_reference_t<ArgT>;

        assert(componentID<ComponentT>().id ==
               archetype_components_[archetype.componentOffset +
                   component_idx].id);

        new (archetype.tblStorage.column<ComponentT>(
                MADRONA_MW_COND(world_id,)
                component_idx + user_component_offset_) + new_row)
            ComponentT(std::forward<ArgT>(arg));

        component_idx++;
    };

    ( constructNextComponent(std::forward<Args>(args)), ... );
    
    entity_store_.setLoc(e, Loc {
        .archetype = archetype_id.id,
        .row = int32_t(new_row),
    });

    return e;
}

template <typename ArchetypeT>
Loc StateManager::makeTemporary(MADRONA_MW_COND(uint32_t world_id))
{
    ArchetypeID archetype_id = archetypeID<ArchetypeT>();
    ArchetypeStore &archetype = *archetype_stores_[archetype_id.id];

    CountT new_row = archetype.tblStorage.addRow(
        MADRONA_MW_COND(world_id));

    return Loc {
        archetype_id.id,
        int32_t(new_row),
    };
}

template <typename ArchetypeT>
void StateManager::clear(MADRONA_MW_COND(uint32_t world_id,) StateCache &cache,
                         bool is_temporary)
{
    clear(MADRONA_MW_COND(world_id,) cache, archetypeID<ArchetypeT>().id,
          is_temporary);
}

#ifdef MADRONA_MW_MODE
uint32_t StateManager::numWorlds() const
{
    return num_worlds_;
}
#endif

template <typename ColumnT>
inline ColumnT * StateManager::TableStorage::column(
    MADRONA_MW_COND(uint32_t world_id,)
    CountT col_idx)
{
#ifdef MADRONA_MW_MODE
    if (maxNumPerWorld == 0) {
        return (ColumnT *)tbls[world_id].data(col_idx);
    } else {
        return ((ColumnT *)fixed.tbl.data(col_idx)) +
            CountT(world_id) * maxNumPerWorld;
    }
#else
    return (ColumnT *)tbl.data(col_idx);
#endif
}

inline CountT StateManager::TableStorage::numRows(
    MADRONA_MW_COND(uint32_t world_id))
{
#ifdef MADRONA_MW_MODE
    if (maxNumPerWorld == 0) {
        return tbls[world_id].numRows();
    } else {
        return fixed.activeRows[world_id];
    }
#else
    return tbl.numRows();
#endif
}

void StateManager::TableStorage::clear(
    MADRONA_MW_COND(uint32_t world_id))
{
#ifdef MADRONA_MW_MODE
    if (maxNumPerWorld == 0) {
        tbls[world_id].clear();
    } else {
        fixed.activeRows[world_id] = 0;
    }
#else
    tbl.clear();
#endif
}

CountT StateManager::TableStorage::addRow(
    MADRONA_MW_COND(uint32_t world_id))
{
#ifdef MADRONA_MW_MODE
    if (maxNumPerWorld == 0) {
        return tbls[world_id].addRow();
    } else {
        return fixed.activeRows[world_id]++;
    }
#else
    return tbl.addRow();
#endif
}

bool StateManager::TableStorage::removeRow(MADRONA_MW_COND(uint32_t world_id,)
                                           CountT row)
{
#ifdef MADRONA_MW_MODE
    if (maxNumPerWorld == 0) {
        return tbls[world_id].removeRow(row);
    } else {
        CountT removed_row = --fixed.activeRows[world_id];
        if (removed_row == row) {
            return false;
        }

        fixed.tbl.copyRow(row, removed_row);

        return true;
    }
#else
    return tbl.removeRow(row);
#endif
}

}
