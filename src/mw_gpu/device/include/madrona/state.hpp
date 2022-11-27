/*
 * Copyright 2021-2022 Brennan Shacklett and contributors
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 */
#pragma once

#include <atomic>
#include <array>

#include <madrona/ecs.hpp>
#include <madrona/hashmap.hpp>
#include <madrona/inline_array.hpp>
#include <madrona/sync.hpp>
#include <madrona/query.hpp>
#include <madrona/optional.hpp>
#include <madrona/type_tracker.hpp>

#include "mw_gpu/const.hpp"

namespace madrona {

class StateManager;

namespace mwGPU {
static inline StateManager *getStateManager()
{
    return (StateManager *)GPUImplConsts::get().stateManagerAddr;
}
}

struct ComponentID {
    uint32_t id;

private:
    ComponentID(uint32_t i) : id(i) {};
friend class StateManager;
};

struct ArchetypeID {
    uint32_t id;

private:
    ArchetypeID(uint32_t i) : id(i) {};
friend class StateManager;
};

class ECSRegistry {
public:
    ECSRegistry(StateManager &state_mgr);

    template <typename ComponentT>
    void registerComponent();

    template <typename ArchetypeT>
    void registerArchetype();

    template <typename SingletonT>
    void registerSingleton();

private:
    StateManager *state_mgr_;
};

struct EntityStore {
    static constexpr CountT maxEntities = 1048576;

    struct EntitySlot {
        Loc loc;
        uint32_t gen;
    };

    std::atomic_int32_t availableOffset = 0;
    std::atomic_int32_t deletedOffset = 0;
    std::array<EntitySlot, maxEntities> entities;
    std::array<int32_t, maxEntities> availableEntities;
    std::array<int32_t, maxEntities> deletedEntities;
};

class StateManager {
public:
    StateManager(uint32_t max_components);

    template <typename ComponentT>
    ComponentID registerComponent();

    template <typename ArchetypeT>
    ArchetypeID registerArchetype();

    template <typename SingletonT>
    void registerSingleton();

    template <typename SingletonT>
    SingletonT & getSingleton(WorldID world_id);

    template <typename... ComponentTs>
    Query<ComponentTs...> query();

    template <int32_t num_components, typename Fn>
    void iterateArchetypesRaw(QueryRef *query_ref, Fn &&fn);

    inline uint32_t numMatchingEntities(QueryRef *query_ref);

    template <typename ArchetypeT>
    Entity makeEntityNow(WorldID world_id);

    template <typename ArchetypeT>
    void destroyEntityNow(Entity e);

    template <typename ArchetypeT>
    Loc makeTemporary(WorldID world_id);

    template <typename ArchetypeT>
    void clearTemporaries();

    void clearTemporaries(uint32_t archetype_id);

    template <typename ComponentT>
    ComponentT & getUnsafe(Entity e);

    template <typename ComponentT>
    ComponentT & getUnsafe(Loc loc);

    template <typename ArchetypeT, typename ComponentT>
    ComponentT * getArchetypeColumn();

private:
    template <typename SingletonT>
    struct SingletonArchetype : public madrona::Archetype<SingletonT> {};

    using ColumnMap = StaticIntegerMap<Table::maxColumns>;
    static constexpr uint32_t max_archetype_components_ = ColumnMap::numFree();

    static inline uint32_t num_components_ = 0;
    static inline uint32_t num_archetypes_ = 0;

    static constexpr uint32_t max_components_ = 512;
    static constexpr uint32_t max_archetypes_ = 128;
    static constexpr uint32_t user_component_offset_ = 2;
    static constexpr uint32_t max_query_slots_ = 65536;

    void registerComponent(uint32_t id, uint32_t alignment,
                           uint32_t num_bytes);
    void registerArchetype(uint32_t id, ComponentID *components,
                           uint32_t num_components);

    template <typename Fn, int32_t... Indices>
    void iterateArchetypesRawImpl(QueryRef *query_ref, Fn &&fn,
                                  std::integer_sequence<int32_t, Indices...>);

    void makeQuery(const uint32_t *components,
                   uint32_t num_components,
                   QueryRef *query_ref);

    struct ArchetypeStore {
        ArchetypeStore(uint32_t offset, uint32_t num_user_components,
                       uint32_t num_columns,
                       TypeInfo *type_infos, IntegerMapPair *lookup_input);
        uint32_t componentOffset;
        uint32_t numUserComponents;
        Table tbl;
        ColumnMap columnLookup;

        static constexpr uint32_t maxRowsPerTable = 131072;
    };

    uint32_t archetype_component_offset_ = 0;
    uint32_t query_data_offset_ = 0;
    utils::SpinLock query_data_lock_ {};
    FixedInlineArray<Optional<TypeInfo>, max_components_> components_ {};
    std::array<uint32_t, max_archetype_components_ * max_archetypes_>
        archetype_components_ {};
    FixedInlineArray<Optional<ArchetypeStore>, max_archetypes_> archetypes_ {};
    std::array<uint32_t, max_query_slots_> query_data_ {};
    EntityStore entity_store_;
};

}

#include "state.inl"
