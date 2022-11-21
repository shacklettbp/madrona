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

class StateManager {
public:
    StateManager(uint32_t max_components);

    template <typename ComponentT>
    ComponentID registerComponent();

    template <typename ArchetypeT>
    ArchetypeID registerArchetype();

    template <typename... ComponentTs>
    Query<ComponentTs...> query();

    template <int32_t num_components, typename Fn>
    void iterateArchetypesRaw(QueryRef *query_ref, Fn &&fn);

    inline uint32_t numMatchingEntities(QueryRef *query_ref);

    template <typename ArchetypeT>
    Entity makeEntityNow();

    template <typename ArchetypeT>
    void clear();

    template <typename ArchetypeT, typename ComponentT>
    ComponentT * getArchetypeColumn();

private:
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

    void makeQuery(const ComponentID *components,
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
    std::array<Optional<TypeInfo>, max_components_> components_;
    std::array<uint32_t, max_archetype_components_ * max_archetypes_>
        archetype_components_;
    std::array<Optional<ArchetypeStore>, max_archetypes_> archetypes_;
    std::array<uint32_t, max_query_slots_> query_data_;
};

}

#include "state.inl"
