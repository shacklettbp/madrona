#pragma once

#include <madrona/ecs.hpp>
#include <madrona/heap_array.hpp>
#include <madrona/dyn_array.hpp>
#include <madrona/span.hpp>
#include <madrona/table.hpp>
#include <madrona/query.hpp>
#include <madrona/optional.hpp>
#include <madrona/type_tracker.hpp>
#include <madrona/hashmap.hpp>

namespace madrona {

class StateManager;

struct ArchetypeID {
    uint32_t id;

private:
    ArchetypeID(uint32_t i) : id(i) {};
friend class StateManager;
};

struct ComponentID {
    uint32_t id;

private:
    ComponentID(uint32_t i) : id(i) {};
friend class StateManager;
};

#ifdef MADRONA_MW_MODE
struct WorldIndex : IndexHelper {};

struct WorldID {
    uint32_t id;
};
#endif

class StateManager {
public:
#ifdef MADRONA_MW_MODE
    StateManager(int num_worlds);
#else
    StateManager();
#endif


    template <typename ComponentT>
    ComponentID registerComponent();

    template <typename ArchetypeT>
    ArchetypeID registerArchetype();

    template <typename ComponentT>
    ComponentID componentID() const;

    template <typename ArchetypeT>
    ArchetypeID archetypeID() const;

    template <typename ComponentT>
    inline ResultRef<ComponentT> get(Entity entity);

    template <typename ArchetypeT>
    inline ArchetypeRef<ArchetypeT> archetype();

    template <typename... ComponentTs>
    inline Query<ComponentTs...> query();

    template <typename... ComponentTs, typename Fn>
    inline void iterateArchetypes(Query<ComponentTs...> query, Fn &&fn);

    template <typename... ComponentTs, typename Fn>
    inline void iterateEntities(Query<ComponentTs...> query, Fn &&fn);

    template <typename ArchetypeT, typename... Args>
    inline Entity makeEntity(Args && ...args);

    inline void destroyEntity(Entity e);

#ifdef MADRONA_MW_MODE
    template <typename... ComponentTs, typename Fn>
    inline void iterateArchetypesMW(uint32_t num_worlds,
                                    Query<ComponentTs...> query,
                                    Fn &&fn);

    inline uint32_t numWorlds() const;
#endif
     
private:
    using ColumnMap = StaticIntegerMap<128>;
    static constexpr uint32_t max_archetype_components_ = ColumnMap::numFree();

    struct ArchetypeStore {
        struct Init;
        inline ArchetypeStore(Init &&init);

        uint32_t componentOffset;
        uint32_t numComponents;
        Table tbl;
        ColumnMap columnLookup;
    };

    template <typename ComponentT>
    inline ComponentT *getArchetypeComponent(ArchetypeStore &archetype,
                                             uint32_t col_idx);

    template <typename... ComponentTs, typename Fn, uint32_t... Indices>
    void iterateArchetypesImpl(Query<ComponentTs...> query, Fn &&fn,
                               std::integer_sequence<uint32_t, Indices...>);

    uint32_t makeQuery(const ComponentID *components, uint32_t num_components,
                       uint32_t *offset);

    void registerComponent(uint32_t id, uint32_t alignment,
                           uint32_t num_bytes);
    void registerArchetype(uint32_t id, Span<ComponentID> components);

    DynArray<uint32_t> query_data_;
    DynArray<Optional<TypeInfo>> component_infos_;
    DynArray<ComponentID> archetype_components_;
    DynArray<Optional<ArchetypeStore>> archetype_stores_;

#ifdef MADRONA_MW_MODE
    uint32_t num_worlds_;
    DynArray<WorldIndex> world_indices_;
    utils::SpinLock register_lock_;
#endif

    static constexpr uint32_t user_component_offset_ =
#ifdef MADRONA_MW_MODE
        2;
#else
        1;
#endif

    static uint32_t next_component_id_;
    static uint32_t next_archetype_id_;
};

}

#include "state.inl"
