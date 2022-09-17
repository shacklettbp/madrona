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
struct WorldID {
    uint32_t id;
};
#endif

class IDManager {
public:
    IDManager();
    inline Loc getLoc(Entity e) const;
    inline void updateLoc(Entity e, uint32_t row);

    Entity newEntity(uint32_t archetype, uint32_t row);
    void freeEntity(Entity e);

    void bulkFree(Entity *entities, uint32_t num_entities);

private:
    struct GenLoc {
        Loc loc;
        uint32_t gen;
    };

    inline GenLoc * getGenLoc(uint32_t id);
    inline const GenLoc * getGenLoc(uint32_t id) const;

    VirtualStore store_;
    uint32_t num_ids_;
    uint32_t free_id_head_;
};

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
    inline void iterateArchetypes(const Query<ComponentTs...> &query, Fn &&fn);

    template <typename... ComponentTs, typename Fn>
    inline void iterateEntities(const Query<ComponentTs...> &query, Fn &&fn);

    template <typename ArchetypeT, typename... Args>
    inline Entity makeEntity(Args && ...args);

    void destroyEntity(Entity e);

    template <typename ArchetypeT>
    inline void reset();

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

    struct QueryState {
        QueryState();

        utils::SpinLock lock;
        VirtualArray<uint32_t> queryData;
    };

    template <typename... ComponentTs, typename Fn, uint32_t... Indices>
    void iterateArchetypesImpl(const Query<ComponentTs...> &query, Fn &&fn,
                               std::integer_sequence<uint32_t, Indices...>);

    void makeQuery(const ComponentID *components, uint32_t num_components,
                   QueryRef *query_ref);

    void registerComponent(uint32_t id, uint32_t alignment,
                           uint32_t num_bytes);
    void registerArchetype(uint32_t id, Span<ComponentID> components);

    void reset(uint32_t archetype_id);

    IDManager id_mgr_;
    DynArray<Optional<TypeInfo>> component_infos_;
    DynArray<ComponentID> archetype_components_;
    DynArray<Optional<ArchetypeStore>> archetype_stores_;

#ifdef MADRONA_MW_MODE
    uint32_t num_worlds_;
    utils::SpinLock register_lock_;
#endif

    static constexpr uint32_t user_component_offset_ =
#ifdef MADRONA_MW_MODE
        2;
#else
        1;
#endif

    static QueryState query_state_;

    static uint32_t next_component_id_;
    static uint32_t next_archetype_id_;
};

}

#include "state.inl"
