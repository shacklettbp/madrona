/*
 * Copyright 2021-2022 Brennan Shacklett and contributors
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 */
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
#include <madrona/sync.hpp>
#include <madrona/impl/id_map.hpp>

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

class Transaction {
private:
    enum Op : uint32_t {
        Make,
        Destroy,
        Modify,
    };

    static constexpr uint32_t bytes_per_block_ = 8192;

    struct Block {
        Block *next;
        uint32_t curOffset;
        uint32_t numEntries;
        char data[bytes_per_block_];
    };

    Block *head;

friend class StateManager;
};

class EntityStore {
private:
    template <typename T>
    struct LockedMapStore {
        VirtualStore store;
        CountT numIDs;
        utils::SpinLock expandLock;

        inline T & operator[](int32_t idx);
        inline const T & operator[](int32_t idx) const;

        LockedMapStore(CountT init_capacity);
        CountT expand(CountT num_new_elems);
    };

    using Map = IDMap<Entity, Loc, LockedMapStore>;
public:
    using Cache = Map::Cache;

    EntityStore();

    inline Loc getLoc(Entity e) const;
    inline Loc getLocUnsafe(int32_t e_id) const;
    inline void setLoc(Entity e, Loc loc);
    inline void setRow(Entity e, uint32_t row);

    Entity newEntity(Cache &cache);
    void freeEntity(Cache &cache, Entity e);

    void bulkFree(Cache &cache, Entity *entities, uint32_t num_entities);

private:
    Map map_;
};

class StateCache {
public:
    StateCache();

private:
    EntityStore::Cache entity_cache_;

friend class StateManager;
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

    inline Loc getLoc(Entity e) const;

    template <typename ComponentT>
    inline ResultRef<ComponentT> get(MADRONA_MW_COND(uint32_t world_id,)
                                     Loc loc);

    template <typename ComponentT>
    inline ResultRef<ComponentT> get(MADRONA_MW_COND(uint32_t world_id,)
                                     Entity entity);

    template <typename ComponentT>
    inline ComponentT & getUnsafe(MADRONA_MW_COND(uint32_t world_id,)
                                  int32_t entity_id);

    template <typename ArchetypeT>
    inline ArchetypeRef<ArchetypeT> archetype(
        MADRONA_MW_COND(uint32_t world_id));

    template <typename... ComponentTs>
    inline Query<ComponentTs...> query();

    template <typename... ComponentTs, typename Fn>
    inline void iterateArchetypes(MADRONA_MW_COND(uint32_t world_id,)
                                  const Query<ComponentTs...> &query, Fn &&fn);

    template <typename... ComponentTs, typename Fn>
    inline void iterateEntities(MADRONA_MW_COND(uint32_t world_id,)
                                const Query<ComponentTs...> &query, Fn &&fn);

    Transaction makeTransaction();
    void commitTransaction(Transaction &&txn);

    template <typename ArchetypeT, typename... Args>
    inline Entity makeEntity(MADRONA_MW_COND(uint32_t world_id,)
                             Transaction &txn, StateCache &cache,
                             Args && ...args);

    void destroyEntity(MADRONA_MW_COND(uint32_t world_id,)
                       Transaction &txn, StateCache &cache, Entity e);

    template <typename ArchetypeT, typename... Args>
    inline Entity makeEntityNow(MADRONA_MW_COND(uint32_t world_id,)
                                StateCache &cache, Args && ...args);

    void destroyEntityNow(MADRONA_MW_COND(uint32_t world_id,)
                          StateCache &cache, Entity e);

    template <typename ArchetypeT>
    inline void clear(MADRONA_MW_COND(uint32_t world_id,) StateCache &cache);

#ifdef MADRONA_MW_MODE
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
#ifdef MADRONA_MW_MODE
        HeapArray<Table> tbls;
#else
        Table tbl;
#endif
        ColumnMap columnLookup;
    };

    struct QueryState {
        QueryState();

        utils::SpinLock lock;
        VirtualArray<uint32_t> queryData;
    };

    template <typename... ComponentTs, typename Fn, uint32_t... Indices>
    void iterateArchetypesImpl(MADRONA_MW_COND(uint32_t world_id,) 
                               const Query<ComponentTs...> &query, Fn &&fn,
                               std::integer_sequence<uint32_t, Indices...>);

    void makeQuery(const ComponentID *components, uint32_t num_components,
                   QueryRef *query_ref);

    void registerComponent(uint32_t id, uint32_t alignment,
                           uint32_t num_bytes);
    void registerArchetype(uint32_t id, Span<ComponentID> components);

    void clear(MADRONA_MW_COND(uint32_t world_id,) StateCache &cache,
               uint32_t archetype_id);

    EntityStore entity_store_;
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
