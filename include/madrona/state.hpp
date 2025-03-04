/*
 * Copyright 2021-2023 Brennan Shacklett and contributors
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 */
#pragma once

#include <madrona/ecs.hpp>
#include <madrona/ecs_flags.hpp>
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
#include <madrona/virtual.hpp>

namespace madrona {

class StateManager;

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
        SpinLock expandLock;

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
    StateManager(CountT num_worlds);
#else
    StateManager();
#endif

    template <typename ComponentT>
    ComponentID registerComponent(uint32_t num_bytes = 0);

    template <typename ArchetypeT, typename... MetadataComponentTs>
    ArchetypeID registerArchetype(
        ComponentMetadataSelector<MetadataComponentTs...> component_metadata,
        ArchetypeFlags archetype_flags,
        CountT max_num_entities_per_world);

    template <typename SingletonT>
    void registerSingleton();

    template <typename BundleT>
    void registerBundle();

    template <typename AliasT, typename BundleT>
    void registerBundleAlias();

    template <typename ArchetypeT, typename ComponentT>
    ComponentT * exportColumn();

    template <typename SingletonT>
    SingletonT * exportSingleton();

    template <typename ArchetypeT, typename ComponentT>
    ComponentT * getWorldComponents(
        MADRONA_MW_COND(uint32_t world_id));

    template <typename ArchetypeT, typename ComponentT>
    std::pair<ComponentT *, uint32_t> getWorldComponentsAndCount(
            uint32_t world_id);

    template <typename ArchetypeT>
    Entity * getWorldEntities(
        MADRONA_MW_COND(uint32_t world_id));

    void copyInExportedColumns();
    void copyOutExportedColumns();

    template <typename SingletonT>
    SingletonT & getSingleton(MADRONA_MW_COND(uint32_t world_id));

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

    template <typename ComponentT>
    inline ComponentT & getUnsafe(MADRONA_MW_COND(uint32_t world_id,)
                                  Loc loc);

    template <typename ComponentT>
    inline ComponentT & getDirect(MADRONA_MW_COND(uint32_t world_id,)
                                  CountT col_idx,
                                  Loc loc);

    template <typename... ComponentTs>
    inline Query<ComponentTs...> query();

    template <typename... ComponentTs, typename Fn>
    inline void iterateArchetypes(MADRONA_MW_COND(uint32_t world_id,)
                                  const Query<ComponentTs...> &query, Fn &&fn);

    template <typename... ComponentTs, typename Fn>
    inline void iterateQuery(MADRONA_MW_COND(uint32_t world_id,)
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

    template <typename... Args>
    inline Entity makeEntityNow(MADRONA_MW_COND(uint32_t world_id,)
                                StateCache &cache,
                                uint32_t archetype_id,
                                Args && ...args);

    void destroyEntityNow(MADRONA_MW_COND(uint32_t world_id,)
                          StateCache &cache, Entity e);

    template <typename ArchetypeT>
    inline Loc makeTemporary(MADRONA_MW_COND(uint32_t world_id));

    inline Loc makeTemporary(MADRONA_MW_COND(uint32_t world_id,)
                             uint32_t archetype_id);

    template <typename ArchetypeT>
    inline void clear(MADRONA_MW_COND(uint32_t world_id,) StateCache &cache,
                      bool is_temporary);

#ifdef MADRONA_MW_MODE
    inline uint32_t numWorlds() const;
#endif

    void * tmpAlloc(MADRONA_MW_COND(uint32_t world_id,) uint64_t num_bytes);
    void resetTmpAlloc(MADRONA_MW_COND(uint32_t world_id));

    void sortArchetype(MADRONA_MW_COND(uint32_t world_id,)
                       uint32_t archetype_id,
                       uint32_t component_id);

    void compactArchetype(MADRONA_MW_COND(uint32_t world_id,)
                          uint32_t archetype_id);

    template <typename ArchetypeT>
    inline CountT numRows(MADRONA_MW_COND(uint32_t world_id));

private:
    template <typename SingletonT>
    struct SingletonArchetype : public madrona::Archetype<SingletonT> {};

    using ColumnMap = StaticIntegerMap<1024>;
    static constexpr uint32_t max_archetype_components_ = ColumnMap::capacity();

    // FIXME: a lot of the conditional logic in this class could be
    // removed by leveraging the fact that the data structure of 
    // Table is always just an array of pointers
    struct TableStorage {
#ifdef MADRONA_MW_MODE
        struct Fixed {
            Table tbl;
            HeapArray<int32_t> activeRows;
        };

        union {
            HeapArray<Table> tbls;
            Fixed fixed;
        };
        CountT maxNumPerWorld;

        inline TableStorage(Span<TypeInfo> types,
                            CountT num_worlds,
                            CountT max_num_per_world);
        ~TableStorage();
#else
        inline TableStorage(Span<TypeInfo> types);

        Table tbl;
#endif

        template <typename ColumnT>
        inline ColumnT * column(MADRONA_MW_COND(uint32_t world_id,)
                                CountT col_idx);

        inline CountT numRows(MADRONA_MW_COND(uint32_t world_id));

        inline void clear(MADRONA_MW_COND(uint32_t world_id));

        inline CountT addRow(MADRONA_MW_COND(uint32_t world_id));
        inline bool removeRow(MADRONA_MW_COND(uint32_t world_id,) CountT row);
    };

    struct ArchetypeStore {
        struct Init;
        inline ArchetypeStore(Init &&init);

        uint32_t componentOffset;
        uint32_t numComponents;
        TableStorage tblStorage;
        ColumnMap columnLookup;
    };

    struct BundleInfo {
        uint32_t componentOffset;
        uint32_t numComponents;
    };

    struct QueryState {
        QueryState();

        SpinLock lock;
        VirtualArray<uint32_t> queryData;
    };

#ifdef MADRONA_MW_MODE
    struct ExportJob {
        uint32_t archetypeIdx;
        uint32_t columnIdx;
        uint32_t numBytesPerRow;

        uint32_t numMappedChunks;

        VirtualRegion mem;
    };
#endif

    template <typename... ComponentTs, typename Fn, uint32_t... Indices>
    void iterateArchetypesImpl(MADRONA_MW_COND(uint32_t world_id,) 
                               const Query<ComponentTs...> &query, Fn &&fn,
                               std::integer_sequence<uint32_t, Indices...>);

    void makeQuery(const ComponentID *components, uint32_t num_components,
                   QueryRef *query_ref);

    void registerComponent(uint32_t id, uint32_t alignment,
                           uint32_t num_bytes);
    void registerArchetype(uint32_t id,
                           ArchetypeFlags archetype_flags,
                           CountT max_num_entities_per_world,
                           CountT num_user_components,
                           const ComponentID *components,
                           const ComponentFlags *component_flags);

    void registerBundle(uint32_t id,
                        const uint32_t *components,
                        CountT num_components);

    void * exportColumn(uint32_t archetype_id, uint32_t component_id);

    void clear(MADRONA_MW_COND(uint32_t world_id,) StateCache &cache,
               uint32_t archetype_id, bool is_temporary);

    StateCache init_state_cache_; // FIXME remove
    EntityStore entity_store_;
    DynArray<Optional<TypeInfo>> component_infos_;
    DynArray<ComponentID> archetype_components_;
    DynArray<Optional<ArchetypeStore>> archetype_stores_;
    DynArray<uint32_t> bundle_components_;
    DynArray<Optional<BundleInfo>> bundle_infos_;

#ifdef MADRONA_MW_MODE
    DynArray<ExportJob> export_jobs_;
#endif

    // FIXME: TmpAllocator doesn't belong here should be per CPU worker
    struct TmpAllocator {
        struct Block;
        struct Metadata {
            Block *next;
            CountT offset;
        };

        static constexpr inline uint64_t numBlockBytes = 32 * 1024 * 1024;

        static constexpr inline uint64_t numFreeBlockBytes =
            numBlockBytes - sizeof(Metadata);

        struct Block {
            char data[numFreeBlockBytes];
            Metadata metadata;
        };

        static_assert(sizeof(Block) == numBlockBytes);

        Block *cur_block_;

        TmpAllocator();
        ~TmpAllocator();

        inline void * alloc(uint64_t num_bytes);
        void reset();
    };

#ifdef MADRONA_MW_MODE
    HeapArray<TmpAllocator> tmp_allocators_;
#else
    TmpAllocator tmp_allocator_;
#endif

#ifdef MADRONA_MW_MODE
    uint32_t num_worlds_;
    SpinLock register_lock_;
#endif

    static constexpr uint32_t user_component_offset_ =
#ifdef MADRONA_MW_MODE
        2;
#else
        1;
#endif

    static constexpr uint32_t bundle_typeid_mask_ = 0x8000'0000_u32;

    static QueryState query_state_;

    static uint32_t next_component_id_;
    static uint32_t next_archetype_id_;
    static uint32_t next_bundle_id_;
};

}

#include "state.inl"
