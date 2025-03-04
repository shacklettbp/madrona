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
#include <madrona/ecs_flags.hpp>
#include <madrona/hashmap.hpp>
#include <madrona/inline_array.hpp>
#include <madrona/sync.hpp>
#include <madrona/query.hpp>
#include <madrona/optional.hpp>
#include <madrona/type_tracker.hpp>
#include <madrona/memory.hpp>

#include "mw_gpu/const.hpp"

namespace madrona {

class StateManager;

namespace mwGPU {
static inline StateManager *getStateManager()
{
    return (StateManager *)GPUImplConsts::get().stateManagerAddr;
}
}

struct EntityStore {
    struct EntitySlot {
        Loc loc;
        uint32_t gen;
    };

    AtomicI32 availableOffset = 0;
    AtomicI32 deletedOffset = 0;

    EntitySlot *entities;
    int32_t *availableEntities;
    int32_t *deletedEntities;

    int32_t numMappedEntities;
    uint32_t numGrowEntities;
    uint32_t numSlotGrowBytes;
    uint32_t numIdxGrowBytes;
    uint32_t numGrows;

    SpinLock growLock {};
};

class StateManager {
public:
    StateManager(uint32_t max_components);

    template <typename ComponentT>
    ComponentID registerComponent(uint32_t num_bytes = 0);

    template <typename ArchetypeT, typename... MetadataComponentTs>
    ArchetypeID registerArchetype(
        ComponentMetadataSelector<MetadataComponentTs...> component_metadatas,
        ArchetypeFlags archetype_flags,
        CountT max_num_entities_per_world);

    template <typename SingletonT>
    void registerSingleton();

    template <typename BundleT>
    void registerBundle();

    template <typename AliasT, typename BundleT>
    void registerBundleAlias();

    template <typename SingletonT>
    SingletonT & getSingleton(WorldID world_id);

    template <typename... ComponentTs>
    Query<ComponentTs...> query();

    template <int32_t num_components, typename Fn>
    void iterateArchetypesRaw(QueryRef *query_ref, Fn &&fn);

    template <int32_t num_components, typename Fn>
    void iterateQuery(uint32_t world_id, QueryRef *query_ref, Fn &&fn);

    inline uint32_t numMatchingEntities(QueryRef *query_ref);

    Entity makeEntityNow(WorldID world_id, uint32_t archetype_id);

    void destroyEntityNow(Entity e);

    Loc makeTemporary(WorldID world_id, uint32_t archetype_id);

    template <typename ArchetypeT>
    void clearTemporaries();

    void clearTemporaries(uint32_t archetype_id);

    inline Loc getLoc(Entity e) const;

    template <typename ComponentT>
    ComponentT & getUnsafe(Entity e);

    template <typename ComponentT>
    ComponentT & getUnsafe(Loc loc);

    template <typename ComponentT>
    inline ResultRef<ComponentT> get(Entity e);

    template <typename ComponentT>
    inline ResultRef<ComponentT> get(Loc loc);

    template <typename ComponentT>
    inline ComponentT & getDirect(int32_t column_idx, Loc loc);

    template <typename ArchetypeT, typename ComponentT>
    ComponentT * getArchetypeComponent();

    template <typename ArchetypeT, typename ComponentT>
    void setArchetypeComponent(void *ptr);

    inline void * getArchetypeComponent(uint32_t archetype_id,
                                        uint32_t component_id);

    inline int32_t getArchetypeColumnIndex(uint32_t archetype_id,
                                           uint32_t component_id);

    inline void * getArchetypeColumn(uint32_t archetype_id,
                                     int32_t column_idx);

    template <typename ArchetypeT, typename ComponentT>
    std::pair<ComponentT *, uint32_t> getWorldComponentsAndCount(
            uint32_t world_id);

    template <typename ArchetypeT>
    Entity * getWorldEntities(uint32_t world_id);

    template <typename ArchetypeT>
    int32_t * getArchetypeWorldOffsets();

    inline int32_t * getArchetypeWorldOffsets(uint32_t archetype_id);

    template <typename ArchetypeT>
    int32_t * getArchetypeWorldCounts();
    
    inline int32_t * getArchetypeWorldCounts(uint32_t archetype_id);

    template <typename ArchetypeT>
    inline void setArchetypeWorldOffsets(void *ptr);

    inline uint32_t getArchetypeColumnBytesPerRow(uint32_t archetype_id,
                                                  int32_t column_idx);

    template <typename ArchetypeT>
    inline uint32_t getArchetypeNumRows();

    inline int32_t getArchetypeNumColumns(uint32_t archetype_id);
    inline uint32_t getArchetypeMaxColumnSize(uint32_t archetype_id);

    inline void remapEntity(Entity e, int32_t row_idx);

    template <typename SingletonT>
    SingletonT * getSingletonColumn();

    void resizeArchetype(uint32_t archetype_id, int32_t num_rows);
    int32_t numArchetypeRows(uint32_t archetype_id) const;

    std::pair<int32_t, int32_t> fetchRecyclableEntities();

    void recycleEntities(int32_t thread_offset,
                         int32_t recycle_base);

    inline bool archetypeNeedsSort(uint32_t archetype_id) const;
    inline void archetypeClearNeedsSort(uint32_t archetype_id);
    inline void archetypeSetNeedsSort(uint32_t archetype_id);

    // Included for compatibility with ECSRegistry
    template <typename ArchetypeT, typename ComponentT>
    ComponentT * exportColumn();
    template <typename SingletonT>
    SingletonT * exportSingleton();

    void freeTables();

private:
    template <typename SingletonT>
    struct SingletonArchetype : public madrona::Archetype<SingletonT> {};

    using ColumnMap = StaticIntegerMap<1024>;
    static inline constexpr uint32_t max_archetype_components_ = ColumnMap::capacity();
    static constexpr uint32_t bundle_typeid_mask_ = 0x8000'0000_u32;

    static inline uint32_t num_components_ = 0;
    static inline uint32_t num_archetypes_ = 0;
    static inline uint32_t next_bundle_id_ = bundle_typeid_mask_;

    static inline constexpr uint32_t max_components_ = 1024;
    static inline constexpr uint32_t max_bundles_ = 512;
    static inline constexpr uint32_t max_archetypes_ = 256;
    static inline constexpr uint32_t user_component_offset_ = 2;
    static inline constexpr uint32_t max_query_slots_ = 65536;
    static inline constexpr int32_t num_elems_per_sort_thread_ = 2;

    void registerComponent(uint32_t id, uint32_t alignment,
                           uint32_t num_bytes);
    void registerArchetype(uint32_t id, 
                           ArchetypeFlags archetype_flags,
                           uint32_t max_num_entities_per_world,
                           ComponentID *components,
                           ComponentFlags *component_flags,
                           uint32_t num_components);

    void registerBundle(uint32_t id,
                        const uint32_t *components,
                        CountT num_components);

    template <typename Fn, int32_t... Indices>
    void iterateArchetypesRawImpl(QueryRef *query_ref, Fn &&fn,
                                  std::integer_sequence<int32_t, Indices...>);
    
    template <typename Fn, int32_t... Indices>
    void iterateQueryImpl(int32_t world_id, QueryRef *query_ref, Fn &&fn,
                                  std::integer_sequence<int32_t, Indices...>);

    void makeQuery(const uint32_t *components,
                   uint32_t num_components,
                   QueryRef *query_ref);

    struct ArchetypeStore {
        ArchetypeStore(uint32_t offset,
                       ArchetypeFlags archetype_flags,
                       uint32_t max_num_entities,
                       uint32_t num_user_components,
                       uint32_t num_columns,
                       TypeInfo *type_infos,
                       IntegerMapPair *lookup_input,
                       ComponentFlags *component_flags);

        uint32_t componentOffset;
        uint32_t numUserComponents;
        Table tbl;
        ColumnMap columnLookup;
        
        // The size of this array corresponds to the number of worlds
        int32_t *worldOffsets;
        int32_t *worldCounts;

        ArchetypeFlags flags;
        bool needsSort;
    };

    struct BundleInfo {
        uint32_t componentOffset;
        uint32_t numComponents;
    };

    uint32_t archetype_component_offset_ = 0;
    uint32_t bundle_component_offset_ = 0;
    uint32_t query_data_offset_ = 0;
    SpinLock query_data_lock_ {};
    FixedInlineArray<Optional<TypeInfo>, max_components_> components_ {};
    std::array<uint32_t, max_archetype_components_ * max_archetypes_>
        archetype_components_ {};
    FixedInlineArray<Optional<ArchetypeStore>, max_archetypes_> archetypes_ {};
    std::array<uint32_t, max_archetype_components_ * max_archetypes_>
        bundle_components_ {};
    std::array<BundleInfo, max_bundles_> bundle_infos_ {};
    std::array<uint32_t, max_query_slots_> query_data_ {};
    EntityStore entity_store_;
};

}

#include "state.inl"
