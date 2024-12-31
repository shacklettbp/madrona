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
#include <madrona/memory_range.hpp>

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

    SpinLock growLock {};
};

struct MemoryRangeElementStore {
    struct Slot {
        Loc loc;
        uint32_t gen;
    };

    AtomicI32 availableOffset = 0;
    AtomicI32 deletedOffset = 0;

    Slot *slots;
    int32_t *availableSlots;
    int32_t *deletedSlots;

    int32_t numMappedSlots;
    uint32_t numGrowSlots;
    uint32_t numSlotGrowBytes;
    uint32_t numIdxGrowBytes;

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

    template <typename ElementT>
    MemoryRangeElementID registerMemoryRangeElement();

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

    MemoryRange allocMemoryRange(WorldID world_id,
                           uint32_t elem_id, uint32_t num_elems);
    void freeMemoryRange(MemoryRange memory_range);

    template <typename ElementT>
    ElementT * memoryRangePointer(MemoryRange memory_range);

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

    // Returns pointer to MemoryRange
    inline void * getMemoryRangeColumn(uint32_t unit_id);
    // Returns pointer to status values of the units
    inline void * getMemoryRangeStatus(uint32_t unit_id);
    // Returns pointer to the actual units
    inline void * getMemoryRangeUnits(uint32_t unit_id);

    // 0: RangeMap *
    // 1: RangeMap::Status *
    // 2: RangeMapUnitT *
    inline void * getMemoryRangeColumn(uint32_t elem_id, uint32_t col_idx);

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

    inline uint32_t getMemoryRangeColumnBytesPerRow(uint32_t element_id,
                                                 uint32_t column_idx);

    template <typename ArchetypeT>
    inline uint32_t getArchetypeNumRows();

    // The archetype has to be sorted by world for this to work
    // TODO: choose better naming. we stick with this for the sake
    // of sticking to what the CPU backend StateManager has
    template <typename ArchetypeT>
    inline uint32_t numRows(uint32_t world_id);
    uint32_t getArchetypeNumRowsInWorld(uint32_t archetype_id, uint32_t world_id);

    // The archetype has to be sorted by world for this to work
    template <typename ArchetypeT, typename ComponentT>
    inline ComponentT * getWorldComponents(uint32_t world_id);

    template <typename ElementT>
    inline uint32_t getMemoryRangeNumRows();
    inline uint32_t getMemoryRangeNumRows(uint32_t element_id);

    inline int32_t getArchetypeNumColumns(uint32_t archetype_id);
    inline uint32_t getArchetypeMaxColumnSize(uint32_t archetype_id);

    inline uint32_t getMemoryRangeMaxColumnSize(uint32_t unit_id);

    inline void remapEntity(Entity e, int32_t row_idx);
    inline void remapMemoryRangeElement(MemoryRange mr, int32_t row_idx);

    template <typename SingletonT>
    SingletonT * getSingletonColumn();

    void resizeArchetype(uint32_t archetype_id, int32_t num_rows);
    int32_t numArchetypeRows(uint32_t archetype_id) const;

    void resizeMemoryRange(uint32_t element_id, int32_t num_rows);

    std::pair<int32_t, int32_t> fetchRecyclableEntities();
    std::pair<int32_t, int32_t> fetchRecyclableMemoryRanges();

    void recycleEntities(int32_t thread_offset,
                         int32_t recycle_base);
    void recycleMemoryRanges(int32_t thread_offset,
                             int32_t recycle_base);

    inline bool archetypeNeedsSort(uint32_t archetype_id) const;
    inline void archetypeClearNeedsSort(uint32_t archetype_id);
    inline void archetypeSetNeedsSort(uint32_t archetype_id);

    inline bool memoryRangeNeedsSort(uint32_t unit_id) const;
    inline void memoryRangeClearNeedsSort(uint32_t unit_id);
    inline void memoryRangeSetNeedsSort(uint32_t unit_id);

    // Included for compatibility with ECSRegistry
    template <typename ArchetypeT, typename ComponentT>
    ComponentT * exportColumn();
    template <typename SingletonT>
    SingletonT * exportSingleton();

private:
    template <typename SingletonT>
    struct SingletonArchetype : public madrona::Archetype<SingletonT> {};

    using ColumnMap = StaticIntegerMap<1024>;
    static inline constexpr uint32_t max_archetype_components_ = ColumnMap::capacity();
    static constexpr uint32_t bundle_typeid_mask_ = 0x8000'0000_u32;

    static inline uint32_t num_components_ = 0;
    static inline uint32_t num_archetypes_ = 0;
    static inline uint32_t num_memory_range_elements_ = 0;
    static inline uint32_t next_bundle_id_ = bundle_typeid_mask_;

    static inline constexpr uint32_t max_components_ = 1024;
    static inline constexpr uint32_t max_memory_range_elements_ = 64;
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
    void registerMemoryRangeElement(uint32_t id, uint32_t alignment,
                                    uint32_t num_bytes);

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

    // This is basically a more stripped down version of an Archetype.
    struct MemoryRangeStore {
        MemoryRangeStore(uint32_t max_num_elements,
                         TypeInfo type_info);

        TypeInfo typeInfo;
        MemoryRangeTable tbl;

        bool needsSort;
    };

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
        ArchetypeTable tbl;
        ColumnMap columnLookup;
        
        // The size of this array corresponds to the number of worlds
        int32_t *worldOffsets;
        int32_t *worldCounts;

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

    FixedInlineArray<Optional<MemoryRangeStore>, max_memory_range_elements_>
        memory_range_elements_ {};

    std::array<uint32_t, max_archetype_components_ * max_archetypes_>
        archetype_components_ {};

    FixedInlineArray<Optional<ArchetypeStore>, max_archetypes_> archetypes_ {};
    std::array<uint32_t, max_archetype_components_ * max_archetypes_>
        bundle_components_ {};
    std::array<BundleInfo, max_bundles_> bundle_infos_ {};
    std::array<uint32_t, max_query_slots_> query_data_ {};
    EntityStore entity_store_;

    MemoryRangeElementStore mr_element_store_;
};

}

#include "state.inl"
