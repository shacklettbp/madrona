#pragma once

#include <madrona/utils.hpp>
#include <madrona/mw_gpu/host_print.hpp>

namespace madrona {

template <typename ComponentT>
ComponentID StateManager::registerComponent(uint32_t num_bytes)
{
    uint32_t id = TypeTracker::registerType<ComponentT>(
        &StateManager::num_components_);

    uint32_t component_size = (num_bytes == 0) ?
        sizeof(ComponentT) : num_bytes;

    registerComponent(id, alignof(ComponentT), component_size);

    return ComponentID {
        id,
    };
}

template <typename ArchetypeT, typename... MetadataComponentTs>
ArchetypeID StateManager::registerArchetype(
        ComponentMetadataSelector<MetadataComponentTs...> component_metadatas,
        ArchetypeFlags archetype_flags,
        CountT max_num_entities_per_world,
        const char *dbg_name)
{
    uint32_t archetype_id = TypeTracker::registerType<ArchetypeT>(
        &StateManager::num_archetypes_);

    using Base = typename ArchetypeT::Base;


    using Delegator = utils::PackDelegator<Base>;

    auto [archetype_components, component_flags] = Delegator::call(
        [&component_metadatas]<typename... Args>()
    {
        static_assert(std::is_same_v<Base, Archetype<Args...>>);

        std::array components {
            ComponentID { TypeTracker::typeID<Args>() }
            ...
        };

        #pragma nv_diagnostic push
        #pragma nv_diag_suppress 445
        auto getComponentFlags =
        []<typename ComponentT, typename... FlagComponentTs>(
            const ComponentMetadataSelector<FlagComponentTs...> &component_metadata
        ) constexpr -> ComponentFlags
        {
            constexpr size_t num_metadatas = sizeof...(FlagComponentTs);

            if constexpr (num_metadatas == 0) {
                return ComponentFlags::None;
            } else {
                bool matches[] = {
                    std::is_same_v<ComponentT, FlagComponentTs>
                    ...
                };

                for (size_t i = 0; i < num_metadatas; i++) {
                    if (matches[i]) {
                        return component_metadata.flags[i];
                    }
                }

                return ComponentFlags::None;
            }
        };
        #pragma nv_diagnostic pop

        std::array component_flags {
            getComponentFlags.template operator()<Args>(component_metadatas)
            ...
        };

        return std::make_pair(components, component_flags);
    });

    registerArchetype(archetype_id,
                      archetype_flags,
                      max_num_entities_per_world,
                      archetype_components.data(),
                      component_flags.data(),
                      archetype_components.size());

    if (dbg_name) {
        printf("%s has mapper ID %llu\n",
                dbg_name,
                archetypes_[archetype_id]->tbl.mapperID);
    }

    return ArchetypeID {
        archetype_id,
    };
}

template <typename ElementT>
MemoryRangeElementID StateManager::registerMemoryRangeElement(const char *dbg_name)
{
    uint32_t id = TypeTracker::registerType<ElementT>(
        &StateManager::num_memory_range_elements_);

    uint32_t component_size = sizeof(ElementT);

    registerMemoryRangeElement(id, alignof(ElementT), component_size);

    if (dbg_name) {
        printf("%s has mapper ID %llu\n",
                dbg_name,
                memory_range_elements_[id]->tbl.mapperID);
    }

    return MemoryRangeElementID {
        id,
    };
}

template <typename BundleT>
void StateManager::registerBundle()
{
    uint32_t bundle_id = TypeTracker::registerType<BundleT>(
        &StateManager::next_bundle_id_);

    using Base = typename BundleT::Base;

    using Delegator = utils::PackDelegator<Base>;

    auto bundle_components = Delegator::call(
    []<typename... Args>()
    {
        static_assert(std::is_same_v<Base, Bundle<Args...>>);

        if constexpr (sizeof...(Args) == 0) {
            return std::array<uint32_t, 0> {};
        } else {
            std::array components {
                TypeTracker::typeID<Args>()
                ...
            };

            return components;
        }
    });
    
    uint32_t id = TypeTracker::typeID<BundleT>();

    registerBundle(id, bundle_components.data(), bundle_components.size());
}

template <typename AliasT, typename BundleT>
void StateManager::registerBundleAlias()
{
    uint32_t bundle_id = TypeTracker::typeID<BundleT>();
    assert(bundle_id != TypeTracker::unassignedTypeID);

    TypeTracker::registerType<AliasT>(&bundle_id);
}

template <typename SingletonT>
void StateManager::registerSingleton(uint32_t num_bytes)
{
    uint32_t num_worlds = mwGPU::GPUImplConsts::get().numWorlds;

    using ArchetypeT = SingletonArchetype<SingletonT>;

    if (num_bytes == 0) {
        num_bytes = sizeof(SingletonT);
    }

    registerComponent<SingletonT>(num_bytes);
    registerArchetype<ArchetypeT>(
        ComponentMetadataSelector<> {}, ArchetypeFlags::None, 1);

    for (uint32_t i = 0; i < num_worlds; i++) {
        makeEntityNow(WorldID { int32_t(i) },
                      TypeTracker::typeID<ArchetypeT>());
    }
}

template <typename SingletonT>
SingletonT & StateManager::getSingleton(WorldID world_id)
{
    auto [col, size] = getSingletonColumnAndSize<SingletonT>();
    return *(SingletonT *)((uint8_t *)col + world_id.idx * size);
}

template <typename... ComponentTs>
Query<ComponentTs...> StateManager::query()
{
    std::array component_ids {
        TypeTracker::typeID<ComponentTs>()
        ...
    };

    QueryRef *ref = &Query<ComponentTs...>::ref_;

    // If necessary, create the query templated on the passed in ComponentTs.
    // Double blocks: threads check the atomic here, then additionally
    // attempt to acquire a SpinLock in StateManager::makeQuery.
    if (ref->numReferences.load_acquire() == 0) {
        makeQuery(component_ids.data(), component_ids.size(), ref);
    }

    return Query<ComponentTs...>(true);
}

template <typename Fn, int32_t... Indices>
void StateManager::iterateArchetypesRawImpl(QueryRef *query_ref, Fn &&fn,
        std::integer_sequence<int32_t, Indices...>)
{
    uint32_t *query_values = &query_data_[query_ref->offset];
    int32_t num_archetypes = query_ref->numMatchingArchetypes;

    for (int i = 0; i < num_archetypes; i++) {
        uint32_t archetype_idx = *query_values;
        query_values += 1;

        ArchetypeTable &tbl = archetypes_[archetype_idx]->tbl;

        bool early_out = fn(tbl.numRows.load_relaxed(),
            (WorldID *)(tbl.columns[1]),
            std::pair<void *, uint32_t> { 
                tbl.columns[query_values[Indices]],
                (uint32_t)tbl.columnSizes[query_values[Indices]]
            } ...);
        if (early_out) {
            return;
        }

        query_values += sizeof...(Indices);
    }
}

template <int32_t num_components, typename Fn>
void StateManager::iterateArchetypesRaw(QueryRef *query_ref, Fn &&fn)
{
    using IndicesWrapper =
        std::make_integer_sequence<int32_t, num_components>;

    iterateArchetypesRawImpl(query_ref, std::forward<Fn>(fn),
                             IndicesWrapper());
}

template<typename Fn, int32_t... Indices>
void StateManager::iterateQueryImpl(int32_t world_id, QueryRef *query_ref, 
        Fn &&fn, 
        std::integer_sequence<int32_t, Indices...>) 
{
    uint32_t *query_values = &query_data_[query_ref->offset];
    int32_t num_archetypes = query_ref->numMatchingArchetypes;

    for (int i = 0; i < num_archetypes; i++) {
        uint32_t archetype_idx = *query_values;
        query_values += 1;

        ArchetypeTable &tbl = archetypes_[archetype_idx]->tbl;

        int32_t worldOffset = 
            getArchetypeWorldOffsets(archetype_idx)[world_id];
        int32_t worldArchetypeCount =
            getArchetypeWorldCounts(archetype_idx)[world_id];

        for (int i = 0; i < worldArchetypeCount; ++i) {
            fn(worldOffset + i, tbl.columns[query_values[Indices]] ...);
        }

        query_values += sizeof...(Indices);
    }
}

template<int32_t num_components, typename Fn>
void StateManager::iterateQuery(uint32_t world_id, QueryRef *query_ref,
    Fn &&fn) 
{

    using IndicesWrapper =
        std::make_integer_sequence<int32_t, num_components>;

    iterateQueryImpl(world_id, query_ref, std::forward<Fn>(fn),
                             IndicesWrapper());

}

uint32_t StateManager::numMatchingEntities(QueryRef *query_ref)
{
    uint32_t *query_values = &query_data_[query_ref->offset];
    int32_t num_archetypes = query_ref->numMatchingArchetypes;
    int32_t num_components = query_ref->numComponents;

    uint32_t total_rows = 0;
    for (int i = 0; i < num_archetypes; i++) {
        uint32_t archetype_idx = *query_values;

        ArchetypeTable &tbl = archetypes_[archetype_idx]->tbl;

        total_rows += tbl.numRows.load_relaxed();

        query_values += 1 + num_components;
    }

    return total_rows;
}

template <typename ArchetypeT>
void StateManager::clearTemporaries()
{
    uint32_t archetype_id = TypeTracker::typeID<ArchetypeT>();
    clearTemporaries(archetype_id);
}

inline Loc StateManager::getLoc(Entity e) const
{
    const EntityStore::EntitySlot &slot = entity_store_.entities[e.id];

    if (slot.gen != e.gen) {
        return Loc::none();
    }

    return slot.loc;
}

template <typename ComponentT>
ComponentT & StateManager::getUnsafe(Entity e)
{
    const EntityStore::EntitySlot &slot = entity_store_.entities[e.id];
    assert(slot.gen == e.gen);

    return getUnsafe<ComponentT>(slot.loc);
}

template <typename ComponentT>
ComponentT & StateManager::getUnsafe(Loc loc)
{
    auto &archetype = *archetypes_[loc.archetype];
    uint32_t component_id = TypeTracker::typeID<ComponentT>();
    auto col_idx = archetype.columnLookup.lookup(component_id);
    assert(col_idx.has_value());

    ArchetypeTable &tbl = archetype.tbl;

    return ((ComponentT *)(tbl.columns[*col_idx]))[loc.row];
}

template <typename ComponentT>
ResultRef<ComponentT> StateManager::get(Entity e)
{
    const EntityStore::EntitySlot &slot = entity_store_.entities[e.id];
    if (slot.gen != e.gen) {
        return ResultRef<ComponentT>(nullptr);
    }

    return get<ComponentT>(slot.loc);
}

template <typename ComponentT>
ResultRef<ComponentT> StateManager::get(Loc loc)
{
    auto &archetype = *archetypes_[loc.archetype];
    uint32_t component_id = TypeTracker::typeID<ComponentT>();
    auto col_idx = archetype.columnLookup.lookup(component_id);

    if (!col_idx.has_value()) {
        return ResultRef<ComponentT>(nullptr);
    }

    assert(col_idx.has_value());

    ArchetypeTable &tbl = archetype.tbl;

    return ResultRef<ComponentT>(
        ((ComponentT *)(tbl.columns[*col_idx])) + loc.row);
}

template <typename ArchetypeT, typename ComponentT>
ComponentT * StateManager::getWorldComponents(uint32_t world_id)
{
    uint32_t archetype_id = TypeTracker::typeID<ArchetypeT>();
    auto &archetype = *archetypes_[archetype_id];

    assert(!archetype.needsSort);

    uint32_t component_id = TypeTracker::typeID<ComponentT>();
    auto col_idx = archetype.columnLookup.lookup(component_id);

    if (!col_idx.has_value()) {
        return nullptr;
    }

    assert(col_idx.has_value());

    ComponentT *base_ptr = (ComponentT *)(archetype.tbl.columns[*col_idx]);
    return base_ptr + archetype.worldOffsets[world_id];
}

template <typename ArchetypeT>
inline Entity * StateManager::getWorldEntities(uint32_t world_id)
{
    uint32_t archetype_id = TypeTracker::typeID<ArchetypeT>();
    auto &archetype = *archetypes_[archetype_id];

    assert(!archetype.needsSort);

    uint32_t col_idx = 0;

    Entity *base_ptr = (Entity *)(archetype.tbl.columns[0]);
    return base_ptr + archetype.worldOffsets[world_id];
}

template <typename ComponentT>
ComponentT & StateManager::getDirect(int32_t column_idx, Loc loc)
{
    return ((ComponentT *)getArchetypeColumn(
        loc.archetype, column_idx))[loc.row];
}

template <typename ArchetypeT, typename ComponentT>
ComponentT * StateManager::getArchetypeComponent()
{
    uint32_t archetype_id = TypeTracker::typeID<ArchetypeT>();
    uint32_t component_id = TypeTracker::typeID<ComponentT>();

    return (ComponentT *)getArchetypeComponent(archetype_id, component_id);
}

template <typename ArchetypeT, typename ComponentT>
std::pair<ComponentT *, uint32_t> StateManager::getWorldComponentsAndCount(
        uint32_t world_id)
{
    ComponentT *glob_comps = getArchetypeComponent<
        ArchetypeT, ComponentT>();
    int32_t *world_offsets = getArchetypeWorldOffsets<
        ArchetypeT>();
    int32_t *world_counts = getArchetypeWorldCounts<
        ArchetypeT>();

    return std::make_pair(glob_comps + world_offsets[world_id],
            world_counts[world_id]);
}

template <typename ArchetypeT, typename ComponentT>
void StateManager::setArchetypeComponent(void *ptr)
{
    uint32_t archetype_id = TypeTracker::typeID<ArchetypeT>();
    uint32_t component_id = TypeTracker::typeID<ComponentT>();

    auto &archetype = *archetypes_[archetype_id];
    uint32_t col_index = getArchetypeColumnIndex(archetype_id, component_id);
    archetype.tbl.columns[col_index] = ptr;
}

void * StateManager::getArchetypeComponent(uint32_t archetype_id,
                                           uint32_t component_id)
{
    auto &archetype = *archetypes_[archetype_id];
    int32_t col_idx = getArchetypeColumnIndex(archetype_id, component_id);
    return archetype.tbl.columns[col_idx];
}

template <typename ArchetypeT>
int32_t * StateManager::getArchetypeWorldOffsets()
{
    uint32_t archetype_id = TypeTracker::typeID<ArchetypeT>();

    return getArchetypeWorldOffsets(archetype_id);
}

int32_t * StateManager::getArchetypeWorldOffsets(uint32_t archetype_id)
{
    auto &archetype = *archetypes_[archetype_id];
    return archetype.worldOffsets;
}

template <typename ArchetypeT>
int32_t * StateManager::getArchetypeWorldCounts()
{
    uint32_t archetype_id = TypeTracker::typeID<ArchetypeT>();

    return getArchetypeWorldCounts(archetype_id);
}

int32_t * StateManager::getArchetypeWorldCounts(uint32_t archetype_id)
{
    auto &archetype = *archetypes_[archetype_id];
    return archetype.worldCounts;
}

int32_t StateManager::getArchetypeColumnIndex(uint32_t archetype_id,
                                              uint32_t component_id)
{
    auto &archetype = *archetypes_[archetype_id];
    if (component_id == 0) {
        return 0; // Entity
    } else if (component_id == 1) {
        return 1; // WorldID
    } else {
        return *archetype.columnLookup.lookup(component_id);
    }
}

template <typename ElementT>
ElementT * StateManager::memoryRangePointer(MemoryRange memory_range)
{
    const MemoryRangeElementStore::Slot &slot =
        mr_element_store_.slots[memory_range.id];
    assert(slot.gen == memory_range.gen);

    auto &element = *memory_range_elements_[slot.loc.archetype];

    return (ElementT *)element.tbl.columns[2] + slot.loc.row;
}

void * StateManager::memoryRangePointer(MemoryRange memory_range)
{
    const MemoryRangeElementStore::Slot &slot = 
        mr_element_store_.slots[memory_range.id];
    assert(slot.gen == memory_range.gen);

    auto &element = *memory_range_elements_[slot.loc.archetype];

    uint8_t *start = (uint8_t *)element.tbl.columns[2];
    return (void *)(start + slot.loc.row * element.tbl.columnSizes[2]);
}

void * StateManager::getMemoryRangeColumn(
        uint32_t element_id, uint32_t col_idx)
{
    return memory_range_elements_[element_id]->tbl.columns[col_idx];
}

template <typename ArchetypeT>
inline uint32_t StateManager::getArchetypeNumRows()
{
    uint32_t archetype_id = TypeTracker::typeID<ArchetypeT>();
    auto &archetype = *archetypes_[archetype_id];
    return archetype.tbl.numRows.load_relaxed();
}

template <typename ArchetypeT>
inline uint32_t StateManager::numRows(uint32_t world_id)
{
    uint32_t archetype_id = TypeTracker::typeID<ArchetypeT>();
    return getArchetypeNumRowsInWorld(archetype_id, world_id);
}

template <typename ElementT>
inline uint32_t StateManager::getMemoryRangeNumRows()
{
    uint32_t element_id = TypeTracker::typeID<ElementT>();
    auto &range = *memory_range_elements_[element_id];
    return range.tbl.numRows.load_relaxed();
}

inline uint32_t StateManager::getMemoryRangeNumRows(uint32_t element_id)
{
    auto &range = *memory_range_elements_[element_id];
    return range.tbl.numRows.load_relaxed();
}

void * StateManager::getArchetypeColumn(uint32_t archetype_id,
                                        int32_t column_idx)
{
    auto &archetype = *archetypes_[archetype_id];
    return archetype.tbl.columns[column_idx];
}

template <typename ArchetypeT>
void StateManager::setArchetypeWorldOffsets(void *ptr)
{
    uint32_t archetype_id = TypeTracker::typeID<ArchetypeT>();
    auto &archetype = *archetypes_[archetype_id];
    archetype.worldOffsets = (int32_t *)ptr;
}

uint32_t StateManager::getArchetypeColumnBytesPerRow(uint32_t archetype_id,
                                                     int32_t column_idx)
{
    auto &archetype = *archetypes_[archetype_id];
    return archetype.tbl.columnSizes[column_idx];
}

uint32_t StateManager::getMemoryRangeColumnBytesPerRow(uint32_t element_id,
                                                       uint32_t column_idx)
{
    auto &unit = *memory_range_elements_[element_id];
    return unit.tbl.columnSizes[column_idx];
}

int32_t StateManager::getArchetypeNumColumns(uint32_t archetype_id)
{
    auto &archetype = *archetypes_[archetype_id];
    return archetype.tbl.numColumns;
}

uint32_t StateManager::getArchetypeMaxColumnSize(uint32_t archetype_id)
{
    auto &archetype = *archetypes_[archetype_id];
    return archetype.tbl.maxColumnSize;
}

inline uint32_t StateManager::getMemoryRangeMaxColumnSize(uint32_t element_id)
{
    auto &range = *memory_range_elements_[element_id];
    return range.tbl.maxColumnSize;
}

void StateManager::remapEntity(Entity e, int32_t row_idx)
{
    entity_store_.entities[e.id].loc.row = row_idx;
}

void StateManager::remapMemoryRangeElement(MemoryRange mr, int32_t row_idx)
{
    mr_element_store_.slots[mr.id].loc.row = row_idx;
}

template <typename SingletonT>
std::pair<SingletonT *, uint32_t> StateManager::getSingletonColumnAndSize()
{
    using ArchetypeT = SingletonArchetype<SingletonT>;
    uint32_t archetype_id = TypeTracker::typeID<ArchetypeT>();

    // Abuse the fact that the singleton only has one component that is going
    // to be in column 2
    
    ArchetypeTable &tbl = archetypes_[archetype_id]->tbl;
    return { (SingletonT *)tbl.columns[2], tbl.columnSizes[2] };
}

template <typename SingletonT>
SingletonT * StateManager::getSingletonColumn()
{
    using ArchetypeT = SingletonArchetype<SingletonT>;
    uint32_t archetype_id = TypeTracker::typeID<ArchetypeT>();

    // Abuse the fact that the singleton only has one component that is going
    // to be in column 2
    
    ArchetypeTable &tbl = archetypes_[archetype_id]->tbl;
    return (SingletonT *)tbl.columns[2];
}

bool StateManager::archetypeNeedsSort(uint32_t archetype_id) const
{
    return archetypes_[archetype_id]->needsSort;
}

void StateManager::archetypeClearNeedsSort(uint32_t archetype_id)
{
    archetypes_[archetype_id]->needsSort = false;
}

void StateManager::archetypeSetNeedsSort(uint32_t archetype_id)
{
    archetypes_[archetype_id]->needsSort = true;
}

bool StateManager::memoryRangeNeedsSort(uint32_t element_id) const
{
    return memory_range_elements_[element_id]->needsSort;
}

void StateManager::memoryRangeClearNeedsSort(uint32_t element_id)
{
    memory_range_elements_[element_id]->needsSort = false;
}

void StateManager::memoryRangeSetNeedsSort(uint32_t unit_id)
{
    memory_range_elements_[unit_id]->needsSort = true;
}

template <typename ArchetypeT, typename ComponentT>
ComponentT * StateManager::exportColumn()
{
    return getArchetypeComponent<ArchetypeT, ComponentT>();
}

template <typename SingletonT>
SingletonT * StateManager::exportSingleton()
{
    return getSingletonColumn<SingletonT>();
}

uint32_t StateManager::getNumCheckpoints() const
{
    return num_checkpoints_;
}

inline uint32_t StateManager::getCheckpointElemSize()
{
    return checkpoint_element_num_bytes_;
}

}
