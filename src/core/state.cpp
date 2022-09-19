#include <madrona/state.hpp>
#include <madrona/utils.hpp>
#include <madrona/dyn_array.hpp>

#include <cassert>
#include <functional>
#include <mutex>
#include <string_view>

namespace madrona {

namespace ICfg {
static constexpr uint32_t maxQueryOffsets = 100'000;
static constexpr uint32_t idsPerCache = 64;
}

EntityStore::Cache::Cache()
    : free_head_(~0u),
      num_free_ids_(0),
      overflow_head_(~0u),
      num_overflow_ids_(0)
{}

StateCache::StateCache()
    : entity_cache_()
{}

EntityStore::EntityStore()
    : store_(sizeof(GenLoc), alignof(GenLoc), 0, ~0u),
      num_ids_(0),
      expand_lock_{},
      free_head_(FreeHead {
          .gen = 0,
          .head = ~0u,
      })
{}

Entity EntityStore::newEntity(Cache &cache,
                                uint32_t archetype, uint32_t row)
{
    auto assignCachedID = [this, archetype, row](uint32_t *head) {
        uint32_t new_id = *head;
        GenLoc *free_loc = getGenLoc(new_id);

        // archetype is overloaded for GenLoc in the freelist
        // to represent multiple contiguous free IDs
        uint32_t num_contiguous = free_loc->loc.archetype;

        if (num_contiguous == 1) {
            *head = free_loc->loc.row;
        } else {
            uint32_t next_free = new_id + 1;
            GenLoc *next_free_loc = getGenLoc(next_free);
            *next_free_loc = GenLoc {
                .loc = Loc {
                    .archetype = num_contiguous - 1,
                    .row = free_loc->loc.row,
                },
                .gen = 0,
            };
            *head = next_free;
        }

        free_loc->loc = Loc {
            .archetype = archetype,
            .row = row,
        };

        return Entity {
            .gen = free_loc->gen,
            .id = new_id,
        };
    };

    // First, check if there is a free entity in the overflow cache
    if (cache.num_overflow_ids_ > 0) {
        cache.num_overflow_ids_ -= 1;
        return assignCachedID(&cache.overflow_head_);
    }

    // Next, check the main cache
    if (cache.num_free_ids_ > 0) {
        cache.num_free_ids_ -= 1;
        return assignCachedID(&cache.free_head_);
    }

    // No free IDs, refill the cache from the global free list
    FreeHead cur_head = free_head_.load(std::memory_order_acquire);
    FreeHead new_head;
    GenLoc *cur_head_loc;

    do {
        if (cur_head.head == ~0u) {
            break;
        }

        new_head.gen = cur_head.gen + 1;
        cur_head_loc = getGenLoc(cur_head.head);

        // On the global list, 'archetype' acts as the next pointer. This
        // preserves 'row' for the sublists added by the caches, each of
        // which is guaranteed to be ICfg::idsPerCache in size.
        // This works, because there are guaranteed to be no contiguous
        // blocks by the time a sublist is added to the global list, so
        // the value of archetype is no longer needed in the sublist
        // context
        new_head.head = cur_head_loc->loc.archetype;
    } while (!free_head_.compare_exchange_weak(
        cur_head, new_head, std::memory_order_release,
        std::memory_order_acquire));

    uint32_t free_ids = cur_head.head;

    if (free_ids != ~0u) {
        // Assign archetype to 1 (id block of size 1) so as to not confuse
        // the cache's processing of the freelist
        cur_head_loc->loc.archetype = 1;

        cache.free_head_ = free_ids;
        cache.num_free_ids_ = ICfg::idsPerCache - 1;
        return assignCachedID(&cache.free_head_);
    }

    // No free IDs at all, expand the ID store
    uint32_t block_start;
    {
        std::lock_guard lock(expand_lock_);

        // Note there is no double checked locking here.
        // It's possible that free ids have been returned at this point,
        // but if there is that much contention overallocating IDs seems
        // relatively harmless

        block_start = num_ids_;
        num_ids_ += ICfg::idsPerCache;
        store_.expand(num_ids_);
    }

    uint32_t first_id = block_start;

    GenLoc *entity_loc = getGenLoc(first_id);
    *entity_loc = {
        .loc = Loc {
            .archetype = archetype,
            .row = row,
        },
        .gen = 0,
    };

    uint32_t free_start = block_start + 1;
    GenLoc *free_loc = getGenLoc(free_start);

    *free_loc = {
        .loc = Loc {
            // In the free list, archetype is overloaded
            // to handle contiguous free elements
            .archetype = ICfg::idsPerCache - 1,
            .row = ~0u,
        },
        .gen = 0,
    };

    cache.free_head_ = free_start;
    cache.num_free_ids_ = ICfg::idsPerCache - 1;

    return Entity {
        .gen = 0,
        .id = first_id,
    };
}

void EntityStore::freeEntity(Cache &cache, Entity e)
{
    GenLoc *gen_loc = getGenLoc(e.id);
    gen_loc->gen++;
    gen_loc->loc.archetype = 1;

    if (cache.num_free_ids_ < ICfg::idsPerCache) {
        gen_loc->loc.row = cache.free_head_;
        cache.free_head_ = e.id;
        cache.num_free_ids_ += 1;

        return;
    }

    if (cache.num_overflow_ids_ < ICfg::idsPerCache) {
        gen_loc->loc.row = cache.overflow_head_;
        cache.overflow_head_ = e.id;
        cache.num_overflow_ids_ += 1;
    }

    // If overflow cache is too big return it to the global free list
    if (cache.num_overflow_ids_ == ICfg::idsPerCache) {
        FreeHead cur_head = free_head_.load(std::memory_order_relaxed);
        FreeHead new_head;
        new_head.head = cache.overflow_head_;
        GenLoc *new_loc = getGenLoc(cache.overflow_head_);

        do {
            new_head.gen = cur_head.gen + 1;
            new_loc->loc.archetype = cur_head.head;
        } while (free_head_.compare_exchange_weak(
            cur_head, new_head, std::memory_order_release,
            std::memory_order_relaxed));

        cache.overflow_head_ = ~0u;
        cache.num_overflow_ids_ = 0;
    }
}

void EntityStore::bulkFree(Cache &cache, Entity *entities,
                           uint32_t num_entities)
{
    if (num_entities == 0) return;

    // The trick with this function is that the sublists added to the
    // global free list need to be exactly ICfg::idsPerCache in size
    // num_entities may not be divisible, so use the cache for any overflow

    auto linkToNext = [this, entities](int idx) {
        Entity cur = entities[idx];
        Entity next = entities[idx + 1];

        GenLoc *gen_loc = getGenLoc(cur.id);
        gen_loc->gen++;
        gen_loc->loc.row = next.id;
        gen_loc->loc.archetype = 1;
    };
    
    int base_idx;
    uint32_t num_remaining;
    GenLoc *global_tail_loc = nullptr;
    for (base_idx = 0; base_idx < (int)num_entities;
         base_idx += ICfg::idsPerCache) {

        num_remaining = num_entities - base_idx;
        if (num_remaining < ICfg::idsPerCache) {
            break;
        }

        uint32_t head_id = entities[base_idx].id;

        for (int sub_idx = 0; sub_idx < (int)ICfg::idsPerCache; sub_idx++) {
            int idx = base_idx + sub_idx;
            linkToNext(idx);
        }

        GenLoc *last_loc = getGenLoc(entities[ICfg::idsPerCache - 1].id);
        last_loc->gen++;
        last_loc->loc.row = ~0u;
        last_loc->loc.archetype = 1;

        if (global_tail_loc != nullptr) {
            global_tail_loc->loc.archetype = head_id;
        }

        global_tail_loc = getGenLoc(head_id);
    }

    // The final chunk has an odd size we need to take care of
    if (num_remaining != ICfg::idsPerCache) {
        uint32_t start_id = entities[base_idx].id;
        for (int idx = base_idx; idx < (int)num_entities - 1; idx++) {
            linkToNext(idx);
        }

        GenLoc *tail_loc = getGenLoc(entities[num_entities - 1].id);
        tail_loc->gen++;
        tail_loc->loc.archetype = 1;
        tail_loc->loc.row = cache.overflow_head_;

        uint32_t num_from_overflow = ICfg::idsPerCache - num_remaining;
        if (cache.num_overflow_ids_ < num_from_overflow) {
            // The extra IDs fit in the overflow cache
            cache.overflow_head_ = start_id;
            cache.num_overflow_ids_ += num_remaining;
        } else {
            // The extra IDs don't fit in the overflow cache, need to add to global list
            uint32_t next_id = cache.overflow_head_;
            GenLoc *overflow_loc;
            for (int i = 0; i < (int)num_from_overflow; i++) {
                overflow_loc = getGenLoc(next_id);
                next_id = overflow_loc->loc.row;
            }

            overflow_loc->loc.row = ~0u;
            cache.overflow_head_ = next_id;
            cache.num_overflow_ids_ -= num_from_overflow;

            if (global_tail_loc != nullptr) {
                global_tail_loc->loc.archetype = start_id;
            }
            global_tail_loc = getGenLoc(start_id);
        }
    }

    // If global_tail_loc is still unset, there is no full sublist to add
    // to the global list
    if (global_tail_loc != nullptr) {
        return;
    }

    uint32_t new_global_head = entities[0].id;

    FreeHead cur_head = free_head_.load(std::memory_order_relaxed);
    FreeHead new_head;
    new_head.head = new_global_head;

    do {
        new_head.gen = cur_head.gen + 1;
        global_tail_loc->loc.archetype = cur_head.head;
    } while (free_head_.compare_exchange_weak(
        cur_head, new_head, std::memory_order_release,
        std::memory_order_relaxed));
}

#ifdef MADRONA_MW_MODE
StateManager::StateManager(int num_worlds)
    : entity_store_(),
      component_infos_(0),
      archetype_components_(0),
      archetype_stores_(0),
      num_worlds_(num_worlds),
      register_lock_()
{
    registerComponent<Entity>();
    registerComponent<WorldID>();
}
#else
StateManager::StateManager()
    : entity_store_(),
      component_infos_(0),
      archetype_components_(0),
      archetype_stores_(0)
{
    registerComponent<Entity>();
}
#endif

void StateManager::destroyEntityNow(StateCache &cache, Entity e)
{
    Loc loc = entity_store_.getLoc(e);
    
    if (!loc.valid()) {
        return;
    }

    ArchetypeStore &archetype = *archetype_stores_[loc.archetype];
    bool row_moved = archetype.tbl.removeRow(loc.row);

    if (row_moved) {
        Entity moved_entity = ((Entity *)archetype.tbl.data(0))[loc.row];
        entity_store_.updateLoc(moved_entity, loc.row);
    }

    entity_store_.freeEntity(cache.entity_cache_, e);
}

struct StateManager::ArchetypeStore::Init {
    uint32_t componentOffset;
    uint32_t numComponents;
    uint32_t id;
    Span<TypeInfo> types;
    Span<IntegerMapPair> lookupInputs;
};

StateManager::ArchetypeStore::ArchetypeStore(Init &&init)
    : componentOffset(init.componentOffset),
      numComponents(init.numComponents),
      tbl(init.types.data(), init.types.size()),
      columnLookup(init.lookupInputs.data(), init.lookupInputs.size())
{
}

StateManager::QueryState::QueryState()
    : lock(),
      queryData(0, ICfg::maxQueryOffsets)
{}

void StateManager::makeQuery(const ComponentID *components,
                             uint32_t num_components,
                             QueryRef *query_ref)
{
    std::lock_guard lock(query_state_.lock);

    if (query_ref->numMatchingArchetypes != ~0u) {
        return;
    }

    // FIXME: should reclaim dropped queries in this function by 
    // recording query_ref and finding unused ranges (numReferences == 0)

    // Enough tmp space for 1 archetype with the maximum number of components
    StackArray<uint32_t, 1 + max_archetype_components_> tmp_query_indices;

    auto saveTmpIndices = [&](uint32_t cur_offset) {
        assert(query_state_.queryData.size() + tmp_query_indices.size() <
               ICfg::maxQueryOffsets);

        query_state_.queryData.resize(cur_offset + tmp_query_indices.size(), [](auto) {});
        memcpy(&query_state_.queryData[cur_offset], tmp_query_indices.data(),
               sizeof(uint32_t) * tmp_query_indices.size());
    };

    const uint32_t query_offset = query_state_.queryData.size();
    uint32_t cur_offset = query_offset;

    uint32_t matching_archetypes = 0;
    for (int archetype_idx = 0, num_archetypes = archetype_stores_.size();
         archetype_idx < num_archetypes; archetype_idx++) {
        if (!archetype_stores_[archetype_idx].has_value()) {
            continue;
        }

        auto &archetype = *archetype_stores_[archetype_idx];

        bool has_components = true;
        for (int component_idx = 0; component_idx < (int)num_components; 
             component_idx++) {
            ComponentID component = components[component_idx];
            if (component.id == componentID<Entity>().id) {
                continue;
            }

            if (!archetype.columnLookup.exists(component.id)) {
                has_components = false;
                break;
            }
        }

        if (!has_components) {
            continue;
        }

        if (tmp_query_indices.size() + 1 + num_components >
                tmp_query_indices.capacity()) {
            assert(tmp_query_indices.size() > 0);

            saveTmpIndices(cur_offset);
            cur_offset += tmp_query_indices.size();
            tmp_query_indices.clear();
        }

        matching_archetypes += 1;
        tmp_query_indices.push_back(uint32_t(archetype_idx));

        int component_idx;
        for (component_idx = 0; component_idx < (int)num_components;
             component_idx++) {
            ComponentID component = components[component_idx];
            if (component.id == componentID<Entity>().id) {
                tmp_query_indices.push_back(0);
            } else {
                tmp_query_indices.push_back(archetype.columnLookup[component.id]);
            }
        }
    }

    saveTmpIndices(cur_offset);

    query_ref->offset = query_offset;
    query_ref->numMatchingArchetypes = matching_archetypes;
}

void StateManager::registerComponent(uint32_t id,
                                     uint32_t alignment,
                                     uint32_t num_bytes)
{
    // IDs are globally assigned, technically there is an edge case where
    // there are gaps in the IDs assigned to a specific StateManager
    if (id >= component_infos_.size()) {
        component_infos_.resize(id + 1, [](auto ptr) {
            Optional<TypeInfo>::noneAt(ptr);
        });
    }

    component_infos_[id].emplace(TypeInfo {
        .alignment = alignment,
        .numBytes = num_bytes,
    });
}

void StateManager::registerArchetype(uint32_t id, Span<ComponentID> components)
{
    uint32_t offset = archetype_components_.size();
    uint32_t num_user_components = components.size();

    uint32_t num_total_components = num_user_components + 1;
#ifdef MADRONA_MW_MODE
    num_total_components += 1;
#endif

    std::array<TypeInfo, max_archetype_components_> type_infos;
    std::array<IntegerMapPair, max_archetype_components_> lookup_input;

    TypeInfo *type_ptr = type_infos.data();

    // Add entity column as first column of every table
    *type_ptr = *component_infos_[0];
    type_ptr++;

#ifdef MADRONA_MW_MODE
    *type_ptr = *component_infos_[componentID<WorldID>().id];
    type_ptr++;
#endif

    for (int i = 0; i < (int)num_user_components; i++) {
        ComponentID component_id = components[i];

        archetype_components_.push_back(component_id);
        type_ptr[i] = *component_infos_[component_id.id];

        lookup_input[i] = IntegerMapPair {
            .key = component_id.id,
            .value = (uint32_t)i + user_component_offset_,
        };
    }

    // IDs are globally assigned, technically there is an edge case where
    // there are gaps in the IDs assigned to a specific StateManager
    if (archetype_stores_.size() <= id) {
        archetype_stores_.resize(id + 1, [](auto ptr) {
            Optional<ArchetypeStore>::noneAt(ptr);
        });
    }

    archetype_stores_[id].emplace(ArchetypeStore::Init {
        offset,
        components.size(),
        id,
        Span(type_infos.data(), num_total_components),
        Span(lookup_input.data(), num_user_components),
    });
}

void StateManager::clear(StateCache &cache, uint32_t archetype_id)
{
    auto &archetype = *archetype_stores_[archetype_id];

    // Free all IDs before deleting the table
    Entity *entities = (Entity *)archetype.tbl.data(0);
    uint32_t num_entities = archetype.tbl.numRows();
    entity_store_.bulkFree(cache.entity_cache_, entities, num_entities);

    archetype.tbl.clear();
}

StateManager::QueryState StateManager::query_state_ = StateManager::QueryState();

uint32_t StateManager::next_component_id_ = 0;
uint32_t StateManager::next_archetype_id_ = 0;

}
