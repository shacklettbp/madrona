/*
 * Copyright 2021-2023 Brennan Shacklett and contributors
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 */
#pragma once

#include <madrona/fwd.hpp>
#include <madrona/ecs.hpp>
#include <madrona/state.hpp>
#include <madrona/registry.hpp>

namespace madrona {

// The Context object is the main entry point to the ECS, with functionality
// for creating & destroying entities and getting component data.
//
// Applications should subclass Context by inheriting from
// madrona::CustomContext (include/madrona/custom_context.hpp)
// which will ensure Context::data() returns the right per world type
// and allows for adding custom helpers to the class.
class Context {
public:
    Context(WorldBase *world_data, const WorkerInit &init);
    Context(const Context &) = delete;

    // Create an Entity of archetype ArchetypeT. Entity is a 64bit
    // identifier that can be used to access the entity's components
    // or later delete it.
    template <typename ArchetypeT>
    inline Entity makeEntity();
    inline Entity makeEntity(uint32_t archetype_id);

    // Create a Temporary of archetype ArchetypeT. Temporaries are
    // entities without a persistent Entity identifier, instead represented
    // by the raw row and table ID (Loc) of the entity. Temporaries can
    // be iterated over by ECS systems, but the Loc returned is only
    // guaranteed to be valid within the ECS system that it was created in.
    // Temporaries are deleted in bulk with the ClearTmpNode
    // in the taskgraph.
    // Example use case: per-step temporaries like candidate collisions.
    template <typename ArchetypeT>
    inline Loc makeTemporary();
    inline Loc makeTemporary(uint32_t archetype_id);

    // Destroy Entity e
    inline void destroyEntity(Entity e);

    // Get the Loc (row and table ID) of Entity e. This can be used to
    // fetch components more efficiently than by entity ID. Loc generally
    // only is valid within a single ECS system or when no entities of the
    // fetched archetype are being created / deleted.
    inline Loc loc(Entity e) const;

    // Returns a reference to ComponentT of Entity e.
    // Note that this function performs no error checking! Bad things happen
    // if e does not have ComponentT!
    template <typename ComponentT>
    inline ComponentT & get(Entity e);

    // Returns a reference for the entity located at Loc l.
    // Again, no error checking.
    template <typename ComponentT>
    inline ComponentT & get(Loc l);

    // The safe version of Context::get. Use ResultRef::valid()
    // to check if the entity had the component and ResultRef::value()
    // to get a reference to the component if so.
    template <typename ComponentT>
    inline ResultRef<ComponentT> getSafe(Entity e);

    // Equivalent to Context::getSafe for entities. Note that the
    // Loc overload below is not safe.
    template <typename ComponentT>
    inline ResultRef<ComponentT> getCheck(Entity e);

    // Checks if the entity at Loc l has ComponentT and returns a reference
    // to the component ata if so. Use ResultRef::valid() to check if the
    // entity has the component and ResultRef::value() to get the reference.
    // NOTE: This function is not safe, because Loc's can be invalidated
    // by creating / deleting entities of the same archetype. This function
    // does not / can not check if the Loc is still valid.
    template <typename ComponentT>
    inline ResultRef<ComponentT> getCheck(Loc l);

    // Directly get an entity's component by column ID and Loc.
    // This is fast but not recommended. Will likely be removed in future.
    template <typename ComponentT>
    ComponentT & getDirect(int32_t column_idx, Loc loc);

    // Get a reference to the singleton component SingletonT. Note that
    // singleton components are immediately created on the call to
    // ECSRegistry::registerSingleton, so you don't have to create them.
    template <typename SingletonT>
    SingletonT & singleton();

    // Allocate a raw chunk of memory num_bytes in length from a global
    // bump allocator. Use ResetTmpAllocNode in the taskgraph to reclaim
    // memory.
    inline void * tmpAlloc(uint64_t num_bytes);

    // Create an ECS query matching the template components.
    // Pass to iterateQuery to iterate over all entities with the
    // template set of components.
    template <typename... ComponentTs>
    inline Query<ComponentTs...> query();

    // Iterate a query over all entities with the template components,
    // executing the lambda fn for each one.
    template <typename Fn, typename... ComponentTs>
    inline void iterateQuery(const Query<ComponentTs...> &query, Fn &&fn);

#ifdef MADRONA_MW_MODE
    // Get the current world's ID: [0, numWorlds - 1]
    inline WorldID worldID() const;
#endif

    // Reference to per world data. Your Context subclass automatically
    // overrides this with a reference of the correct type if inherited
    // from CustomContext.
    inline WorldBase & data() { return *data_; }

    // StateManager provides lower-level control of internal data
    // structures underlying the ECS. Can get direct pointers to
    // table columns, etc...
    inline StateManager * getStateManager() { return state_mgr_; }

protected:
    WorldBase *data_;

private:
    StateManager * const state_mgr_;
    StateCache * const state_cache_;
#ifdef MADRONA_MW_MODE
    uint32_t cur_world_id_;
#endif
};

}

#include "context.inl"
