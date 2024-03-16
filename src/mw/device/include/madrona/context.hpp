/*
 * Copyright 2021-2023 Brennan Shacklett and contributors
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 */
#pragma once

#include <madrona/state.hpp>
#include <madrona/registry.hpp>

#include "mw_gpu/worker_init.hpp"

namespace madrona {

class Context {
public:
    inline Context(WorldBase *world_data, const WorkerInit &init);

    template <typename ArchetypeT>
    Entity makeEntity();
    inline Entity makeEntity(uint32_t archetype_id);

    template <typename ArchetypeT>
    Loc makeTemporary();
    inline Loc makeTemporary(uint32_t archetype_id);

    inline void destroyEntity(Entity e);

    inline Loc loc(Entity e) const;

    template <typename ComponentT>
    ComponentT & get(Entity e);

    template <typename ComponentT>
    ComponentT & get(Loc loc);

    template <typename ComponentT>
    ResultRef<ComponentT> getSafe(Entity e);

    template <typename ComponentT>
    ResultRef<ComponentT> getCheck(Entity e);

    template <typename ComponentT>
    ResultRef<ComponentT> getCheck(Loc loc);

    template <typename ComponentT>
    ComponentT & getDirect(int32_t column_idx, Loc loc);

    template <typename SingletonT>
    SingletonT & singleton();

    inline void * tmpAlloc(uint64_t num_bytes);

    inline WorldID worldID() const { return world_id_; }

    inline WorldBase & data() const { return *data_; }
    
    template <typename... ComponentTs>
    inline Query<ComponentTs...> query();

    template <typename... ComponentTs, typename Fn>
    inline void iterateQuery(Query<ComponentTs...> &query, Fn &&fn);

protected:
    WorldBase *data_;

private:
    struct WaveInfo {
        uint32_t activeMask;
        uint32_t numActive;
        uint32_t coalescedIDX;

        inline bool isLeader() const
        {
            return coalescedIDX == 0;
        }
    };

    WorldID world_id_;
};

}

#include "context.inl"
