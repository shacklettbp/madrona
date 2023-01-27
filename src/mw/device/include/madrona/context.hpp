/*
 * Copyright 2021-2022 Brennan Shacklett and contributors
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 */
#pragma once

#include <madrona/state.hpp>

#include "mw_gpu/worker_init.hpp"

namespace madrona {

struct JobID {}; // FIXME

class Context {
public:
    inline Context(WorldBase *world_data, const WorkerInit &init);

    template <typename ArchetypeT>
    Entity makeEntityNow();

    template <typename ArchetypeT>
    Loc makeTemporary();

    inline void destroyEntityNow(Entity e);

    inline Loc getLoc(Entity e) const;

    template <typename ComponentT>
    ComponentT & getUnsafe(Entity e);

    template <typename ComponentT>
    ComponentT & getUnsafe(Loc loc);

    template <typename ComponentT>
    ResultRef<ComponentT> get(Entity e);

    template <typename ComponentT>
    ResultRef<ComponentT> get(Loc loc);

    template <typename ComponentT>
    ComponentT & getDirect(int32_t column_idx, Loc loc);

    template <typename SingletonT>
    SingletonT & getSingleton();

#if 0
    template <typename Fn, typename... DepTs>
    inline JobID submit(Fn &&fn, bool is_child = true,
                        DepTs && ...dependencies);

    template <typename Fn, typename... DepTs>
    inline JobID submitN(Fn &&fn, uint32_t num_invocations,
        bool is_child = true, DepTs && ...dependencies);

    template <typename... ColTypes, typename Fn, typename... DepTs>
    inline JobID parallelFor(const Query<ColTypes...> &query, Fn &&fn,
        bool is_child = true, DepTs && ... dependencies);

    void markJobFinished();

    inline JobID currentJobID() const { return job_id_; }
#endif

    inline void * tmpAlloc(uint64_t num_bytes);

    inline WorldID worldID() const { return world_id_; }

    inline WorldBase & data() const { return *data_; }

protected:
#if 0
    template <typename ContextT, typename Fn, typename... DepTs>
    inline JobID submitImpl(Fn &&fn, bool is_child,
                            DepTs && ... dependencies);

    template <typename ContextT, typename Fn, typename... DepTs>
    inline JobID submitNImpl(Fn &&fn, uint32_t num_invocations, bool is_child,
                             DepTs && ... dependencies);

    template <typename ContextT, typename... ComponentTs,
              typename Fn, typename... DepTs>
    inline JobID parallelForImpl(const Query<ComponentTs...> &query, Fn &&fn,
                                 bool is_child, DepTs && ... dependencies);
#endif

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
#if 0
    inline StateManager & state();

    WaveInfo computeWaveInfo();

    JobID waveSetupNewJob(uint32_t func_id, bool link_parent,
                          uint32_t num_invocations, uint32_t bytes_per_job,
                          void **thread_data_store);

    JobContainerBase * allocJob(uint32_t bytes_per_job, WaveInfo wave_info);

    inline void stageChildJob(uint32_t func_id, uint32_t num_combined_jobs,
                              uint32_t bytes_per_job, void *containers);

    JobID job_id_;

    uint32_t lane_id_;
#endif
};

}

#include "context.inl"
