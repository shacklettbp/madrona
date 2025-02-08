#pragma once

namespace madrona {

template <typename ArchetypeT>
void TaskGraph::clearTemporaries()
{
    state_mgr_->clear<ArchetypeT>(MADRONA_MW_COND(cur_world_id_,)
                                  *state_cache_, true);
}

template <typename ContextT, typename Fn, typename ...ComponentTs>
void TaskGraph::iterateQuery(ContextT &ctx,
                             Query<ComponentTs...> &query,
                             Fn &&fn)
{
    state_mgr_->iterateQuery(MADRONA_MW_COND(cur_world_id_,) query,
        [&](auto &...refs) {
            fn(ctx, refs...);
        });
}

StateManager & TaskGraph::stateManager() const
{
  return *state_mgr_;
}

}
