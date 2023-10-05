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
    // Interesting, this is how you do argument forwarding of a general lambda.
    state_mgr_->iterateQuery(MADRONA_MW_COND(cur_world_id_,) query,
        [&](auto &...refs) {
            fn(ctx, refs...);
        });
}

}
