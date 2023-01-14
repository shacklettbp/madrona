/*
 * Copyright 2021-2022 Brennan Shacklett and contributors
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 */
#pragma once

#include <madrona/context.hpp>

namespace madrona {

template <typename ContextT, typename DataT>
class CustomContext : public Context {
public:
    inline CustomContext(DataT *world_data, const WorkerInit &worker_init);

    template <typename Fn, typename... Deps>
    inline JobID submit(Fn &&fn, bool is_child = true,
                        Deps && ... dependencies);

    template <typename Fn, typename... Deps>
    inline JobID submitN(Fn &&fn, uint32_t num_invocations,
                         bool is_child = true,
                         Deps && ... dependencies);

    template <typename... ComponentTs, typename Fn, typename... Deps>
    inline JobID parallelFor(const Query<ComponentTs...> &query, Fn &&fn,
                             bool is_child = true,
                             Deps && ... dependencies);

    inline DataT & data() const { return *static_cast<DataT *>(data_); }

private:
    using WorldDataT = DataT;

friend class JobManager;
friend class TaskGraph;
};

}

#include "custom_context.inl"
