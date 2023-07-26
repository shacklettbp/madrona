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

// Subclass this type with the CRTP:
// class MyContext : public CustomContext<MyContext, MyPerWorldState> {}
template <typename ContextT, typename DataT>
class CustomContext : public Context {
public:
    inline CustomContext(DataT *world_data, const WorkerInit &worker_init);

    inline DataT & data() const { return *static_cast<DataT *>(data_); }

private:
    using WorldDataT = DataT;

friend class JobManager;
friend class TaskGraph;
};

}

#include "custom_context.inl"
