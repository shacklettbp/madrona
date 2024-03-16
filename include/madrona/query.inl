/*
 * Copyright 2021-2022 Brennan Shacklett and contributors
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 */
#pragma once

#include <madrona/type_tracker.hpp>

namespace madrona {

template <typename... ComponentTs>
Query<ComponentTs...>::Query()
    : initialized_(false)
{}

template <typename... ComponentTs>
Query<ComponentTs...>::Query(bool initialized)
    : initialized_(initialized)
{
    if (initialized) {
        ref_.numReferences.fetch_add_release(1);
    }
}

template <typename... ComponentTs>
Query<ComponentTs...>::Query(Query &&o)
    : initialized_(o.initialized_)
{
    o.initialized_ = false;
}

template <typename... ComponentTs>
Query<ComponentTs...>::~Query()
{
    if (initialized_) {
        ref_.numReferences.fetch_sub_release(1);
    }
}

template <typename... ComponentTs>
Query<ComponentTs...> & Query<ComponentTs...>::operator=(Query &&o)
{
    if (initialized_) {
        ref_.numReferences.fetch_sub_release(1);
    }

    initialized_ = o.initialized_;
    o.initialized_ = false;

    return *this;
}

template <typename... ComponentTs>
uint32_t Query<ComponentTs...>::numMatchingArchetypes() const
{
    return ref_.numMatchingArchetypes;
}

template <typename... ComponentTs>
QueryRef * Query<ComponentTs...>::getSharedRef() const
{
    return &ref_;
}

template <typename... ComponentTs>
QueryRef Query<ComponentTs...>::ref_ = QueryRef {
    0,
    0xFFFF'FFFF,
    0xFFFF'FFFF,
    0,
};

template <typename ComponentT>
ResultRef<ComponentT>::ResultRef(ComponentT *ptr)
    : ptr_(ptr)
{}

template <typename ComponentT>
ComponentT & ResultRef<ComponentT>::value()
{
    return *ptr_;
}

template <typename ComponentT>
bool ResultRef<ComponentT>::valid() const
{
    return ptr_ != nullptr;
}

}
