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

template <typename ComponentT>
ComponentRef<ComponentT>::ComponentRef(Table *tbl, uint32_t col_idx)
    : tbl_(tbl),
      col_idx_(col_idx)
{}

template <typename ComponentT>
ComponentT & ComponentRef<ComponentT>::operator[](uint32_t row) const
{ 
    return data()[row];
}

template <typename ComponentT>
ComponentT * ComponentRef<ComponentT>::data() const
{
    return (ComponentT *)tbl_->data(col_idx_);
}

template <typename ComponentT>
uint32_t ComponentRef<ComponentT>::size() const
{
    return tbl_->numRows();
}

template <typename ComponentT>
ComponentT * ComponentRef<ComponentT>::begin() const { return data(); }

template <typename ComponentT>
ComponentT * ComponentRef<ComponentT>::end() const
{
    return data() + size();
}

template <typename ArchetypeT>
ArchetypeRef<ArchetypeT>::ArchetypeRef(Table *tbl)
    : tbl_(tbl)
{}

template <typename ArchetypeT>
template <typename ComponentT>
ComponentRef<ComponentT> ArchetypeRef<ArchetypeT>::component()
{
    return ComponentRef<ComponentT>(tbl_, getComponentIndex<ComponentT>());
}

template <typename ArchetypeT>
template <typename ComponentT>
ComponentRef<const ComponentT> ArchetypeRef<ArchetypeT>::component() const
{
    return ComponentRef<const ComponentT>(tbl_, getComponentIndex<ComponentT>());
}

template <typename ArchetypeT>
template <typename ComponentT>
ComponentT & ArchetypeRef<ArchetypeT>::get(uint32_t idx)
{
    return component<ComponentT>()[idx];
}

template <typename ArchetypeT>
template <typename ComponentT>
const ComponentT & ArchetypeRef<ArchetypeT>::get(uint32_t idx) const
{
    return component<ComponentT>()[idx];
}

template <typename ArchetypeT>
uint32_t ArchetypeRef<ArchetypeT>::size() const
{
    return tbl_->numRows();
}

template <typename ArchetypeT>
template <typename ComponentT>
uint32_t ArchetypeRef<ArchetypeT>::getComponentIndex() const
{
    if constexpr (std::is_same_v<ComponentT, Entity>) {
        return 0;
    } else {
        return TypeTracker::typeID<ComponentLookup<ComponentT>>();
    }
}

}
