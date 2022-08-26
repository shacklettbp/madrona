#pragma once

#include <madrona/type_tracker.hpp>

namespace madrona {

template <typename ComponentT>
ComponentRef<ComponentT>::Result::Result(ComponentT *ptr)
    : ptr_(ptr)
{}

template <typename ComponentT>
ComponentT & ComponentRef<ComponentT>::Result::value()
{
    return *ptr_;
}

template <typename ComponentT>
bool ComponentRef<ComponentT>::Result::valid() const
{
    return ptr_ != nullptr;
}

template <typename ComponentT>
ComponentRef<ComponentT>::ComponentRef(Table *tbl, uint32_t col_idx)
    : tbl_(tbl),
      col_idx_(col_idx)
{}

template <typename ComponentT>
typename ComponentRef<ComponentT>::Result
    ComponentRef<ComponentT>::operator[](Entity e) const
{ 
    auto idx = tbl_->getIndex(e);

    if (!idx.valid()) {
        return Result(nullptr);
    }

    return Result((ComponentT *)tbl_->getValue(col_idx_, idx));
}

template <typename ComponentT>
ComponentT & ComponentRef<ComponentT>::operator[](Table::Index idx) const
{ 
    return *(ComponentT *)tbl_->getValue(col_idx_, idx);
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
Table::Index ArchetypeRef<ArchetypeT>::getIndex(Entity e) const
{
    return tbl_->getIndex(e);
}

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
typename ArchetypeRef<ArchetypeT>::template Result<ComponentT>
    ArchetypeRef<ArchetypeT>::get(Entity e)
{
    return component<ComponentT>()[e];
}

template <typename ArchetypeT>
template <typename ComponentT>
typename ArchetypeRef<ArchetypeT>::template Result<const ComponentT>
    ArchetypeRef<ArchetypeT>::get(Entity e) const
{
    return component<ComponentT>()[e];
}

template <typename ArchetypeT>
template <typename ComponentT>
ComponentT & ArchetypeRef<ArchetypeT>::get(Table::Index idx)
{
    return component<ComponentT>()[idx];
}

template <typename ArchetypeT>
template <typename ComponentT>
const ComponentT & ArchetypeRef<ArchetypeT>::get(Table::Index idx) const
{
    return component<ComponentT>()[idx];
}

template <typename ArchetypeT>
ArchetypeRef<ArchetypeT>::ArchetypeRef(Table *tbl)
    : tbl_(tbl)
{}

template <typename ArchetypeT>
template <typename ComponentT>
uint32_t ArchetypeRef<ArchetypeT>::getComponentIndex() const
{
    return TypeTracker::typeID<ComponentLookup<ComponentT>>();
}

}
