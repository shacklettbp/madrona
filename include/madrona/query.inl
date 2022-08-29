#pragma once

#include <madrona/type_tracker.hpp>

namespace madrona {

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
ResultRef<ComponentT> ComponentRef<ComponentT>::operator[](Entity e) const
{ 
    auto loc = tbl_->getLoc(e);

    if (!loc.valid()) {
        return ResultRef<ComponentT>(nullptr);
    }

    return ResultRef<ComponentT>((ComponentT *)tbl_->getValue(col_idx_, loc));
}

template <typename ComponentT>
ComponentT & ComponentRef<ComponentT>::operator[](Loc idx) const
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
ArchetypeRef<ArchetypeT>::ArchetypeRef(Table *tbl)
    : tbl_(tbl)
{}

template <typename ArchetypeT>
Loc ArchetypeRef<ArchetypeT>::getLoc(Entity e) const
{
    return tbl_->getLoc(e);
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
ResultRef<ComponentT> ArchetypeRef<ArchetypeT>::get(Entity e)
{
    return component<ComponentT>()[e];
}

template <typename ArchetypeT>
template <typename ComponentT>
ResultRef<const ComponentT> ArchetypeRef<ArchetypeT>::get(Entity e) const
{
    return component<ComponentT>()[e];
}

template <typename ArchetypeT>
template <typename ComponentT>
ComponentT & ArchetypeRef<ArchetypeT>::get(Loc idx)
{
    return component<ComponentT>()[idx];
}

template <typename ArchetypeT>
template <typename ComponentT>
const ComponentT & ArchetypeRef<ArchetypeT>::get(Loc idx) const
{
    return component<ComponentT>()[idx];
}

template <typename ArchetypeT>
uint32_t ArchetypeRef<ArchetypeT>::size() const
{
    return tbl_->numRows();
}

template <typename ArchetypeT>
Loc ArchetypeRef<ArchetypeT>::Iter::operator*()
{
    return loc;
}

template <typename ArchetypeT>
auto ArchetypeRef<ArchetypeT>::Iter::operator++() -> Iter &
{
    ++loc.idx;
    return *this;
}

template <typename ArchetypeT>
bool ArchetypeRef<ArchetypeT>::Iter::operator==(Iter o)
{
    return loc == o.loc;
}

template <typename ArchetypeT>
bool ArchetypeRef<ArchetypeT>::Iter::operator!=(Iter o)
{
    return !(*this == o);
}

template <typename ArchetypeT>
auto ArchetypeRef<ArchetypeT>::begin() const -> Iter
{
    return Iter {
        Loc {
            0,
        },
    };
}

template <typename ArchetypeT>
auto ArchetypeRef<ArchetypeT>::end() const -> Iter
{
    return Iter {
        Loc {
            tbl_->numRows(),
        },
    };
}

template <typename ArchetypeT>
template <typename ComponentT>
uint32_t ArchetypeRef<ArchetypeT>::getComponentIndex() const
{
    return TypeTracker::typeID<ComponentLookup<ComponentT>>();
}

}
