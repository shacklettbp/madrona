#pragma once

#include <madrona/table.hpp>
#include <madrona/optional.hpp>

namespace madrona {

template <typename... ComponentTs>
class Query {
public:
    template <typename Fn>
    void forAll(Fn &&fn);

private:
    static constexpr uint32_t num_components_ =
        sizeof...(ComponentTs);
};

template <typename ComponentT>
class ComponentRef {
public:
    class Result {
    public:
        inline Result(ComponentT *ptr);
    
        inline ComponentT & value();
        inline bool valid() const;
    
    private:
        ComponentT *ptr_;
    };

    ComponentRef(Table *tbl, uint32_t col_idx);

    inline Result operator[](Entity e) const;

    inline ComponentT & operator[](Table::Index idx) const;

    inline ComponentT * data() const;
    inline uint32_t size() const;

    inline ComponentT * begin() const;
    inline ComponentT * end() const;

private:
    Table *tbl_;
    uint32_t col_idx_;
};

// Issue: need to map from (ArchetypeT, ComponentT) => table column index
// For compile time known type pairs, this isn't super hard because you can use a ID lookup struct like ComponentLookup above. This is a bit annoying though as you need to leverage a complex ID (by string) / pointer tracking system to ensure consistency across library boundaries. 
// Additionally, this doesn't help with query generation: given some ComponentT find all archetypes that contain ComponentT *AND* find the column index corresponding to ComponentT. In this situation, you can't recover ArchetypeT when searching over the list of archetypes saved at runtime.
// The obvious solution here is to build a lookup table (hash map) for each archetype that maps from ComponentID to column index. The downside is that this winds up imposing a unavoidable non-trivial cost (hash + indirection) to cases where the (ArchetypeT, ComponentT) is compile-time known

template <typename ArchetypeT>
class ArchetypeRef {
public:
    inline ArchetypeRef(Table *tbl);

    inline Table::Index getIndex(Entity e) const;

    template <typename ComponentT>
    inline ComponentRef<ComponentT> component();

    template <typename ComponentT>
    inline ComponentRef<const ComponentT> component() const;

    template <typename ComponentT>
    using Result = typename ComponentRef<ComponentT>::Result;

    template <typename ComponentT>
    inline Result<ComponentT> get(Entity e);
    template <typename ComponentT>
    inline Result<const ComponentT> get(Entity e) const;

    template <typename ComponentT>
    inline ComponentT & get(Table::Index idx);
    template <typename ComponentT>
    inline const ComponentT & get(Table::Index idx) const;

private:
    template <typename ComponentT>
    struct ComponentLookup;

    template <typename ComponentT>
    inline uint32_t getComponentIndex() const;

    Table *tbl_;
};

}

#include "query.inl"
