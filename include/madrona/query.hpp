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

template <typename ArchetypeT>
class ArchetypeRef {
public:
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

    inline uint32_t size() const;

private:
    template <typename ComponentT> struct ComponentLookup {};

    inline ArchetypeRef(Table *tbl);

    template <typename ComponentT>
    inline uint32_t getComponentIndex() const;

    Table *tbl_;

friend class StateManager;
};

}

#include "query.inl"
