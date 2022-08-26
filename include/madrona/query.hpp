#pragma once

#include <madrona/table.hpp>
#include <madrona/optional.hpp>

namespace madrona {

template <typename T>
class ResultRef {
public:
    inline ResultRef(T *ptr);

    inline T & value();
    inline bool valid() const;

private:
    T *ptr_;
};

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
    ComponentRef(Table *tbl, uint32_t col_idx);

    inline ResultRef<ComponentT> operator[](Entity e) const;

    inline ComponentT & operator[](Loc idx) const;

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
    inline Loc getLoc(Entity e) const;

    template <typename ComponentT>
    inline ComponentRef<ComponentT> component();

    template <typename ComponentT>
    inline ComponentRef<const ComponentT> component() const;

    template <typename ComponentT>
    inline ResultRef<ComponentT> get(Entity e);
    template <typename ComponentT>
    inline ResultRef<const ComponentT> get(Entity e) const;

    template <typename ComponentT>
    inline ComponentT & get(Loc idx);
    template <typename ComponentT>
    inline const ComponentT & get(Loc idx) const;

    inline uint32_t size() const;

    struct Iter {
        Loc loc;

        inline Loc operator*();
        inline Iter & operator++();
        inline bool operator==(Iter o);
        inline bool operator!=(Iter o);
    };

    inline Iter begin() const;
    inline Iter end() const;

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
