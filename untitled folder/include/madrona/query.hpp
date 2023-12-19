/*
 * Copyright 2021-2022 Brennan Shacklett and contributors
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 */
#pragma once

#include <madrona/table.hpp>
#include <madrona/optional.hpp>

#include <atomic>

namespace madrona {

struct QueryRef {
    AtomicU32 numReferences;
    uint32_t offset;
    uint32_t numMatchingArchetypes;
    uint32_t numComponents;
};

template <typename... ComponentTs>
class Query {
public:
    Query();
    Query(const Query &) = delete;
    Query(Query &&o);

    ~Query();

    Query & operator=(const Query &) = delete;
    Query & operator=(Query &&o);

    inline uint32_t numMatchingArchetypes() const;
    inline QueryRef * getSharedRef() const;

private:
    Query(bool initialized);
    bool initialized_;

    static QueryRef ref_;

friend class StateManager;
};

template <typename T>
class ResultRef {
public:
    inline ResultRef(T *ptr);

    inline T & value();
    inline bool valid() const;

private:
    T *ptr_;
};

template <typename ComponentT>
class ComponentRef {
public:
    ComponentRef(Table *tbl, uint32_t col_idx);

    inline ComponentT & operator[](uint32_t row) const;

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
    inline ArchetypeRef(Table *tbl);

    template <typename ComponentT>
    inline ComponentRef<ComponentT> component();

    template <typename ComponentT>
    inline ComponentRef<const ComponentT> component() const;

    template <typename ComponentT>
    inline ComponentT & get(uint32_t idx);
    template <typename ComponentT>
    inline const ComponentT & get(uint32_t idx) const;

    inline uint32_t size() const;

private:
    template <typename ComponentT> struct ComponentLookup {};

    template <typename ComponentT>
    inline uint32_t getComponentIndex() const;

    Table *tbl_;

friend class StateManager;
};

}

#include "query.inl"
