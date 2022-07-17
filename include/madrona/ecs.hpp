#pragma once

#include <cstdint>

namespace madrona {

struct Entity {
    uint32_t id;

    static constexpr inline Entity none();
};

template <typename... Components>
struct Archetype {};

#if 0
template <typename... ColTypes>
struct Query {
    uint32_t size() const;

    template <typename ColType>
    ColType *col() const;

    template <typename ColType>
    ColType &get(Entity entity_id) const;
};

struct TableRef {
    template <typename T>
    T &get(Entity entity_id) const;

    template <typename... Args>
    Query<Args...> query();
};
#endif

inline bool operator==(Entity a, Entity b);
inline bool operator!=(Entity a, Entity b);

}

#include "ecs.inl"
