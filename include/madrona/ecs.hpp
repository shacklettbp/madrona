#pragma once

#include <madrona/fwd.hpp>

#include <cstdint>

namespace madrona {

struct Entity {
    uint64_t gen : 18;
    uint64_t archetype : 16;
    uint32_t id : 30;

    static constexpr inline Entity none();
};

template <typename... Components>
struct Archetype {};

template <typename... ColTypes>
struct Query {
    uint32_t size() const;

    template <typename ColType>
    ColType &get(Entity entity) const;
};

#if 0
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
