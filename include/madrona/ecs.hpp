#pragma once

#include <cstdint>

namespace madrona {

struct Entity {
    uint32_t id;

    bool operator!=(Entity o) const;

    static constexpr inline Entity none();
};

template <typename... Components>
struct Archetype {};

template <typename ComponentT>
struct Component {
private:
    static Entity _id;

friend class StateManager;
};

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
