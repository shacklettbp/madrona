#pragma once

#include <cstdint>

namespace madrona {

struct Entity {
    uint64_t gen : 18;
    uint64_t archetype : 16;
    uint64_t id : 30;

    static constexpr inline Entity none();
};

struct Loc {
    uint32_t idx;

    inline bool valid() const;
};

struct IndexHelper {
    uint32_t prev;
    uint32_t next;
};

template <typename... ComponentTs> struct Archetype {
    using Base = Archetype<ComponentTs...>;
};

inline bool operator==(Entity a, Entity b);
inline bool operator!=(Entity a, Entity b);

inline bool operator==(Loc a, Loc b);
inline bool operator!=(Loc a, Loc b);

}

#include "ecs.inl"
