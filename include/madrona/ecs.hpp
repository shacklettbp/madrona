#pragma once

#include <cstdint>

namespace madrona {

struct Entity {
    uint32_t gen;
    uint32_t id;

    static constexpr inline Entity none();
};

struct Loc {
    uint32_t archetype;
    uint32_t row;

    inline bool valid() const;

    static inline Loc none();
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
