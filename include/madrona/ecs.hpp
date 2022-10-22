/*
 * Copyright 2021-2022 Brennan Shacklett and contributors
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 */
#pragma once

#include <madrona/fwd.hpp>

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

// Base class that per-world user data must inherit from
// In the future may include any per-world data that the engine
// itself needs. For now, just provides a common base that Context
// stores a pointer to.
class WorldBase {
public:
    inline WorldBase(Context &) {}
    WorldBase(const WorldBase &) = delete;
};

class SystemBase {
};

template <typename SystemT>
class System : public SystemBase {
    inline void operator()(Context &ctx, uint32_t invocation_idx);
};

template <typename... ComponentTs>
class ParallelForSystem : public SystemBase {
public:
    void run(Context &ctx, uint32_t invocation_idx);
private:
};

inline bool operator==(Entity a, Entity b);
inline bool operator!=(Entity a, Entity b);

inline bool operator==(Loc a, Loc b);
inline bool operator!=(Loc a, Loc b);

}

#include "ecs.inl"
