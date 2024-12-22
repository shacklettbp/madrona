/*
 * Copyright 2021-2023 Brennan Shacklett and contributors
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 */
#pragma once

#include <stdint.h>
#include <madrona/ecs.hpp>
#include <madrona/span.hpp>
#include <madrona/type_tracker.hpp>

namespace madrona {

enum class ArchetypeFlags : uint32_t {
    None = 0,
    ImportOffsets = 1_u32 << 0,
};

enum class ComponentFlags : uint32_t {
    None = 0,
    ExportMemory = 1_u32 << 0,
    ImportMemory = 1_u32 << 1,
    CudaReserveMemory = 1_u32 << 2,
    CudaAllocMemory = 1_u32 << 3,
};

template <typename... ComponentTs>
struct ComponentMetadataSelector {
    std::array<ComponentFlags, sizeof...(ComponentTs)> flags;

    inline ComponentMetadataSelector(ComponentFlags component_flags);

    template <typename... FlagTs>
    inline ComponentMetadataSelector(FlagTs ...in_flags);
};

inline ArchetypeFlags & operator|=(ArchetypeFlags &a, ArchetypeFlags b);
inline ArchetypeFlags operator|(ArchetypeFlags a, ArchetypeFlags b);
inline ArchetypeFlags & operator&=(ArchetypeFlags &a, ArchetypeFlags b);
inline ArchetypeFlags operator&(ArchetypeFlags a, ArchetypeFlags b);

inline ComponentFlags & operator|=(ComponentFlags &a, ComponentFlags b);
inline ComponentFlags operator|(ComponentFlags a, ComponentFlags b);
inline ComponentFlags & operator&=(ComponentFlags &a, ComponentFlags b);
inline ComponentFlags operator&(ComponentFlags a, ComponentFlags b);

}

#include "ecs_flags.inl"
