/*
 * Copyright 2021-2023 Brennan Shacklett and contributors
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 */
#pragma once

#include <array>
#include <stdint.h>
#include <madrona/span.hpp>
#include <madrona/type_tracker.hpp>

namespace madrona {

enum ArchetypeFlagBits {
    ArchetypeNone = 0,
    ArchetypeImportOffsets = (1u << 0)
};

using ArchetypeFlags = uint32_t;

enum ComponentSelectFlagBits {
    ComponentSelectExportPointer = (1u << 0),
    ComponentSelectImportPointer = (1u << 1)
};

using ComponentSelectFlags = uint32_t;

struct ComponentSelectorGeneric {
    Span<const uint32_t> ids;
    Span<const ComponentSelectFlags> flags;
};

template <typename ...ComponentT>
struct ComponentSelector {
public:
    template <typename ...FlagT>
    ComponentSelector(FlagT ...in_flags)
        : ids{TypeTracker::typeID<ComponentT>()...},
          flags{in_flags...}
    {
    }

    ComponentSelector(ComponentSelectFlags in_flags)
        : ids{TypeTracker::typeID<ComponentT>()...}
    {
        // Set the flags to the same for all components
        for (int i = 0; i < flags.size(); ++i) {
            flags[i] = in_flags;
        }
    }

    ComponentSelectorGeneric makeGenericSelector()
    {
        return {
            .ids = Span(ids.data(), ids.size()),
            .flags = Span(flags.data(), flags.size())
        };
    }

    std::array<uint32_t, sizeof...(ComponentT)> ids;
    std::array<ComponentSelectFlags, sizeof...(ComponentT)> flags;
};

}
