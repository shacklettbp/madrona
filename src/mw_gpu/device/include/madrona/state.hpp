/*
 * Copyright 2021-2022 Brennan Shacklett and contributors
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 */
#pragma once

#include <atomic>
#include <array>

#include <madrona/ecs.hpp>
#include <madrona/utils.hpp>

namespace madrona {

class StateManager {
public:
    StateManager(uint32_t max_components);

    template <typename ComponentT>
    Entity registerComponent();

    template <typename ArchetypeT>
    Entity registerArchetype();

    template <typename ComponentT>
    Entity componentID();

private:
    template <typename T> struct TypeID;

    template <typename T>
    static Entity registerType();

    static inline uint32_t num_components_ = 0;
    static inline utils::SpinLock register_lock_ {};

    static constexpr uint32_t max_archetype_components_ = 16384;

    std::array<uint32_t, max_archetype_components_> archetype_components_;
};

}

#include "state.inl"
