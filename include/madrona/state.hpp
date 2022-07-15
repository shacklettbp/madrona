#pragma once

#include <madrona/ecs.hpp>

#include <type_traits>

namespace madrona {

class StateManager {
public:
    StateManager(uint32_t max_components);

    template <typename T>
    Entity registerComponent();

    template <typename T>
    Entity componentID();

    template <typename T>
    void registerArchetype();

private:
    uint32_t num_components_;
};

}

#include "state.inl"
