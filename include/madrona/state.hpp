#pragma once

#include <madrona/ecs.hpp>

namespace madrona {

class StateManager {
public:
    StateManager(uint32_t max_components);

    template <typename T>
    Entity registerComponent();

    template <typename T>
    Entity componentID();

private:
    uint32_t num_components_;
};

}

#include "state.inl"
