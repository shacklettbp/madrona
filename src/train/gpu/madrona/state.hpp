#pragma once

#include <atomic>

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
};

}

#include "state.inl"
