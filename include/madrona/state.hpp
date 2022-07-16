#pragma once

#include <madrona/ecs.hpp>

namespace madrona {

class IDManager {
public:
    template <typename T>
    static Entity registerType();

    static uint32_t numTypes();

private:
    IDManager() = delete;
    ~IDManager() = delete;

    static uint32_t assignID(const char *identifier);
};

class StateManager {
public:
    StateManager(uint32_t max_components);

    template <typename ComponentT>
    Entity registerComponent();

    template <typename ComponentT>
    Entity componentID();

    template <typename ArchetypeT>
    void registerArchetype();

private:
};

}

#include "state.inl"
