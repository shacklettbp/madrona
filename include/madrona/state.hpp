#pragma once

#include <madrona/ecs.hpp>

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

    template <typename ComponentT>
    Entity archetypeID();


private:
    template <typename T> struct TypeID;

    template <typename T>
    static Entity trackType(Entity *ptr);

    static Entity trackByName(Entity *ptr, const char *name);

    static void registerType(Entity *ptr);

};

}

#include "state.inl"
