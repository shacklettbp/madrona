#pragma once

#include <madrona/ecs.hpp>
#include <madrona/heap_array.hpp>
#include <madrona/dyn_array.hpp>
#include <madrona/span.hpp>

namespace madrona {

class StateManager {
public:
    StateManager(uint32_t max_types);

    template <typename ComponentT>
    Entity registerComponent();

    template <typename ArchetypeT>
    Entity registerArchetype();

    template <typename ComponentT>
    Entity componentID();

    template <typename ComponentT>
    Entity archetypeID();

private:
    union TypeInfo {
        struct {
            uint32_t alignment;
            uint32_t numBytes;
        };
        struct {
            uint32_t componentOffset;
            uint32_t numComponents;
        };
    };

    template <typename T> struct TypeID;

    template <typename T>
    static Entity trackType(Entity *ptr);
    static Entity trackByName(Entity *ptr, const char *name);

    static void registerType(Entity *ptr);

    void saveComponentInfo(Entity id, uint32_t alignment, uint32_t num_bytes);
    void saveArchetypeInfo(Entity id, Span<Entity> components);

    HeapArray<TypeInfo, InitAlloc> type_infos_;
    DynArray<Entity> archetype_components_;
};

}

#include "state.inl"
