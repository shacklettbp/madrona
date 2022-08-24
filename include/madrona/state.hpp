#pragma once

#include <madrona/ecs.hpp>
#include <madrona/heap_array.hpp>
#include <madrona/dyn_array.hpp>
#include <madrona/span.hpp>
#include <madrona/table.hpp>
#include <madrona/optional.hpp>

namespace madrona {

class StateManager;

class ArchetypeID {
    ArchetypeID(uint32_t i) : id(i) {};
    uint32_t id;
friend class StateManager;
};

class ComponentID {
    ComponentID(uint32_t i) : id(i) {};
    uint32_t id;
friend class StateManager;
};

class StateManager {
public:
    StateManager();

    template <typename ComponentT>
    ComponentID registerComponent();

    template <typename ArchetypeT>
    ArchetypeID registerArchetype();

    template <typename ComponentT>
    ComponentID componentID() const;

    template <typename ArchetypeT>
    ArchetypeID archetypeID() const;

    template <typename ComponentT>
    inline ComponentT & getComponent(Entity entity);

    template <typename... ComponentTs>
    inline Query<ComponentTs...> query();

    template <typename ArchetypeT, typename... Args>
    inline Entity makeEntity(Args && ...args);

    inline void destroyEntity(Entity e);

private:
    struct ArchetypeInfo {
        uint32_t componentOffset;
        uint32_t numComponents;
        Table tbl;
    };

    template <typename T> struct TypeID;

    template <typename T>
    static uint32_t trackType(uint32_t *ptr);
    static uint32_t trackByName(uint32_t *ptr, const char *name);

    static void registerType(uint32_t *ptr, bool component);

    void saveComponentInfo(uint32_t id, uint32_t alignment,
                           uint32_t num_bytes);
    void saveArchetypeInfo(uint32_t id, Span<ComponentID> components);

    DynArray<TypeInfo> component_infos_;
    DynArray<ComponentID> archetype_components_;
    DynArray<Optional<ArchetypeInfo>> archetype_infos_;
};

}

#include "state.inl"
