#pragma once

#include <madrona/ecs.hpp>
#include <madrona/heap_array.hpp>
#include <madrona/dyn_array.hpp>
#include <madrona/span.hpp>
#include <madrona/table.hpp>
#include <madrona/query.hpp>
#include <madrona/optional.hpp>
#include <madrona/type_tracker.hpp>

namespace madrona {

struct ArchetypeID {
    uint32_t id;

private:
    ArchetypeID(uint32_t i) : id(i) {};
friend class StateManager;
};

struct ComponentID {
    uint32_t id;

private:
    ComponentID(uint32_t i) : id(i) {};
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
    inline ResultRef<ComponentT> get(Entity entity);

    template <typename... ComponentTs>
    inline Query<ComponentTs...> query();

    template <typename ArchetypeT>
    inline ArchetypeRef<ArchetypeT> archetype();

    template <typename ArchetypeT, typename... Args>
    inline Entity makeEntity(Args && ...args);

    inline void destroyEntity(Entity e);

private:
    struct ArchetypeInfo {
        uint32_t componentOffset;
        uint32_t numComponents;
        Table tbl;
    };

    void saveComponentInfo(uint32_t id, uint32_t alignment,
                           uint32_t num_bytes);
    void saveArchetypeInfo(uint32_t id, Span<ComponentID> components);

    DynArray<TypeInfo> component_infos_;
    DynArray<ComponentID> archetype_components_;
    DynArray<Optional<ArchetypeInfo>> archetype_infos_;

    static uint32_t next_component_id_;
    static uint32_t next_archetype_id_;
};

}

#include "state.inl"
