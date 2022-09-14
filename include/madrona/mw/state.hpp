#pragma once

#include <madrona/state.hpp>

namespace madrona {
namespace mw {

struct WorldIndex : IndexHelper {};

struct WorldID {
    uint32_t id;
};

class StateManager : public madrona::StateManager {
public:
    StateManager(int num_worlds);

    template <typename ComponentT>
    ComponentID registerComponent();

    template <typename ArchetypeT>
    ArchetypeID registerArchetype();

private:
    DynArray<WorldIndex> world_indices_;
    utils::SpinLock register_lock_;
    uint32_t num_worlds_;
};

}
}

#include "state.inl"
