#pragma once

#include <madrona/taskgraph_builder.hpp>

namespace madrona::phys {

// Attach this component to entities that you want to have obey physics.
struct PhysicalComponent {
    // This is going to be one of the DOF entities (i.e. DOFFreeBodyArchetype).
    Entity physicsEntity;
};



static constexpr uint32_t kMaxCoords = 6;

enum class DofType {
    FreeBody == 6,

    // When we add other types of physics DOF objects, we will encode
    // the number of degrees of freedom they all have here.
};

struct DofObjectPosition {
    float q[kMaxCoords];
};

struct DofObjectVelocity {
    float qv[kMaxCoords];
};

struct DofNumDofs {
    uint32_t numDofs;
};

struct DofObjectArchetype : public Archetype<
    DofObjectPosition,
    DofObjectVelocity,
    DofNumDofs
> {};

 
namespace PhysicsSystem {

void registerTypes(ECSRegistry &registry);
    
// For now, initial velocities are just going to be 0
void makeFreeBodyEntityPhysical(Context &ctx, Entity e,
                                Position position,
                                Rotation rotation);

void cleanupPhysicalEntity(Context &ctx, Entity e);

TaskGraphNodeID setupTasks(TaskGraphBuilder &builder,
                           Span<const TaskGraphNodeID> deps);

}

}
