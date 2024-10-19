#include "cvphysics.hpp"

namespace madrona::phys {
    
namespace PhysicsSystem {

void registerTypes(ECSRegistry &registry)
{
    registry.registerComponent<PhysicalComponent>();

    registry.registerComponent<DofObjectPosition>();
    registry.registerComponent<DofObjectVelocity>();
    registry.registerComponent<DofNumDofs>();

    registry.registerArchetype<DofObjectArchetype>();
}

void makeFreeBodyEntityPhysical(Context &ctx, Entity e,
                                Position position,
                                Rotation rotation)
{
    Entity physical_entity = ctx.makeEntity<DofObjectArchetype>();

    auto &pos = ctx.get<DofObjectPosition>(physical_entity);

    pos.q[0] = position.x;
    pos.q[1] = position.y;
    pos.q[2] = position.z;

    Vector3 pyr = rotation.extractPYR();

    pos.q[3] = pyr.x;
    pos.q[4] = pyr.x;
    pos.q[5] = pyr.x;

    auto &vel = ctx.get<DofObjectVelocity>(physical_entity);

    vel.qv[0] = 0.f;
    vel.qv[1] = 0.f;
    vel.qv[2] = 0.f;
    vel.qv[3] = 0.f;
    vel.qv[4] = 0.f;
    vel.qv[5] = 0.f;

    ctx.get<DofNumDofs>(physical_entity).numDofs = 6;

    ctx.get<PhysicalComponent>(e) = {
        .physicsEntity = physical_entity,
    };
}

void cleanupPhysicalEntity(Context &ctx, Entity e)
{
    PhysicalComponent physical_comp = ctx.get<PhysicalComponent>(e);
    ctx.destroyEntity(physical_comp.physicsEntity);
}

TaskGraphNodeID setupTasks(TaskGraphBuilder &builder,
                           Span<const TaskGraphNodeID> deps)
{
    
}
    
}

}
