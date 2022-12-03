/*
 * Copyright 2021-2022 Brennan Shacklett and contributors
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 */
#include "simple.hpp"

#include <madrona/render.hpp>

#include <cinttypes>

using namespace madrona;
using namespace madrona::base;
using namespace madrona::math;
using namespace madrona::phys;

namespace SimpleTaskgraph {

inline void clampSystem(Engine &ctx,
                        Position &position)
{
    // Clamp to world bounds
    position.x = std::clamp(position.x,
                               ctx.data().worldBounds.pMin.x,
                               ctx.data().worldBounds.pMax.x);
    position.y = std::clamp(position.y,
                               ctx.data().worldBounds.pMin.y,
                               ctx.data().worldBounds.pMax.y);
    position.z = std::clamp(position.z,
                               ctx.data().worldBounds.pMin.z,
                               ctx.data().worldBounds.pMax.z);
}

void SimpleSim::registerTypes(ECSRegistry &registry)
{
    base::registerTypes(registry);
    RigidBodyPhysicsSystem::registerTypes(registry);
    render::RenderingSystem::registerTypes(registry);

    registry.registerArchetype<Sphere>();
    registry.registerArchetype<Agent>();
}

void SimpleSim::setupTasks(TaskGraph::Builder &builder)
{
    auto clamp_sys =
        builder.parallelForNode<Engine, clampSystem, Position>({});

    //auto phys_sys = RigidBodyPhysicsSystem::setupTasks(builder, {clamp_sys});

    //auto phys_cleanup_sys = RigidBodyPhysicsSystem::setupCleanupTasks(builder,
    //    {phys_sys});

    auto renderer_sys = render::RenderingSystem::setupTasks(builder,
        {clamp_sys});

    (void)renderer_sys;

    printf("Setup done\n");
}

SimpleSim::SimpleSim(Engine &ctx, const EnvInit &env_init)
    : WorldBase(ctx),
      worldBounds(AABB::invalid()),
      spheres((Entity *)malloc(sizeof(Entity) * env_init.numObjs)),
      numSpheres(env_init.numObjs),
      agent()
{
    worldBounds = env_init.worldBounds;

    RigidBodyPhysicsSystem::init(ctx, env_init.numObjs * 10,
                                 env_init.numObjs * 50);

    render::RenderingSystem::init(ctx);

    broadphase::BVH &bp_bvh = ctx.getSingleton<broadphase::BVH>();

    auto setupEntity = [&](Entity e,
                           const Position &pos,
                           const Rotation &rot) {
        ctx.getUnsafe<Position>(e) = pos;
        ctx.getUnsafe<Rotation>(e) = rot;

        CollisionAABB &aabb = ctx.getUnsafe<CollisionAABB>(e);
        aabb = CollisionAABB(pos, rot);

        broadphase::LeafID &leaf_id = ctx.getUnsafe<broadphase::LeafID>(e);
        leaf_id = bp_bvh.reserveLeaf();

        bp_bvh.updateLeaf(e, leaf_id, aabb);
        
        ctx.getUnsafe<render::ObjectID>(e).idx = 0;
    };

    for (int i = 0; i < (int)env_init.numObjs; i++) {
        Entity e = ctx.makeEntityNow<Sphere>();
        setupEntity(e, env_init.objsInit[i].initPosition,
                    env_init.objsInit[i].initRotation);

        spheres[i] = e;
    }

    agent = ctx.makeEntityNow<Agent>();
    setupEntity(agent, {{ 0, 0, 0 }}, {Quat::angleAxis(0.f, {0, 1, 0})});
    ctx.getUnsafe<render::ActiveView>(agent) =
        render::RenderingSystem::setupView(ctx, 90.f);

    Entity test = ctx.makeEntityNow<Sphere>();
    setupEntity(test, {{ -10, 0, 0 }}, {Quat::angleAxis(0.f, {0, 1, 0})});

    bp_bvh.rebuild();
}

}

#ifdef MADRONA_GPU_MODE
extern "C" __global__ void madronaMWGPUInitialize(
        uint32_t num_worlds,
        void *inits_raw)
{
    using namespace SimpleTaskgraph;

    auto inits = (EnvInit *)inits_raw;
    SimEntry::init(inits, num_worlds);
}

#endif

