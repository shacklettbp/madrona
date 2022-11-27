/*
 * Copyright 2021-2022 Brennan Shacklett and contributors
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 */
#include "simple.hpp"

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

    registry.registerArchetype<Sphere>();
}

void SimpleSim::setupTasks(TaskGraph::Builder &builder)
{
    auto clamp_sys =
        builder.parallelForNode<Engine, clampSystem, Position>({});

    auto phys_sys = RigidBodyPhysicsSystem::setupTasks(builder, {clamp_sys});
    (void)phys_sys;

    printf("Setup done\n");
}

SimpleSim::SimpleSim(Engine &ctx, const EnvInit &env_init)
    : WorldBase(ctx),
      worldBounds(AABB::invalid()),
      spheres((Entity *)malloc(sizeof(Entity) * env_init.numObjs)),
      numSpheres(env_init.numObjs)
{
    worldBounds = env_init.worldBounds;

    RigidBodyPhysicsSystem::init(ctx, env_init.numObjs * 10,
                                 env_init.numObjs * 50);

    broadphase::BVH &bp_bvh = ctx.getSingleton<broadphase::BVH>();
    for (int i = 0; i < (int)env_init.numObjs; i++) {
        Entity e = ctx.makeEntityNow<Sphere>();
        Position &position = ctx.getUnsafe<Position>(e);
        Rotation &rotation = ctx.getUnsafe<Rotation>(e);
        position = env_init.objsInit[i].initPosition;
        rotation = env_init.objsInit[i].initRotation;
        spheres[i] = e;

        CollisionAABB &aabb = ctx.getUnsafe<CollisionAABB>(e);
        aabb = CollisionAABB(position, rotation);

        broadphase::LeafID &leaf_id = ctx.getUnsafe<broadphase::LeafID>(e);
        leaf_id = bp_bvh.reserveLeaf();

        bp_bvh.updateLeaf(e, leaf_id, aabb);
    }

    bp_bvh.rebuild();
}

}

#ifdef MADRONA_GPU_MODE
extern "C" __global__ void madronaMWGPUInitialize(uint32_t num_worlds,
                                                  void *inits_raw)
{
    using namespace SimpleTaskgraph;

    auto inits = (EnvInit *)inits_raw;
    SimEntry::init(inits, num_worlds);
}

#endif

