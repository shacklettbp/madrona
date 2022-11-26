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

inline void solverSystem(Engine &ctx, SolverData &)
{
    printf("Solver %d\n", ctx.worldID().idx);
}

void SimpleSim::registerTypes(ECSRegistry &registry)
{
    base::registerTypes(registry);
    phys::registerTypes(registry);
    broadphase::System::registerTypes(registry);

    registry.registerSingleton<SolverData>();

    registry.registerArchetype<Sphere>();
}

void SimpleSim::setupTasks(TaskGraph::Builder &builder)
{
    auto clamp_sys =
        builder.parallelForNode<Engine, clampSystem, Position>({});

    auto aabb_sys = CollisionAABB::setupTasks(builder, {clamp_sys});

    auto broadphase_sys = broadphase::System::setupTasks(builder,
        { aabb_sys });

    auto narrowphase_sys = narrowphase::System::setupTasks(builder,
        { broadphase_sys });
    
    builder.parallelForNode<Engine, solverSystem, SolverData>(
        { narrowphase_sys });

    printf("Setup done\n");
}

SimpleSim::SimpleSim(Engine &ctx, const EnvInit &env_init)
    : WorldBase(ctx),
      worldBounds(AABB::invalid()),
      spheres((Entity *)malloc(sizeof(Entity) * env_init.numObjs)),
      numSpheres(env_init.numObjs)
{
    worldBounds = env_init.worldBounds;

    broadphase::System::init(ctx, env_init.numObjs * 10);

    broadphase::BVH &bp_bvh = ctx.getSingleton<broadphase::BVH>();
    for (int i = 0; i < (int)env_init.numObjs; i++) {
        Entity e = ctx.makeEntityNow<Sphere>();
        Position &position = ctx.getComponent<Sphere, Position>(e);
        Rotation &rotation = ctx.getComponent<Sphere, Rotation>(e);
        position = env_init.objsInit[i].initPosition;
        rotation = env_init.objsInit[i].initRotation;
        spheres[i] = e;

        CollisionAABB &aabb = ctx.getComponent<Sphere, CollisionAABB>(e);
        aabb = CollisionAABB(position, rotation);

        broadphase::LeafID &leaf_id = ctx.getComponent<Sphere, broadphase::LeafID>(e);
        leaf_id = bp_bvh.reserveLeaf();

        bp_bvh.updateLeaf(e, leaf_id, aabb);
    }

    bp_bvh.rebuild();
}

#if 0
template <bool use_atomic>
static inline void narrowPhase(SimpleSim &sim, uint32_t a_idx, uint32_t b_idx)
{
    const SphereObject &a = sim.sphereObjects[a_idx];
    const SphereObject &b = sim.sphereObjects[b_idx];

    Translation a_pos = a.translation;
    Translation b_pos = b.translation;
    Vector3 to_b = (b_pos - a_pos).normalize();

    // FIXME: No actual narrow phase here
    uint32_t contact_idx;

    if constexpr (use_atomic) {
        contact_idx =
            sim.numContacts.fetch_add(1, std::memory_order_relaxed);
    } else {
        contact_idx = sim.numContacts.load(std::memory_order_relaxed);
        sim.numContacts.store(contact_idx + 1, std::memory_order_relaxed);
    }

    sim.contacts[contact_idx] = ContactData {
        to_b,
        a_idx,
        b_idx,
    };
}

void NarrowphaseSystem::run(void *gen_data, uint32_t invocation_offset)
{
    SimManager &mgr = *(SimManager *)gen_data;
    CandidatePair &c = mgr.candidatePairs[invocation_offset];

    SimpleSim &sim = mgr.sims[c.world];
    narrowPhase<true>(sim, c.a, c.b);
}

static void processContacts(SimpleSim &sim)
{
    // Push objects in serial based on the contact normal - total BS.
    int num_contacts = sim.numContacts.load(std::memory_order_relaxed);

    for (int i = 0; i < num_contacts; i++) {
        ContactData &contact = sim.contacts[i];

        SphereObject &a = sim.sphereObjects[contact.a];
        SphereObject &b = sim.sphereObjects[contact.b];

        Translation &a_pos = a.translation;
        Translation &b_pos = b.translation;

        a_pos -= contact.normal;
        b_pos += contact.normal;
    }
}

void SolverSystem::run(void *gen_data, uint32_t invocation_offset)
{
    SimManager &mgr = *(SimManager *)gen_data;
    uint32_t world_idx = invocation_offset;

    SimpleSim &sim = mgr.sims[world_idx];

    processContacts(sim);

    sim.numContacts.store(0, std::memory_order_relaxed);
}

#endif

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

