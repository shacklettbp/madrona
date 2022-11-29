/*
 * Copyright 2021-2022 Brennan Shacklett and contributors
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 */
#pragma once

#include <madrona/math.hpp>
#include <madrona/taskgraph.hpp>
#include <madrona/custom_context.hpp>
#include <madrona/physics.hpp>
#include <madrona/render.hpp>

namespace SimpleTaskgraph {

using Position = madrona::base::Position;
using Rotation = madrona::base::Rotation;

struct CandidatePair {
    uint32_t world;
    uint32_t a;
    uint32_t b;
};

struct ContactData {
    madrona::math::Vector3 normal;
    uint32_t a;
    uint32_t b;
};

struct ObjectInit {
    Position initPosition;
    Rotation initRotation;
};

struct EnvInit {
    madrona::math::AABB worldBounds;
    ObjectInit *objsInit;
    uint32_t numObjs;
};

struct Sphere : public madrona::Archetype<
    Position, 
    Rotation,
    madrona::phys::CollisionAABB,
    madrona::phys::broadphase::LeafID,
    madrona::render::ObjectID
> {};

class Engine;

struct SimpleSim : public madrona::WorldBase {
    static void registerTypes(madrona::ECSRegistry &registry);

    static void setupTasks(madrona::TaskGraph::Builder &builder);

    SimpleSim(Engine &ctx, const EnvInit &env_init);

    madrona::math::AABB worldBounds;

    madrona::Entity *spheres;
    int32_t numSpheres;
};

class Engine : public ::madrona::CustomContext<Engine, SimpleSim> {
    using CustomContext::CustomContext;
};

using SimEntry = madrona::TaskGraphEntry<Engine, SimpleSim, EnvInit>;

#if 0
struct PreprocessSystem : madrona::CustomSystem<PreprocessSystem> {
    inline void run(void *data, uint32_t invocation_offset);
};

struct BVHSystem : madrona::CustomSystem<BVHSystem> {
    inline void run(void *data, uint32_t invocation_offset);
};

struct BroadphaseSystem : madrona::CustomSystem<BroadphaseSystem> {
    inline void run(void *data, uint32_t invocation_offset);
};

struct NarrowphaseSystem : madrona::CustomSystem<NarrowphaseSystem> {
    inline void run(void *data, uint32_t invocation_offset);
};

struct SolverSystem : madrona::CustomSystem<SolverSystem> {
    inline void run(void *data, uint32_t invocation_offset);
};

struct UnifiedSystem : madrona::CustomSystem<UnifiedSystem> {
    inline void run(void *data, uint32_t invocation_offset);
};

struct SimpleSim {
    SimpleSim(madrona::TaskGraph &graph, const EnvInit &env_init);

    static void registerSystems(madrona::TaskGraph::Builder &builder);

    madrona::math::AABB worldBounds;

    PreprocessSystem preprocess;
    BVHSystem bvhUpdate;
    BroadphaseSystem broad;
    NarrowphaseSystem narrow;
    SolverSystem solver;

    SphereObject *sphereObjects;
    ContactData *contacts;
    int32_t *bvhObjIDs;
    uint32_t numSphereObjects;
    std::atomic_uint32_t numContacts;
    PhysicsBVH bvh;

    madrona::utils::SpinLock candidateCreateLock {};
    madrona::utils::SpinLock contactCreateLock {};
};

struct SphereIndex {
    uint32_t world;
    uint32_t offset;
};

struct SimManager {
    SimManager(const EnvInit *env_inits, uint32_t num_worlds);
    void taskgraphSetup(madrona::TaskGraph::Builder &builder);

    PreprocessSystem preprocess;
    BVHSystem bvhUpdate;
    BroadphaseSystem broad;
    NarrowphaseSystem narrow;
    SolverSystem solver;
    UnifiedSystem unified;

    bool useUnified = true;

    SimpleSim *sims;

    SphereIndex *sphereIndices;
    CandidatePair *candidatePairs;
};

#endif

class Engine;

}
