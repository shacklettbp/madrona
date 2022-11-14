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

namespace SimpleTaskgraph {

// Components
struct Translation : madrona::math::Vector3 {
    Translation(madrona::math::Vector3 v)
        : Vector3(v)
    {}
};

struct Rotation : madrona::math::Quat {
    Rotation(madrona::math::Quat q)
        : Quat(q)
    {}
};

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
    Translation initPosition;
    Rotation initRotation;
};

struct EnvInit {
    madrona::math::AABB worldBounds;
    ObjectInit *objsInit;
    uint32_t numObjs;
};

// List of Physics Related Components per-object

struct PhysicsAABB : madrona::math::AABB {
    PhysicsAABB(madrona::math::AABB b)
        : AABB(b)
    {}
};

struct PhysicsBVHNode {
    float minX[4];
    float minY[4];
    float minZ[4];
    float maxX[4];
    float maxY[4];
    float maxZ[4];
    int32_t children[4];
    int32_t parentID;

    inline void clearChild(int32_t i)
    {
        children[i] = 0xFFFFFFFF;
    }

    inline bool isLeaf(int32_t i)
    {
        return children[i] & 0x80000000;
    }

    inline void setLeaf(int32_t i, int32_t obj_id)
    {
        children[i] = 0x80000000 | obj_id;
    }
};

struct SimpleSim;

struct PhysicsBVH {
    PhysicsBVH(int32_t initial_node_allocation);

    inline void update(SimpleSim &sim,
                       uint32_t *added_objects, uint32_t num_added_objects,
                       uint32_t *removed_objects, uint32_t num_removed_objects,
                       uint32_t *moved_objects, uint32_t num_moved_objects);

    inline void addObject(SimpleSim &sim, uint32_t obj_id);

    static constexpr int32_t sentinel = int32_t(-1);

    PhysicsBVHNode *nodes;
    int32_t numNodes;
    int32_t numAllocatedNodes;
    int32_t freeHead;
};

struct SphereObject {
    Translation translation;
    Rotation rotation;
    PhysicsAABB aabb;
    uint32_t leafID;
};

struct PreprocessSystem : madrona::CustomSystem<PreprocessSystem> {
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
    SimpleSim(const EnvInit &env_init);

    madrona::math::AABB worldBounds;

    SphereObject *sphereObjects;
    ContactData *contacts;
    uint32_t numSphereObjects;
    std::atomic_uint32_t numContacts;

    madrona::utils::SpinLock candidateCreateLock {};
    madrona::utils::SpinLock contactCreateLock {};
};

struct SphereIndex {
    uint32_t world;
    uint32_t offset;
};

struct TestIndex {
    uint32_t world;
    uint32_t a;
    uint32_t b;
};

struct SimManager {
    SimManager(const EnvInit *env_inits, uint32_t num_worlds);
    void taskgraphSetup(madrona::TaskGraph::Builder &builder);

    PreprocessSystem preprocess;
    BroadphaseSystem broad;
    NarrowphaseSystem narrow;
    SolverSystem solver;
    UnifiedSystem unified;

    bool useUnified = true;

    SimpleSim *sims;

    SphereIndex *sphereIndices;
    TestIndex *testIndices;
    CandidatePair *candidatePairs;
};

using SimEntry = madrona::TaskGraphEntry<SimManager, EnvInit>;

class Engine;

}
