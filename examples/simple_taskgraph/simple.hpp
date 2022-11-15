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

    inline uint32_t leafObjID(int32_t i)
    {
        return uint32_t(children[i] & ~0x80000000);
    }
};

struct SimpleSim;

struct PhysicsBVH {
    PhysicsBVH(int32_t initial_node_allocation);

    inline void build(SimpleSim &sim, int32_t *objs, int32_t num_objs);

    inline void update(SimpleSim &sim,
                       int32_t *added_objects, int32_t num_added_objects,
                       int32_t *removed_objects, int32_t num_removed_objects,
                       int32_t *moved_objects, int32_t num_moved_objects);

    template <typename Fn>
    inline void test(madrona::math::AABB &aabb, Fn &&fn)
    {
        int32_t stack[32];
        stack[0] = 0;
        int32_t stack_size = 1;

        while (stack_size > 0) {
            int32_t node_idx = stack[--stack_size];
            PhysicsBVHNode &node = nodes[node_idx];
            for (int i = 0; i < 4; i++) {
                int child_idx = node.children[i];
                if (child_idx == sentinel) {
                    continue;
                }

                madrona::math::AABB child_aabb {
                    .pMin = {
                        node.minX[i],
                        node.minY[i],
                        node.minZ[i],
                    },
                    .pMax = {
                        node.maxX[i],
                        node.maxY[i],
                        node.maxZ[i],
                    },
                };

                if (aabb.overlaps(child_aabb)) {
                    if (node.isLeaf(i)) {
                        fn(node.leafObjID(i));
                    } else {
                        stack[stack_size++] = child_idx;
                    }
                }
            }
        }
    }

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
    madrona::math::Vector3 physCenter;
    uint32_t leafID;
};

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
    SimpleSim(const EnvInit &env_init);

    madrona::math::AABB worldBounds;

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

using SimEntry = madrona::TaskGraphEntry<SimManager, EnvInit>;

class Engine;

}
