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
    printf("%d\n", ctx.worldID().idx);
}

void SimpleSim::registerTypes(ECSRegistry &registry)
{
    base::registerTypes(registry);
    broadphase::registerTypes(registry);

    registry.registerComponent<SolverData>();

    registry.registerArchetype<Sphere>();
    registry.registerArchetype<SolverSystem>();
}

void SimpleSim::setupTasks(TaskGraph::Builder &builder)
{
    auto clamp_sys =
        builder.parallelForNode<Engine, clampSystem, Position>({});

    auto broadphase_sys = broadphase::setupTasks(builder, { clamp_sys });
    
    builder.parallelForNode<Engine, solverSystem, SolverData>(
        { broadphase_sys });

    printf("Setup done\n");
}

SimpleSim::SimpleSim(Engine &ctx, const EnvInit &env_init)
    : WorldBase(ctx),
      worldBounds(AABB::invalid()),
      spheres((Entity *)malloc(sizeof(Entity) * env_init.numObjs)),
      numSpheres(env_init.numObjs),
      broadphaseBVH(env_init.numObjs * 10)
{
    worldBounds = env_init.worldBounds;

    ctx.makeEntityNow<SolverSystem>();

    for (int i = 0; i < (int)env_init.numObjs; i++) {
        Entity e = ctx.makeEntityNow<Sphere>();
        Position &position = ctx.getComponent<Sphere, Position>(e);
        Rotation &rotation = ctx.getComponent<Sphere, Rotation>(e);

        position = env_init.objsInit[i].initPosition;
        rotation = env_init.objsInit[i].initRotation;
        spheres[i] = e;
    }

#if 0
    const int max_collisions = env_init.numObjs * env_init.numObjs;

    sphereObjects =
        (SphereObject *)malloc(sizeof(SphereObject) * env_init.numObjs);
    contacts =
        (ContactData *)malloc(sizeof(ContactData) * max_collisions);
    bvhObjIDs = 
        (int32_t *)malloc(sizeof(int32_t) * env_init.numObjs);

    numSphereObjects = env_init.numObjs;
    numContacts = 0;

    for (int i = 0; i < (int)env_init.numObjs; i++) {
        sphereObjects[i] = SphereObject {
            env_init.objsInit[i].initPosition,
            env_init.objsInit[i].initRotation,
            AABB::invalid(),
            Vector3 {},
            0xFFFFFFFF,
        };
        preprocessObject(*this, i);

        bvhObjIDs[i] = i;
    }

    bvh.update(*this, bvhObjIDs, env_init.numObjs, nullptr, 0, nullptr, 0);
#endif
}

#if 0
static inline void preprocessObject(SimpleSim &sim, uint32_t obj_id)
{
    SphereObject &object = sim.sphereObjects[obj_id];

    // Clamp to world bounds
    object.translation.x = std::clamp(object.translation.x,
                                      sim.worldBounds.pMin.x,
                                      sim.worldBounds.pMax.x);
    object.translation.y = std::clamp(object.translation.y,
                                      sim.worldBounds.pMin.y,
                                      sim.worldBounds.pMax.y);
    object.translation.z = std::clamp(object.translation.z,
                                      sim.worldBounds.pMin.z,
                                      sim.worldBounds.pMax.z);

    // No actual mesh, just hardcode a fake 2 *unit cube centered around
    // translation
    
    Mat3x4 model_mat =
        Mat3x4::fromTRS(object.translation, object.rotation);

    Vector3 cube[8] = {
        model_mat.txfmPoint(Vector3 {-1.f, -1.f, -1.f}),
        model_mat.txfmPoint(Vector3 { 1.f, -1.f, -1.f}),
        model_mat.txfmPoint(Vector3 { 1.f,  1.f, -1.f}),
        model_mat.txfmPoint(Vector3 {-1.f,  1.f, -1.f}),
        model_mat.txfmPoint(Vector3 {-1.f, -1.f,  1.f}),
        model_mat.txfmPoint(Vector3 { 1.f, -1.f,  1.f}),
        model_mat.txfmPoint(Vector3 { 1.f,  1.f,  1.f}),
        model_mat.txfmPoint(Vector3 {-1.f,  1.f,  1.f}),
    };

    object.aabb = AABB::point(cube[0]);
    for (int i = 1; i < 8; i++) {
        object.aabb.expand(cube[i]);
    }

    object.physCenter = (object.aabb.pMin + object.aabb.pMax) / 2;
}

SimpleSim::SimpleSim(const EnvInit &env_init)
    : worldBounds(AABB::invalid()),
      sphereObjects(nullptr),
      contacts(nullptr),
      bvhObjIDs(nullptr),
      numSphereObjects(env_init.numObjs),
      numContacts(0),
      bvh(env_init.numObjs * 10)
{
    worldBounds = env_init.worldBounds;

    const int max_collisions = env_init.numObjs * env_init.numObjs;

    sphereObjects =
        (SphereObject *)malloc(sizeof(SphereObject) * env_init.numObjs);
    contacts =
        (ContactData *)malloc(sizeof(ContactData) * max_collisions);
    bvhObjIDs = 
        (int32_t *)malloc(sizeof(int32_t) * env_init.numObjs);

    numSphereObjects = env_init.numObjs;
    numContacts = 0;

    for (int i = 0; i < (int)env_init.numObjs; i++) {
        sphereObjects[i] = SphereObject {
            env_init.objsInit[i].initPosition,
            env_init.objsInit[i].initRotation,
            AABB::invalid(),
            Vector3 {},
            0xFFFFFFFF,
        };
        preprocessObject(*this, i);

        bvhObjIDs[i] = i;
    }

    bvh.update(*this, bvhObjIDs, env_init.numObjs, nullptr, 0, nullptr, 0);
}

void SimpleSim::registerSystems(TaskGraph::Builder &builder)
{
}

SimManager::SimManager(const EnvInit *env_inits, uint32_t num_worlds)
    : preprocess(),
      bvhUpdate(),
      broad(),
      narrow(),
      solver(),
      unified(),
      sims((SimpleSim *)malloc(sizeof(SimpleSim) * num_worlds)),
      sphereIndices(nullptr),
      candidatePairs(nullptr)
{
    uint32_t total_spheres = 0;
    uint32_t max_collisions = 0;
    for (int i = 0; i < (int)num_worlds; i++) {
        new (&sims[i]) SimpleSim(env_inits[i]);
        uint32_t num_world_spheres = sims[i].numSphereObjects;
        total_spheres += num_world_spheres;
        max_collisions += num_world_spheres * num_world_spheres;
    }

    sphereIndices = (SphereIndex *)malloc(sizeof(SphereIndex) * total_spheres);
    candidatePairs = (CandidatePair *)malloc(sizeof(CandidatePair) * max_collisions);

    uint32_t cur_global_sphere = 0;
    for (int world_idx = 0; world_idx < (int)num_worlds; world_idx++) {
        uint32_t num_world_spheres = sims[world_idx].numSphereObjects;
        for (int offset = 0; offset < (int)num_world_spheres; offset++) {
            sphereIndices[cur_global_sphere++] = SphereIndex {
                (uint32_t)world_idx,
                (uint32_t)offset,
            };
        }
    }

    preprocess.numInvocations.store(total_spheres, std::memory_order_relaxed);
    bvhUpdate.numInvocations.store(num_worlds, std::memory_order_relaxed);
    broad.numInvocations.store(total_spheres, std::memory_order_relaxed);
    solver.numInvocations.store(num_worlds, std::memory_order_relaxed);
    unified.numInvocations.store(num_worlds, std::memory_order_relaxed);
}

void SimManager::taskgraphSetup(TaskGraph::Builder &builder)
{
    if (useUnified) {
        builder.registerSystem(unified, {});
    } else {
        auto preprocess_id = builder.registerSystem(preprocess, {});
        auto bvh_id = builder.registerSystem(bvhUpdate, { preprocess_id });
        auto broad_id = builder.registerSystem(broad, { bvh_id });
        auto narrow_id = builder.registerSystem(narrow, { broad_id });
        builder.registerSystem(solver, { narrow_id });
    }
}

PhysicsBVH::PhysicsBVH(int32_t initial_node_allocation)
    : nodes((PhysicsBVHNode *)malloc(
            sizeof(PhysicsBVHNode) * initial_node_allocation)),
      numNodes(0),
      numAllocatedNodes(initial_node_allocation),
      freeHead(sentinel)
{
}

void PhysicsBVH::build(SimpleSim &sim,
                       int32_t *added_objects, int32_t num_added_objects)
{
    int32_t num_internal_nodes =
        utils::divideRoundUp((int32_t)num_added_objects - 1, 3);

    int32_t cur_node_offset = numNodes;
    assert(cur_node_offset == 0); // FIXME
    numNodes += num_internal_nodes;

    struct StackEntry {
        int32_t nodeID;
        int32_t parentID;
        int32_t offset;
        int32_t numObjs;
    };

    StackEntry stack[128];
    stack[0] = StackEntry {
        sentinel,
        sentinel,
        0,
        num_added_objects,
    };

    int32_t stack_size = 1;

    while (stack_size > 0) {
        StackEntry &entry = stack[stack_size - 1];
        int32_t node_id;
        if (entry.numObjs <= 4) {
            node_id = cur_node_offset++;
            PhysicsBVHNode &node = nodes[node_id];
            node.parentID = entry.parentID;

            for (int i = 0; i < 4; i++) {
                if (i < entry.numObjs) {
                    int32_t obj_id = added_objects[entry.offset + i];
                    SphereObject &obj = sim.sphereObjects[obj_id];
                    node.setLeaf(i, obj_id);
                    node.minX[i] = obj.aabb.pMin.x;
                    node.minY[i] = obj.aabb.pMin.y;
                    node.minZ[i] = obj.aabb.pMin.z;
                    node.maxX[i] = obj.aabb.pMax.x;
                    node.maxY[i] = obj.aabb.pMax.y;
                    node.maxZ[i] = obj.aabb.pMax.z;
                } else {
                    node.children[i] = sentinel;
                    node.minX[i] = FLT_MAX;
                    node.minY[i] = FLT_MAX;
                    node.minZ[i] = FLT_MAX;
                    node.maxX[i] = FLT_MIN;
                    node.maxY[i] = FLT_MIN;
                    node.maxZ[i] = FLT_MIN;
                }
            }
        } else if (entry.nodeID == sentinel) {
            node_id = cur_node_offset++;
            // Record the node id in the stack entry for when this entry
            // is reprocessed
            entry.nodeID = node_id;

            PhysicsBVHNode &node = nodes[node_id];
            for (int i = 0; i < 4; i++) {
                node.children[i] = sentinel;
            }
            node.parentID = entry.parentID;

            // midpoint sort items
            auto midpoint_split = [&sim, added_objects](
                                      int32_t base, int32_t num_elems) {
                Vector3 center_min {
                    FLT_MAX,
                    FLT_MAX,
                    FLT_MAX,
                };

                Vector3 center_max {
                    FLT_MIN,
                    FLT_MIN,
                    FLT_MIN,
                };

                for (int i = 0; i < num_elems; i++) {
                    int32_t obj_id = added_objects[base + i];
                    SphereObject &obj = sim.sphereObjects[obj_id];
                    center_min = Vector3::min(center_min, obj.physCenter);
                    center_max = Vector3::max(center_max, obj.physCenter);
                }

                auto split = [&](auto get_component) {
                    float split_val = 0.5f * (get_component(center_min) +
                                              get_component(center_max));

                    int start = 0;
                    int end = num_elems;

                    while (start < end) {
                        auto center_component = [&](int32_t idx) {
                            int32_t obj_id = added_objects[base + idx];
                            return get_component(
                                sim.sphereObjects[obj_id].physCenter);
                        };

                        while (start < end && center_component(start) < 
                               split_val) {
                            ++start;
                        }

                        while (start < end && center_component(end - 1) >=
                               split_val) {
                            --end;
                        }

                        if (start < end) {
                            std::swap(added_objects[base + start],
                                      added_objects[base + end - 1]);
                            ++start;
                            --end;
                        }
                    }

                    if (start > 0 && start < num_elems) {
                        return start;
                    } else {
                        return num_elems / 2;
                    }
                };

                Vector3 center_diff = center_max - center_min;
                if (center_diff.x > center_diff.y &&
                    center_diff.x > center_diff.z) {
                    return split([](Vector3 v) {
                        return v.x;
                    });
                } else if (center_diff.y > center_diff.x &&
                           center_diff.y > center_diff.z) {
                    return split([](Vector3 v) {
                        return v.y;
                    });
                } else {
                    return split([](Vector3 v) {
                        return v.z;
                    });
                }
            };

            int32_t second_split = midpoint_split(entry.offset, entry.numObjs);
            int32_t num_h1 = second_split;
            int32_t num_h2 = entry.numObjs - second_split;

            int32_t first_split = midpoint_split(entry.offset, num_h1);
            int32_t third_split =
                midpoint_split(entry.offset + second_split, num_h2);

#if 0
            printf("%u %u\n", entry.offset, entry.numObjs);
            printf("[%u %u) [%u %u) [%u %u) [%u %u)\n",
                   entry.offset, entry.offset + first_split,
                   entry.offset + first_split, entry.offset + first_split + num_h1 - first_split,
                   entry.offset + num_h1, entry.offset + num_h1 + third_split,
                   entry.offset + num_h1 + third_split, entry.offset + num_h1 + third_split + num_h2 - third_split);
#endif

            // Setup stack to recurse into fourths. Put fourths on stack in
            // reverse order to preserve left-right depth first ordering

            stack[stack_size++] = {
                -1,
                entry.nodeID,
                entry.offset + num_h1 + third_split,
                num_h2 - third_split,
            };

            stack[stack_size++] = {
                -1,
                entry.nodeID,
                entry.offset + num_h1,
                third_split,
            };

            stack[stack_size++] = {
                -1,
                entry.nodeID,
                entry.offset + first_split,
                num_h1 - first_split,
            };

            stack[stack_size++] = {
                -1,
                entry.nodeID,
                entry.offset,
                first_split,
            };

            // Don't finish processing this node until children are processed
            continue;
        } else {
            // Revisiting this node after having processed children
            node_id = entry.nodeID;
        }

        // At this point, remove the current entry from the stack
        stack_size -= 1;

        PhysicsBVHNode &node = nodes[node_id];
        if (node.parentID == -1) {
            continue;
        }

        AABB combined_aabb  = AABB::invalid(); 
        for (int i = 0; i < 4; i++) {
            if (node.children[i] == sentinel) {
                break;
            }

            combined_aabb = AABB::merge(combined_aabb, AABB {
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
            });
        }

        PhysicsBVHNode &parent = nodes[node.parentID];
        int child_offset;
        for (child_offset = 0; ; child_offset++) {
            if (parent.children[child_offset] == sentinel) {
                break;
            }
        }

        parent.children[child_offset] = entry.nodeID;
        parent.minX[child_offset] = combined_aabb.pMin.x;
        parent.minY[child_offset] = combined_aabb.pMin.y;
        parent.minZ[child_offset] = combined_aabb.pMin.z;
        parent.maxX[child_offset] = combined_aabb.pMax.x;
        parent.maxY[child_offset] = combined_aabb.pMax.y;
        parent.maxZ[child_offset] = combined_aabb.pMax.z;
    }
}

void PhysicsBVH::update(SimpleSim &sim,
                        int32_t *added_objects, int32_t num_added_objects,
                        int32_t *removed_objects, int32_t num_removed_objects,
                        int32_t *moved_objects, int32_t num_moved_objects)
{
    if (num_added_objects > 0) {
        build(sim, added_objects, num_added_objects);
    }
    (void)removed_objects;
    (void)num_removed_objects;

    for (int i = 0; i < (int)num_moved_objects; i++) {
        int32_t obj_id = moved_objects[i];
        SphereObject &obj = sim.sphereObjects[obj_id];
        AABB obj_aabb = obj.aabb;

        int32_t node_idx = int32_t(obj.leafID >> 2_u32);
        int32_t sub_idx = int32_t(obj.leafID & 3);

        PhysicsBVHNode &leaf_node = nodes[node_idx];
        leaf_node.minX[sub_idx] = obj_aabb.pMin.x;
        leaf_node.minY[sub_idx] = obj_aabb.pMin.y;
        leaf_node.minZ[sub_idx] = obj_aabb.pMin.z;
        leaf_node.maxX[sub_idx] = obj_aabb.pMax.x;
        leaf_node.maxY[sub_idx] = obj_aabb.pMax.y;
        leaf_node.maxZ[sub_idx] = obj_aabb.pMax.z;

        int32_t child_idx = node_idx;
        node_idx = leaf_node.parentID;

        while (node_idx != sentinel) {
            PhysicsBVHNode &node = nodes[node_idx];
            int child_offset = -1;
            for (int j = 0; j < 4; j++) {
                if (node.children[j] == child_idx) {
                    child_offset = j;
                    break;
                }
            }
            assert(child_offset != -1);

            bool expanded = false;
            if (obj_aabb.pMin.x < node.minX[child_offset]) {
                node.minX[child_offset] = obj_aabb.pMin.x;
                expanded = true;
            }

            if (obj_aabb.pMin.y < node.minY[child_offset]) {
                node.minY[child_offset] = obj_aabb.pMin.y;
                expanded = true;
            }

            if (obj_aabb.pMin.z < node.minZ[child_offset]) {
                node.minZ[child_offset] = obj_aabb.pMin.z;
                expanded = true;
            }

            if (obj_aabb.pMax.x > node.maxX[child_offset]) {
                node.maxX[child_offset] = obj_aabb.pMax.x;
                expanded = true;
            }

            if (obj_aabb.pMax.y > node.maxY[child_offset]) {
                node.maxY[child_offset] = obj_aabb.pMax.y;
                expanded = true;
            }

            if (obj_aabb.pMax.z > node.maxZ[child_offset]) {
                node.maxZ[child_offset] = obj_aabb.pMax.z;
                expanded = true;
            }

            if (!expanded) {
                break;
            }
            
            child_idx = node_idx;
            node_idx = node.parentID;
        }
    }
}

// Update all entity bounding boxes:
void PreprocessSystem::run(void *gen_data, uint32_t invocation_offset)
{
    SimManager &mgr = *(SimManager *)gen_data;

    // Hacky one time setup
    if (invocation_offset == 0) {
        mgr.narrow.numInvocations.store(0, std::memory_order_relaxed);
    }

    SphereIndex &sphere_idx = mgr.sphereIndices[invocation_offset];
    SimpleSim &sim = mgr.sims[sphere_idx.world];
    preprocessObject(sim, sphere_idx.offset);
}

void BVHSystem::run(void *gen_data, uint32_t invocation_offset)
{
    SimManager &mgr = *(SimManager *)gen_data;

    SimpleSim &sim = mgr.sims[invocation_offset];
    sim.bvh.update(sim, nullptr, 0, nullptr, 0, sim.bvhObjIDs,
                   sim.numSphereObjects);
}

static inline bool compareObjAABBs(SimpleSim &sim, uint32_t a_idx, uint32_t b_idx)
{
    if (a_idx == b_idx) return false;

    SphereObject &a_obj = sim.sphereObjects[a_idx];
    SphereObject &b_obj = sim.sphereObjects[b_idx];

    return a_obj.aabb.overlaps(b_obj.aabb);
}

void BroadphaseSystem::run(void *gen_data, uint32_t invocation_offset)
{
    SimManager &mgr = *(SimManager *)gen_data;
    SphereIndex &sphere_idx = mgr.sphereIndices[invocation_offset];
    SimpleSim &sim = mgr.sims[sphere_idx.world];
    SphereObject &obj = sim.sphereObjects[sphere_idx.offset];

    sim.bvh.test(obj.aabb, [&](uint32_t other) {
        uint32_t candidate_idx = mgr.narrow.numInvocations.fetch_add(1,
             std::memory_order_relaxed);
        if (sphere_idx.offset < other) {
            mgr.candidatePairs[candidate_idx] = CandidatePair {
                sphere_idx.world,
                sphere_idx.offset,
                other,
            };
        }
    });
}

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

void UnifiedSystem::run(void *gen_data, uint32_t invocation_offset)
{
    SimManager &mgr = *(SimManager *)gen_data;

    uint32_t world_idx = invocation_offset;

    SimpleSim &sim = mgr.sims[world_idx];

    for (int i = 0; i < (int)sim.numSphereObjects; i++) {
        preprocessObject(sim, i);
    }

    for (int i = 0; i < (int)sim.numSphereObjects; i++) {
        for (int j = 0; j < (int)sim.numSphereObjects; j++) {
            bool overlap = compareObjAABBs(sim, i, j);
            if (overlap) {
                narrowPhase<false>(sim, i, j);
            }
        }
    }

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

