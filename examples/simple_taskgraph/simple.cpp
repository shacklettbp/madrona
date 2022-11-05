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

namespace SimpleTaskgraph {

SimpleSim::SimpleSim(const EnvInit &env_init)
{
    worldBounds = env_init.worldBounds;

    const int max_collisions = env_init.numObjs * env_init.numObjs;

    sphereObjects =
        (SphereObject *)malloc(sizeof(SphereObject) * env_init.numObjs);
    contacts = (ContactData *)malloc(
        sizeof(ContactData) * max_collisions);

    numSphereObjects = env_init.numObjs;
    numContacts = 0;

    for (int64_t i = 0; i < (int64_t)env_init.numObjs; i++) {
        sphereObjects[i] = SphereObject {
            env_init.objsInit[i].initPosition,
            env_init.objsInit[i].initRotation,
            AABB::invalid(),
        };
    }
}

SimManager::SimManager(const EnvInit *env_inits, uint32_t num_worlds)
    : preprocess(),
      broad(),
      narrow(),
      solver(),
      taskgraph(nullptr),
      sims((SimpleSim *)malloc(sizeof(SimpleSim) * num_worlds)),
      sphereIndices(nullptr),
      testIndices(nullptr)
{
    TaskGraph::Builder builder;
    auto preprocess_id = builder.registerSystem(preprocess, {});
    auto broad_id = builder.registerSystem(broad, {preprocess_id});
    auto narrow_id = builder.registerSystem(narrow, { broad_id });
    builder.registerSystem(solver, { narrow_id });
    taskgraph = builder.build();

    uint32_t total_spheres = 0;
    uint32_t max_collisions = 0;
    for (int i = 0; i < (int)num_worlds; i++) {
        new (&sims[i]) SimpleSim(env_inits[i]);
        uint32_t num_world_spheres = sims[i].numSphereObjects;
        total_spheres += num_world_spheres;
        max_collisions += num_world_spheres * num_world_spheres;
    }

    sphereIndices = (SphereIndex *)malloc(sizeof(SphereIndex) * total_spheres);
    testIndices = (TestIndex *)malloc(sizeof(TestIndex) * max_collisions);
    candidatePairs = (CandidatePair *)malloc(sizeof(CandidatePair) * max_collisions);

    uint32_t cur_global_sphere = 0;
    uint32_t cur_global_test = 0;
    for (int world_idx = 0; world_idx < (int)num_worlds; world_idx++) {
        uint32_t num_world_spheres = sims[world_idx].numSphereObjects;
        for (int offset = 0; offset < (int)num_world_spheres; offset++) {
            sphereIndices[cur_global_sphere++] = SphereIndex {
                (uint32_t)world_idx,
                (uint32_t)offset,
            };
            for (int other = 0; other < (int)num_world_spheres; other++) {
                testIndices[cur_global_test++] = TestIndex {
                    (uint32_t)world_idx,
                    (uint32_t)offset,
                    (uint32_t)other,
                };
            }
        }
    }

    preprocess.numInvocations.store(total_spheres, std::memory_order_relaxed);
    broad.numInvocations.store(max_collisions, std::memory_order_relaxed);
    solver.numInvocations.store(num_worlds, std::memory_order_relaxed);
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
    SphereObject &object = mgr.sims[sphere_idx.world].sphereObjects[sphere_idx.offset];

    // Clamp to world bounds
    SimpleSim &sim = mgr.sims[sphere_idx.world];
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
}

void BroadphaseSystem::run(void *gen_data, uint32_t invocation_offset)
{
    SimManager &mgr = *(SimManager *)gen_data;

    TestIndex &test = mgr.testIndices[invocation_offset];
    SimpleSim &sim = mgr.sims[test.world];
    uint32_t a_idx = test.a;
    uint32_t b_idx = test.b;

    if (a_idx == b_idx) return;

    SphereObject &a_obj = sim.sphereObjects[a_idx];
    SphereObject &b_obj = sim.sphereObjects[b_idx];

    if (a_obj.aabb.overlaps(b_obj.aabb)) {
        uint32_t candidate_idx = mgr.narrow.numInvocations.fetch_add(1,
             std::memory_order_relaxed);
        mgr.candidatePairs[candidate_idx] = CandidatePair {
            test.world,
            a_idx,
            b_idx,
        };
    }
}

void NarrowphaseSystem::run(void *gen_data, uint32_t invocation_offset)
{
    SimManager &mgr = *(SimManager *)gen_data;
    CandidatePair &c = mgr.candidatePairs[invocation_offset];

    SimpleSim &sim = mgr.sims[c.world];
    const SphereObject &a = sim.sphereObjects[c.a];
    const SphereObject &b = sim.sphereObjects[c.b];

    Translation a_pos = a.translation;
    Translation b_pos = b.translation;
    Vector3 to_b = (b_pos - a_pos).normalize();

    // FIXME: No actual narrow phase here
    uint32_t contact_idx =
        sim.numContacts.fetch_add(1, std::memory_order_relaxed);

    sim.contacts[contact_idx] = ContactData {
        to_b,
        c.a,
        c.b,
    };
}

void SolverSystem::run(void *gen_data, uint32_t invocation_offset)
{
    SimManager &mgr = *(SimManager *)gen_data;
    uint32_t world_idx = invocation_offset;

    SimpleSim &sim = mgr.sims[world_idx];

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

    sim.numContacts.store(0, std::memory_order_relaxed);
}

}
