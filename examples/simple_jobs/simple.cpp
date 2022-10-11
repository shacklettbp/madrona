/*
 * Copyright 2021-2022 Brennan Shacklett and contributors
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 */
#include "simple.hpp"

#include <cinttypes>
#include <random>

using namespace madrona;
using namespace madrona::math;

namespace SimpleExample {

// FIXME: thread-safe random number generator.
// Need to provide this as subroutine in the framework for GPU compat
static std::mt19937 & randGen()
{
    //thread_local std::mt19937 rand_gen { std::random_device {}() };
    thread_local std::mt19937 rand_gen { 0 };

    return rand_gen;
}

static Vector3 randomPosition(const AABB &bounds)
{
    std::uniform_real_distribution<float> x_dist(bounds.pMin.x, bounds.pMax.x);
    std::uniform_real_distribution<float> y_dist(bounds.pMin.y, bounds.pMax.y);
    std::uniform_real_distribution<float> z_dist(bounds.pMin.z, bounds.pMax.z);
    
    return Vector3 {
        x_dist(randGen()),
        y_dist(randGen()),
        z_dist(randGen()),
    };
}

SimpleSim::SimpleSim(Engine &)
{
    // World attributes (constant for now)
    tickCount = 0;
    deltaT = 1.f / 60.f;

    worldBounds = {
        .pMin = Vector3 { -10, -10, 0, },
        .pMax = Vector3 { 10, 10, 10, },
    };

    const int init_num_objects = 100;
    const int max_collisions = init_num_objects * init_num_objects;

    sphereObjects =
        (SphereObject *)malloc(sizeof(SphereObject) * init_num_objects);
    candidatePairs = (CandidatePair *)malloc(
        sizeof(CandidatePair) * max_collisions);
    contacts = (ContactData *)malloc(
        sizeof(ContactData) * max_collisions);

    numSphereObjects = init_num_objects;
    numCandidatePairs = 0;
    numContacts = 0;

    std::uniform_real_distribution<float> angle_dist(0.f, M_PI); 
    for (int i = 0; i < init_num_objects; i++) {
        Translation rand_pos = randomPosition(worldBounds);
        Rotation rand_rot = Quat::angleAxis(angle_dist(randGen()),
            Vector3 { 0, 1, 0 });

        PhysicsAABB aabb = AABB::invalid();

        sphereObjects[i] = SphereObject {
            rand_pos,
            rand_rot,
            aabb,
        };
    }
}

static JobID broadphaseSystem(Engine &ctx)
{
    // Update all entity bounding boxes:
    // FIXME: future improvement - sleeping entities for physics
    JobID preprocess = ctx.submitN([](Engine &ctx, uint32_t idx) {
        SphereObject &object = ctx.sim().sphereObjects[idx];
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
    }, ctx.sim().numSphereObjects);

    // Generate list of CollisionCandidates for narrowphase
    return ctx.submit([](Engine &ctx) {
        uint32_t num_checks = ctx.sim().numSphereObjects * ctx.sim().numSphereObjects;
        ctx.submitN([](Engine &ctx, uint32_t idx) {
            SimpleSim &sim = ctx.sim();
            uint32_t a_idx = idx / sim.numSphereObjects;
            uint32_t b_idx = idx % sim.numSphereObjects;

            if (a_idx == b_idx) return;

            SphereObject &a_obj = sim.sphereObjects[a_idx];
            SphereObject &b_obj = sim.sphereObjects[b_idx];

            if (a_obj.aabb.overlaps(b_obj.aabb)) {
                uint32_t candidate_idx = sim.numCandidatePairs.fetch_add(1,
                     std::memory_order_relaxed);
                sim.candidatePairs[candidate_idx] = CandidatePair {
                    a_idx,
                    b_idx,
                };
            }
        }, num_checks);
    }, true, preprocess);
}

static JobID narrowphaseSystem(Engine &ctx, JobID broadphase_job)
{
    JobID contact_job = ctx.submit([](Engine &ctx) {
        uint32_t num_candidates =
            ctx.sim().numCandidatePairs.load(std::memory_order_relaxed);

        if (num_candidates == 0) {
            return;
        }

        ctx.submitN([](Engine &ctx, uint32_t idx) {
            SimpleSim &sim = ctx.sim();
            const CandidatePair &c = sim.candidatePairs[idx];
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
        }, num_candidates);
    }, true, broadphase_job);
        
    // Once narrowphase is done, wipe CollisionCandidate table for next frame
    return ctx.submit([](Engine &ctx) {
        ctx.sim().numCandidatePairs.store(0, std::memory_order_relaxed);
    }, true, contact_job);
}

static JobID solverSystem(Engine &ctx, JobID narrowphase_job)
{
    return ctx.submit([](Engine &ctx) {
        SimpleSim &sim = ctx.sim();

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
    }, true, narrowphase_job);
}

static void tick(Engine &ctx)
{
    JobID broadphase_job = broadphaseSystem(ctx);
    JobID narrowphase_job = narrowphaseSystem(ctx, broadphase_job);

    solverSystem(ctx, narrowphase_job);
}

static void simLoop(Engine &ctx)
{
    ctx.submit([](Engine &ctx) {
        if (ctx.sim().tickCount % 10000 == 0) {
            printf("Tick start %" PRIu64 "\n", ctx.sim().tickCount);
        }

        tick(ctx);

        ctx.sim().tickCount += 1;

        // While this call appears recursive, all it does is immediately queue
        // up the simLoop dependency on the current job finishing.
        simLoop(ctx);
    }, /* Don't count this as a child job of the current job */ false,
    /* The next tick doesn't run until this tick is finished */
    ctx.currentJobID());
}

void SimpleSim::entry(Engine &ctx)
{
    SimpleSim &sim = ctx.sim();
    // Initialization
    new (&sim) SimpleSim(ctx);

    simLoop(ctx);
}

void SimpleSim::init(Engine &ctx)
{
    SimpleSim &sim = ctx.sim();
    // Initialization
    new (&sim) SimpleSim(ctx);
}

void SimpleSim::update(Engine &ctx)
{
    tick(ctx);
}

}
