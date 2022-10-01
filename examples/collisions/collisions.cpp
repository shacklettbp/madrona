#include "collisions.hpp"

#include <cinttypes>
#include <random>

using namespace madrona;
using namespace madrona::math;

namespace CollisionExample {

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

CollisionSim::CollisionSim(Engine &ctx)
{
    // World attributes (constant for now)
    tickCount = 0;
    deltaT = 1.f / 60.f;

    worldBounds = {
        .pMin = Vector3 { -10, -10, 0, },
        .pMax = Vector3 { 10, 10, 10, },
    };

    // Register components
    ctx.registerComponent<Translation>();
    ctx.registerComponent<Rotation>();
    ctx.registerComponent<PhysicsAABB>();
    ctx.registerComponent<CandidatePair>();
    ctx.registerComponent<ContactData>();

    // Register archetypes
    ctx.registerArchetype<CubeObject>();
    ctx.registerArchetype<CollisionCandidate>();
    ctx.registerArchetype<Contact>();

    // Build and cache queries outside of sim loop
    physicsPreprocessQuery =
        ctx.query<const Translation, const Rotation, PhysicsAABB>();
    broadphaseQuery =
        ctx.query<const madrona::Entity, const PhysicsAABB>();
    candidateQuery =
        ctx.query<const CandidatePair>();

    const int init_num_objects = 100;

    // Entity creation / deletion is not thread safe (for now)
    std::uniform_real_distribution<float> angle_dist(0.f, M_PIf); 
    for (int i = 0; i < init_num_objects; i++) {
        Translation rand_pos = randomPosition(worldBounds);
        Rotation rand_rot = Quat::angleAxis(angle_dist(randGen()),
            Vector3 { 0, 1, 0 });

        PhysicsAABB aabb = AABB::invalid();

        ctx.makeEntityNow<CubeObject>(rand_pos, rand_rot, aabb);
    }
}

static JobID broadphaseSystem(Engine &ctx)
{
    // Update all entity bounding boxes:
    // FIXME: future improvement - sleeping entities for physics
    JobID preprocess = ctx.parallelFor(ctx.sim().physicsPreprocessQuery,
            [](Engine &, const Translation &translation,
               const Rotation &rotation, PhysicsAABB &aabb) {
        // No actual mesh, just hardcode a fake 2 *unit cube centered around
        // translation
        
        Mat3x4 model_mat = Mat3x4::fromTRS(translation, rotation);

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

        aabb = AABB::point(cube[0]);
        for (int i = 1; i < 8; i++) {
            aabb.expand(cube[i]);
        }
    });

    // Generate list of CollisionCandidates for narrowphase
    return ctx.parallelFor(ctx.sim().broadphaseQuery,
            [](Engine &ctx, Entity a, const PhysicsAABB &a_bbox) {
        // Note that capturing a_bbox by reference here may seem risky
        // but as long as entities matching the query aren't being created
        // deleted in parallel, the location of a_bbox won't change
        ctx.parallelFor(ctx.sim().broadphaseQuery,
                [a, &a_bbox](Engine &ctx, Entity b,
                             const PhysicsAABB &b_bbox) {
            if (a == b) {
                return;
            }

            if (a_bbox.overlaps(b_bbox)) {
                // No threadsafe way to create entities currently. Probably
                // will change
                std::lock_guard lock(ctx.sim().candidateCreateLock);
                ctx.makeEntityNow<CollisionCandidate>(CandidatePair { a, b });
            }
        });
    }, true, preprocess);
}

static JobID narrowphaseSystem(Engine &ctx, JobID broadphase_job)
{
    JobID contact_job = ctx.parallelFor(ctx.sim().candidateQuery, 
            [](Engine &ctx, const CandidatePair &pair) {
        // FIXME: Narrow phase is a no-op here, just passing data through to
        // the solver

        // Here we directly grab the Translation component on the two entities
        // in the CandidatePair. Note that in reality you'll need more data
        // than this, like the object / mesh itself, this is just an example.
        Translation a_pos = ctx.get<Translation>(pair.a).value();
        Translation b_pos = ctx.get<Translation>(pair.b).value();

        Vector3 to_b = (b_pos - a_pos).normalize();
        {
            // Same situation as above - will have better solution here
            std::lock_guard lock(ctx.sim().contactCreateLock);
            ctx.makeEntityNow<Contact>(ContactData {
                to_b,
                pair.a,
                pair.b,
            });
        }
    }, true, broadphase_job);

    // Once narrowphase is done, wipe CollisionCandidate table for next frame
    return ctx.submit([](Engine &ctx) {
        ctx.clearArchetype<CollisionCandidate>();
    }, true, contact_job);
}

static JobID solverSystem(Engine &ctx, JobID narrowphase_job)
{
    return ctx.submit([](Engine &ctx) {
        // Push objects in serial based on the contact normal - total BS.
        auto contacts = ctx.archetype<Contact>();
        int num_contacts = (int)contacts.size();
        ContactData *contacts_data = contacts.component<ContactData>().data();

        for (int i = 0; i < num_contacts; i++) {
            ContactData &contact = contacts_data[i];

            Translation &a_pos = ctx.get<Translation>(contact.a).value();
            Translation &b_pos = ctx.get<Translation>(contact.b).value();

            a_pos -= contact.normal;
            b_pos += contact.normal;
        }

        ctx.clearArchetype<Contact>();
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
        if (ctx.sim().tickCount % 1 == 0) {
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

void CollisionSim::entry(Engine &ctx)
{
    CollisionSim &sim = ctx.sim();
    // Initialization
    new (&sim) CollisionSim(ctx);

    simLoop(ctx);
}

}
