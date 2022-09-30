#include "fvs.hpp"

#include <cinttypes>

using namespace madrona;
using namespace madrona::math;

namespace fvs {

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

Game::Game(Engine &ctx, const BenchmarkConfig &bench)
{
    // World attributes (constant for now)
    tickCount = 0;
    deltaT = 1.f / 60.f;
    moveSpeed = 0.1f; // Move speed in m / s
    manaRegenRate = 1.f;
    castTime = 2.f;
    shootTime = 0.5f;

    worldBounds = {
        .pMin = Vector3 { -10, -10, 0, },
        .pMax = Vector3 { 10, 10, 10, },
    };

    // Must register all components
    ctx.registerComponent<Position>();
    ctx.registerComponent<Health>();
    ctx.registerComponent<Action>();
    ctx.registerComponent<Mana>();
    ctx.registerComponent<Quiver>();
    ctx.registerComponent<CleanupEntity>();

    // Must register all archetypes
    ctx.registerArchetype<Dragon>();
    ctx.registerArchetype<Knight>();
    ctx.registerArchetype<CleanupTracker>();

    // Queries should be built ahead of time and cached, like this.
    actionQuery = ctx.query<Position, Action>();
    healthQuery = ctx.query<Position, Health>();
    casterQuery = ctx.query<Action, Mana>();
    archerQuery = ctx.query<Action, Quiver>();
    cleanupQuery = ctx.query<Entity, Health>();

    int init_num_dragons;
    int init_num_knights;

    if (bench.enable) {
        init_num_dragons = bench.numDragons;
        init_num_knights = bench.numKnights;
    } else {
        init_num_dragons = 10;
        init_num_knights = 20;
    }

    const int dragon_hp = 1000;
    const int knight_hp = 100;

    std::uniform_real_distribution<float> mp_dist(0.f, 50.f);
    std::uniform_int_distribution<int> arrows_dist(20, 40);

    // Entity creation / deletion is not thread safe
    for (int i = 0; i < init_num_dragons; i++) {
        Position rand_pos = randomPosition(worldBounds);
        Health health { dragon_hp };
        Action act { 0 };
        Mana mp { mp_dist(randGen()) };

        ctx.makeEntityNow<Dragon>(rand_pos, health, act, mp);
    }

    for (int i = 0; i < init_num_knights; i++) {
        Position rand_pos = randomPosition(worldBounds);
        Health health { knight_hp };
        Action act { 0 };
        Quiver q { arrows_dist(randGen()) };

        ctx.makeEntityNow<Knight>(rand_pos, health, act, q);
    }
}

static JobID actionSelectSystem(Engine &ctx)
{
    return ctx.parallelFor(ctx.game().actionQuery, [](Engine &ctx,
                                                      Position &pos,
                                                      Action &action) {
        const Game &game = ctx.game();

        if (action.remainingTime > 0) {
            action.remainingTime -= game.deltaT;
            return;
        }
        
        std::uniform_real_distribution<float> action_prob(0.f, 1.f);

        float move_cutoff = 0.5f;

        if (action_prob(randGen()) <= move_cutoff) {
            ctx.submit([&pos, &action](Engine &ctx) {
                const AABB &world_bounds = ctx.game().worldBounds;

                // Move
                std::uniform_real_distribution<float> pos_dist(-1.f, 1.f);

                Vector3 new_pos = pos + Vector3 {
                    pos_dist(randGen()),
                    pos_dist(randGen()),
                    pos_dist(randGen()),
                };

                new_pos.x = std::clamp(new_pos.x, world_bounds.pMin.x, world_bounds.pMax.x);
                new_pos.y = std::clamp(new_pos.y, world_bounds.pMin.y, world_bounds.pMax.y);
                new_pos.z = std::clamp(new_pos.x, world_bounds.pMin.z, world_bounds.pMax.z);

                Vector3 pos_delta = new_pos - pos;
                pos = new_pos;

                action.remainingTime = pos_delta.length() / ctx.game().moveSpeed;
            });
        } 
    });
}

static JobID casterSystem(Engine &ctx, JobID action_job)
{
    return ctx.parallelFor(ctx.game().casterQuery, [](Engine &ctx,
                                                      Action &action,
                                                      Mana &mana) {
        const Game &game = ctx.game();

        mana.mp += game.manaRegenRate * game.deltaT;

        if (action.remainingTime > 0) {
            return;
        }
        // move job runs first so if remainingTime == 0, always act (otherwise would do nothing)

        const float cast_cost = 20.f;

        if (mana.mp < cast_cost) {
            return;
        }

        mana.mp -= cast_cost;

        auto target_pos = randomPosition(game.worldBounds);

        ctx.parallelFor(game.healthQuery, [target_pos](Engine &,
                                                       const Position &pos,
                                                       Health &health) {
            const float blast_radius = 2.f;
            const float damage = 20.f;

            if (target_pos.distance(pos) <= blast_radius) {
                health.hp -= damage;
            }
        });

        action.remainingTime = game.castTime;
    }, true, action_job);
}

static JobID archerSystem(Engine &ctx, JobID action_job)
{
    return ctx.parallelFor(ctx.game().archerQuery, [](Engine &ctx,
                                                      Action &action,
                                                      Quiver &quiver) {
        if (action.remainingTime > 0 || quiver.numArrows == 0) {
            return;
        }

        auto dragons = ctx.archetype<Dragon>();
        uint32_t num_dragons = dragons.size();

        std::uniform_int_distribution<uint32_t> dragon_sel(0, num_dragons - 1);

        uint32_t dragon_idx = dragon_sel(randGen());
        Health &dragon_health = dragons.get<Health>(dragon_idx);

        const float damage = 15.f;
        dragon_health.hp -= damage;

        quiver.numArrows -= 1;
        action.remainingTime = ctx.game().shootTime;
    }, true, action_job);
}

void Game::tick(Engine &ctx)
{
    JobID init_action_job = actionSelectSystem(ctx);

    JobID cast_job = casterSystem(ctx, init_action_job);

    JobID archer_job = archerSystem(ctx, init_action_job);

    ctx.submit([this](Engine &ctx) {
        ctx.forEach(cleanupQuery, [&ctx](Entity e, Health &health) {
            if (health.hp <= 0) {
                ctx.makeEntityNow<CleanupTracker>(CleanupEntity(e));
            }
        });

        auto cleanup_tracker = ctx.archetype<CleanupTracker>();
        auto cleanup_entities = cleanup_tracker.component<CleanupEntity>();
        for (int i = 0, n = cleanup_tracker.size(); i < n; i++) {
            ctx.destroyEntityNow(cleanup_entities[i]);
        }

        ctx.clearArchetype<CleanupTracker>();
    }, true, cast_job, archer_job);
}

void Game::gameLoop(Engine &ctx)
{
    ctx.submit([this](Engine &ctx) {
        auto dragons = ctx.archetype<Dragon>();
        auto knights = ctx.archetype<Knight>();

        if (dragons.size() == 0) {
            printf("Knights win!\n");
            return;
        }

        if (knights.size() == 0) {
            printf("Dragons win!\n");
            return;
        }

        if (tickCount % 10000 == 0) {
            printf("Tick start %" PRIu64 "\n", tickCount);
        }
        tick(ctx);

        tickCount += 1;

        // While this call appears recursive, all it does is immediately queue
        // up the gameloop job again with a dependency on the current job
        // finishing.
        gameLoop(ctx);
    }, /* Don't count this as a child job of the current job */ false, ctx.currentJobID());
}

void Game::benchmarkTick(Engine &ctx)
{
    JobID init_action_job = actionSelectSystem(ctx);
    casterSystem(ctx, init_action_job);
    archerSystem(ctx, init_action_job);
}

void Game::benchmark(Engine &ctx, const BenchmarkConfig &bench)
{
    ctx.submit([this, &bench](Engine &ctx) {
        if (tickCount == bench.numTicks) {
            return;
        }

        benchmarkTick(ctx);

        tickCount++;

        benchmark(ctx, bench);
    }, false, ctx.currentJobID());
}

void Game::entry(Engine &ctx, const BenchmarkConfig &bench)
{
    Game &game = ctx.game();
    // Initialization
    new (&game) Game(ctx, bench);

    // Start game loop
    if (bench.enable) {
        game.benchmark(ctx, bench);
    } else {
        game.gameLoop(ctx);
    }
}

}
