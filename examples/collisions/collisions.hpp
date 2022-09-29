#pragma once

#include <madrona/custom_context.hpp>
#include <madrona/geometry.hpp>

namespace CollisionExample {

// Components
struct Position : madrona::Vector3 {
    Position(Vector3 v)
        : Vector3(v)
    {}
};

struct Rotation : madrona::Quat {
    Rotation(Quat q)
        : Quat(q)
    {}
};

struct PhysicsState {
    madrona::Vector3 velocity;
};

// Archetypes
struct Object : madrona::Archetype<Position, Health, Action, Mana> {};
struct CleanupTracker : madrona::Archetype<CleanupEntity> {};

class Engine;

struct BenchmarkConfig {
    bool enable;
    uint32_t numTicks;
    uint32_t numKnights;
    uint32_t numDragons;
};

// Per-world state object (one per-world created by JobManager)
struct CollisionSim {
    static void entry(Engine &ctx, const BenchmarkConfig &bench);

    Game(Engine &ctx, const BenchmarkConfig &bench);
    void tick(Engine &ctx);
    void simLoop(Engine &ctx);

    uint64_t tickCount;
    float deltaT;

    madrona::AABB worldBounds;

    madrona::Query<Position, Action> physicsQuery;
};

// madrona::Context subclass, allows easy access to per-world state through
// game() method
class Engine : public::madrona::CustomContext<Engine> {
public:
    inline Engine(CollisionSim *sim, madrona::WorkerInit &&init);

    inline CollisionSim & sim() { return *sim_; }

private:
    CollisionSim *sim_;
};

}

#include "collisions.inl"
