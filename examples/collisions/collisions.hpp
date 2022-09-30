#pragma once

#include <madrona/custom_context.hpp>
#include <madrona/math.hpp>

namespace CollisionExample {

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

struct PhysicsAABB : madrona::math::AABB {
    PhysicsAABB(madrona::math::AABB b)
        : AABB(b)
    {}
};

struct CandidatePair {
    madrona::Entity a;
    madrona::Entity b;
};

struct ContactData {
    madrona::math::Vector3 normal;
    madrona::Entity a;
    madrona::Entity b;
};

// Archetypes
struct CubeObject : madrona::Archetype<Translation, Rotation, PhysicsAABB> {};
struct CollisionCandidate : madrona::Archetype<CandidatePair> {};
struct Contact : madrona::Archetype<ContactData> {};

class Engine;

// Per-world state object (one per-world created by JobManager)
struct CollisionSim {
    CollisionSim(Engine &ctx);

    static void entry(Engine &ctx);

    uint64_t tickCount;
    float deltaT;

    madrona::math::AABB worldBounds;

    madrona::Query<const Translation, const Rotation, PhysicsAABB>
        physicsPreprocessQuery;
    madrona::Query<const madrona::Entity, const PhysicsAABB> broadphaseQuery;
    madrona::Query<const CandidatePair> candidateQuery;

    madrona::utils::SpinLock candidateCreateLock {};
    madrona::utils::SpinLock contactCreateLock {};
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
