/*
 * Copyright 2021-2022 Brennan Shacklett and contributors
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 */
#pragma once

#include <madrona/custom_context.hpp>
#include <madrona/math.hpp>

namespace SimpleExample {

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
    uint32_t a;
    uint32_t b;
};

struct ContactData {
    madrona::math::Vector3 normal;
    uint32_t a;
    uint32_t b;
};

struct SphereObject {
    Translation translation;
    Rotation rotation;
    PhysicsAABB aabb;
};

class Engine;

// Per-world state object (one per-world created by JobManager)
struct SimpleSim {
    SimpleSim(Engine &ctx);

    static void entry(Engine &ctx);
    static void init(Engine &ctx);
    static void update(Engine &ctx);

    uint64_t tickCount;
    float deltaT;

    madrona::math::AABB worldBounds;

    SphereObject *sphereObjects;
    CandidatePair *candidatePairs;
    ContactData *contacts;
    uint32_t numSphereObjects;
    std::atomic_uint32_t numCandidatePairs;
    std::atomic_uint32_t numContacts;

    madrona::utils::SpinLock candidateCreateLock {};
    madrona::utils::SpinLock contactCreateLock {};
};

// madrona::Context subclass, allows easy access to per-world state through
// game() method
class Engine : public::madrona::CustomContext<Engine> {
public:
    inline Engine(SimpleSim *sim, madrona::WorkerInit &&init)
        : madrona::CustomContext<Engine>(std::move(init)),
          sim_(sim)
    {}

    inline SimpleSim & sim() { return *sim_; }

private:
    SimpleSim *sim_;
};

}
