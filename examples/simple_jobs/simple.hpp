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

struct ObjectInit {
    Translation initPosition;
    Rotation initRotation;
};

struct EnvInit {
    madrona::math::AABB worldBounds;
    uint64_t numBenchmarkTicks;
    ObjectInit *objsInit;
    uint32_t numObjs;
};

class Engine;

// Per-world state object (one per-world created by JobManager)
struct SimpleSim : public madrona::WorldBase {
    SimpleSim(Engine &ctx, const EnvInit &env_init);

    static void entry(Engine &ctx, const EnvInit &env_init);
    static void init(Engine &ctx, const EnvInit &env_init);
    static void update(Engine &ctx);

    bool benchmarkMode;
    uint64_t maxTicks;

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

class Engine : public::madrona::CustomContext<Engine, SimpleSim> {
public:
    using CustomContext::CustomContext;
    inline SimpleSim & sim() { return data(); }
};

}
