#pragma once

#include <madrona/sync.hpp>
#include <madrona/physics.hpp>

#ifdef MADRONA_GPU_MODE
#define CV_COUNT_GPU_CLOCKS
#endif



#ifdef CV_COUNT_GPU_CLOCKS
#define DECLARE_STAGE_VARS(name) extern madrona::AtomicU64 cv##name; \
                                 extern uint64_t cv##name##_avg; \
                                 extern double cv##name##_min; \
                                 extern double cv##name##_max;
#define DEFINE_STAGE_VARS(name) madrona::AtomicU64 cv##name = 0; \
                                uint64_t cv##name##_avg = 0; \
                                double cv##name##_min = 99999.0; \
                                double cv##name##_max = 0.0;

extern "C" {
DECLARE_STAGE_VARS(com);
DECLARE_STAGE_VARS(inertias);
DECLARE_STAGE_VARS(rne);
DECLARE_STAGE_VARS(crb);
DECLARE_STAGE_VARS(invMass);
DECLARE_STAGE_VARS(processContacts);
DECLARE_STAGE_VARS(convert);
DECLARE_STAGE_VARS(destroy);
DECLARE_STAGE_VARS(init);
DECLARE_STAGE_VARS(damp);
DECLARE_STAGE_VARS(intg);
DECLARE_STAGE_VARS(fk);
DECLARE_STAGE_VARS(narrowphase);
DECLARE_STAGE_VARS(broadphase1);
DECLARE_STAGE_VARS(broadphase2);
DECLARE_STAGE_VARS(allocScratch);
DECLARE_STAGE_VARS(prepSolver);
DECLARE_STAGE_VARS(contAccRef);
DECLARE_STAGE_VARS(eqAccRef);
DECLARE_STAGE_VARS(cg);
DECLARE_STAGE_VARS(lineSearch);
}

class CVClockHelper {
public:
    inline CVClockHelper(madrona::AtomicU64 &counter)
        : counter_(&counter)
    {
        cuda::atomic_thread_fence(cuda::memory_order_seq_cst,
                                  cuda::thread_scope_thread);
        start_ = timestamp();
    }

    inline void end()
    {
        cuda::atomic_thread_fence(cuda::memory_order_seq_cst,
                                  cuda::thread_scope_thread);
        auto end = timestamp();
        counter_->fetch_add_relaxed(end - start_);
        counter_ = nullptr;
    }

    inline ~CVClockHelper()
    {
        if (counter_ != nullptr) {
            end();
        }
    }

private:
    inline uint64_t timestamp() const
    {
        uint64_t v;
        asm volatile("mov.u64 %0, %%globaltimer;"
                     : "=l"(v));
        return v;
    }

    madrona::AtomicU64 *counter_;
    uint64_t start_;
};

#define CV_PROF_START(name, counter) \
    CVClockHelper name(cv##counter);

#define CV_PROF_END(name) name.end();
#else
#define CV_PROF_START(name, counter)
#define CV_PROF_END(name)
#endif

namespace madrona::phys {

struct PhysicsSystemState {
    float deltaT;
    float h;
    math::Vector3 g;
    float gMagnitude;
    float restitutionThreshold;
    uint32_t contactArchetypeID;
    uint32_t jointArchetypeID;
    bool createRenderObjects;

    uint64_t cvNumFrames;
    uint64_t cvNumInitFrames;
};

struct CandidateTemporary : Archetype<CandidateCollision> {};

// This is going to create the actual CandidateCollisions
struct BroadphaseObjectTemporary {
    uint32_t offset;
    uint32_t count;

    Loc a;
    Loc b;

    math::Vector3 aPos;
    math::Vector3 bPos;
    math::Quat aRot;
    math::Quat bRot;
    math::Diag3x3 aScale;
    math::Diag3x3 bScale;
};

struct BroadphaseTemporary : Archetype<BroadphaseObjectTemporary> {};

namespace broadphase {

TaskGraphNodeID setupBVHTasks(
    TaskGraphBuilder &builder,
    Span<const TaskGraphNodeID> deps);

TaskGraphNodeID setupPreIntegrationTasks(
    TaskGraphBuilder &builder,
    Span<const TaskGraphNodeID> deps);

TaskGraphNodeID setupPostIntegrationTasks(
    TaskGraphBuilder &builder,
    Span<const TaskGraphNodeID> deps);

}

namespace narrowphase {

TaskGraphNodeID setupTasks(
    TaskGraphBuilder &builder,
    Span<const TaskGraphNodeID> deps);

}

namespace RGDCols {
    constexpr inline CountT Position = 2;
    constexpr inline CountT Rotation = 3;
    constexpr inline CountT Scale = 4;
    constexpr inline CountT ObjectID = 5;
    constexpr inline CountT ResponseType = 6;
    constexpr inline CountT LeafID = 7;
    constexpr inline CountT Velocity = 8;
    constexpr inline CountT ExternalForce = 9;
    constexpr inline CountT ExternalTorque = 10;
    constexpr inline CountT DisableColliders = 11;
    constexpr inline CountT RigidBodyStatic = 12;
    constexpr inline CountT SolverBase = 13;

    constexpr inline CountT BroadphaseObjectTemporary = 2;
    constexpr inline CountT CandidateCollision = 2;
    constexpr inline CountT ContactConstraint = 2;
    constexpr inline CountT JointConstraint = 2;
};

}
