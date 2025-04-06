#pragma once

#include <madrona/sync.hpp>
#include <madrona/physics.hpp>

#ifdef MADRONA_GPU_MODE
#define CV_COUNT_GPU_CLOCKS
#endif



#ifdef CV_COUNT_GPU_CLOCKS
extern "C" {
extern madrona::AtomicU64 cvcom;
extern madrona::AtomicU64 cvinertias;
extern madrona::AtomicU64 cvrne;
extern madrona::AtomicU64 cvcrb;
extern madrona::AtomicU64 cvinvMass;
extern madrona::AtomicU64 cvprocessContacts;
extern madrona::AtomicU64 cvconvert;
extern madrona::AtomicU64 cvdestroy;
extern madrona::AtomicU64 cvinit;
extern madrona::AtomicU64 cvintg;
extern madrona::AtomicU64 cvfk;
extern madrona::AtomicU64 cvnarrowphase;
extern madrona::AtomicU64 cvallocScratch;
extern madrona::AtomicU64 cvprepSolver;
extern madrona::AtomicU64 cvcontAccRef;
extern madrona::AtomicU64 cveqAccRef;
extern madrona::AtomicU64 cvcg;
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
};

struct CandidateTemporary : Archetype<CandidateCollision> {};

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
    constexpr inline CountT SolverBase = 11;

    constexpr inline CountT CandidateCollision = 2;
    constexpr inline CountT ContactConstraint = 2;
    constexpr inline CountT JointConstraint = 2;
};

}
