#pragma once

#include <madrona/physics.hpp>

#ifdef MADRONA_GPU_MODE
// #define CV_COUNT_GPU_CLOCKS
#endif



#ifdef CV_COUNT_GPU_CLOCKS
extern "C" {
struct CVClocks {
    AtomicU64 com;
    AtomicU64 interias;

    // RNE with just G
    AtomicU64 rneG;
    // RNE with full acceleration as input
    AtomicU64 rne;

    AtomicU64 crb;
    AtomicU64 invMass;

    AtomicU64 processContacts;

    AtomicU64 convert;

    AtomicU64 destroy;
    AtomicU64 init;

    // Integration
    AtomicU64 intg;

    AtomicU64 fk;

    AtomicU64 narrowphase;

    AtomicU64 allocScratch;
    AtomicU64 prepSolver;
    AtomicU64 contAccRef;
    AtomicU64 eqAccRef;
    AtomicU64 cg;
};

extern CVClocks cvClocks;
}

class CVClockHelper {
public:
    inline CVClockHelper(AtomicU64 &counter)
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

    AtomicU64 *counter_;
    uint64_t start_;
};

#define CV_PROF_START(name, counter) \
    CVClockHelper name(cvClocks. counter)

#define CV_PROF_END(name) name.end()
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
