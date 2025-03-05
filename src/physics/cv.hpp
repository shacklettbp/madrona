#pragma once

#include <madrona/cv_physics.hpp>

#define ASSERT_PTR_ACCESS(a, offset, b) assert((uint8_t *)(a + offset) < (uint8_t *)b)

#ifdef MADRONA_GPU_MODE
#define MADRONA_GPU_SINGLE_THREAD if (threadIdx.x % 32 == 0)
#else
#define MADRONA_GPU_SINGLE_THREAD
#endif

#ifdef MADRONA_GPU_MODE
#define MADRONA_SYNCWARP() __syncwarp()
#else
#define MADRONA_SYNCWARP()
#endif

namespace madrona::phys::cv {

struct MRElement128b {
    uint8_t d[128];
};

struct SolverScratch256b {
    uint8_t d[256];
};

struct Contact : Archetype<
    ContactConstraint,
    ContactTmpState
> {};

struct Joint : Archetype<
    JointConstraint
> {};
    
// All implementation stuff goes here
StateManager * getStateManager(Context &ctx);
StateManager * getStateManager();

template <typename ArchetypeT, typename ComponentT>
inline ComponentT * getRows(StateManager *state_mgr, uint32_t world_id);

void getSolverArchetypeIDs(uint32_t *contact_archetype_id,
                           uint32_t *joint_archetype_id);

// All task setup
TaskGraphNodeID setupPrepareTasks(TaskGraphBuilder &builder,
                                  TaskGraphNodeID narrowphase);

TaskGraphNodeID setupSolveTasks(TaskGraphBuilder &builder,
                                TaskGraphNodeID prepare);

TaskGraphNodeID setupPostTasks(TaskGraphBuilder &builder,
                               TaskGraphNodeID solve);

namespace tasks {
void refreshPointers(Context &ctx,
                     BodyGroupMemory &m);
void computeExpandedParent(Context &ctx,
                           BodyGroupMemory m,
                           BodyGroupProperties p);
// This task is required by a couple different files
void forwardKinematics(Context &ctx,
                       BodyGroupMemory m,
                       BodyGroupProperties p);
void computeGroupCOM(Context &ctx,
                     BodyGroupProperties &prop,
                     BodyGroupMemory &mem);
void computeSpatialInertiasAndPhi(Context &ctx,
                                  DofObjectGroup obj_grp);
void compositeRigidBody(Context &ctx,
                        BodyGroupProperties p,
                        BodyGroupMemory m);
void computePhi(DofType dof_type,
                BodyPhi& body_phi,
                float* S,
                math::Vector3 origin);
void computePhiTrans(DofType dof_type,
                     BodyPhi &body_phi,
                     math::Vector3 origin,
                     float (&S)[18]);
void solveM(
        BodyGroupProperties prop,
        BodyGroupMemory mem, 
        float* x);
void convertPostSolve(
        Context &ctx,
        base::Position &position,
        base::Rotation &rotation,
        base::Scale &scale,
        LinkParentDofObject &link);
}

struct SparseBlkDiag {
    struct Blk {
        uint32_t dim;
        uint32_t scratch;
        float *values;

        // factorized version
        float *ltdl;
        int32_t *expandedParent;
    };

    uint32_t fullDim;
    uint32_t numBlks;
    Blk *blks;
};

struct CVSolveData {
    uint32_t enablePhysics;

    uint32_t numBodyGroups;
    uint32_t *dofOffsets;

    uint32_t totalNumDofs;
    uint32_t numContactPts;
    float h;

    // Values
    float *mass;
    float *freeAcc;
    float *vel;
    float *J_c;
    float *J_e;
    float *mu;
    float *penetrations;
    float *eqResiduals;
    // Diagonal approximations of A = J * M^-1 * J^T
    float *diagApprox_c;
    float *diagApprox_e;

    uint32_t massDim;
    uint32_t freeAccDim;
    uint32_t velDim;

    uint32_t numRowsJc;
    uint32_t numColsJc;

    uint32_t numRowsJe;
    uint32_t numColsJe;

    uint32_t muDim;
    uint32_t penetrationsDim;

    // Sum of diagonals of mass matrix
    float totalMass;

    enum StateFlags {
        // Is a_ref stored in shared memory?
        ARefSmem = 1 << 0
    };

    uint32_t flags;

    SparseBlkDiag massSparse;

    CVXSolve *cvxSolve;

    uint8_t *solverScratchMem;
    uint8_t *accRefMem;
    uint8_t *prepMem;
    uint32_t scratchAllocatedBytes;
    uint32_t accRefAllocatedBytes;
    uint32_t prepAllocatedBytes;

    static constexpr uint32_t kNumRegisters = 9;

    struct RegInfo {
        uint64_t size;
        bool inSmem;
        void *ptr;
    };

    RegInfo regInfos[kNumRegisters];

    bool reset;

#ifdef MADRONA_GPU_MODE
    inline float * getContactAccRef(StateManager *state_mgr)
    {
        return (float *)accRefMem;
    }

    inline float * getEqualityAccRef(StateManager *state_mgr)
    {
        return (float *)accRefMem + numRowsJc;
    }

    inline SparseBlkDiag::Blk * getMassBlks(StateManager *state_mgr)
    {
        (void)state_mgr;
        uint8_t *bytes =
            //(uint8_t *)state_mgr->memoryRangePointer<SolverScratch256b>(prepMemory);
            prepMem;
        return (SparseBlkDiag::Blk *)bytes;
    }

    inline float * getFullVel(StateManager *state_mgr)
    {
        (void)state_mgr;
        uint8_t *bytes =
            //(uint8_t *)state_mgr->memoryRangePointer<SolverScratch256b>(prepMemory) +
            prepMem +
            sizeof(SparseBlkDiag::Blk) * numBodyGroups;
        return (float *)bytes;
    }

    inline float * getFreeAcc(StateManager *state_mgr)
    {
        (void)state_mgr;
        uint8_t *bytes =
            // (uint8_t *)state_mgr->memoryRangePointer<SolverScratch256b>(prepMemory) +
            prepMem +
            sizeof(SparseBlkDiag::Blk) * numBodyGroups +
            sizeof(float) * totalNumDofs;
        return (float *)bytes;
    }

    inline float * getMu(StateManager *state_mgr)
    {
        (void)state_mgr;
        uint8_t *bytes =
            // (uint8_t *)state_mgr->memoryRangePointer<SolverScratch256b>(prepMemory) +
            prepMem +
            sizeof(SparseBlkDiag::Blk) * numBodyGroups +
            sizeof(float) * totalNumDofs +
            sizeof(float) * totalNumDofs;
        return (float *)bytes;
    }

    inline float * getPenetrations(StateManager *state_mgr)
    {
        (void)state_mgr;
        uint8_t *bytes =
            //(uint8_t *)state_mgr->memoryRangePointer<SolverScratch256b>(prepMemory) +
            prepMem +
            sizeof(SparseBlkDiag::Blk) * numBodyGroups +
            sizeof(float) * totalNumDofs +
            sizeof(float) * totalNumDofs +
            sizeof(float) * numContactPts;
        return (float *)bytes;
    }

    inline float * getContactJacobian(StateManager *state_mgr)
    {
        (void)state_mgr;
        uint8_t *bytes =
            // (uint8_t *)state_mgr->memoryRangePointer<SolverScratch256b>(prepMemory) +
            prepMem +
            sizeof(SparseBlkDiag::Blk) * numBodyGroups +
            sizeof(float) * totalNumDofs +
            sizeof(float) * totalNumDofs +
            sizeof(float) * numContactPts +
            sizeof(float) * numContactPts;
        return (float *)bytes;
    }

    inline float * getContactDiagApprox(StateManager *state_mgr)
    {
        (void)state_mgr;
        uint8_t *bytes =
            // (uint8_t *)state_mgr->memoryRangePointer<SolverScratch256b>(prepMemory) +
            prepMem +
            sizeof(SparseBlkDiag::Blk) * numBodyGroups +
            sizeof(float) * totalNumDofs +
            sizeof(float) * totalNumDofs +
            sizeof(float) * numContactPts +
            sizeof(float) * numContactPts +
            sizeof(float) * numRowsJc * numColsJc;
        return (float *)bytes;
    }

    inline float * getEqualityJacobian(StateManager *state_mgr)
    {
        (void)state_mgr;
        uint8_t *bytes =
            // (uint8_t *)state_mgr->memoryRangePointer<SolverScratch256b>(prepMemory) +
            prepMem +
            sizeof(SparseBlkDiag::Blk) * numBodyGroups +
            sizeof(float) * totalNumDofs +
            sizeof(float) * totalNumDofs +
            sizeof(float) * numContactPts +
            sizeof(float) * numContactPts +
            sizeof(float) * numRowsJc * numColsJc +
            sizeof(float) * numRowsJc;
        return (float *)bytes;
    }

    inline float * getEqualityDiagApprox(StateManager *state_mgr)
    {
        (void)state_mgr;
        uint8_t *bytes =
            // (uint8_t *)state_mgr->memoryRangePointer<SolverScratch256b>(prepMemory) +
            prepMem +
            sizeof(SparseBlkDiag::Blk) * numBodyGroups +
            sizeof(float) * totalNumDofs +
            sizeof(float) * totalNumDofs +
            sizeof(float) * numContactPts +
            sizeof(float) * numContactPts +
            sizeof(float) * numRowsJc * numColsJc +
            sizeof(float) * numRowsJc +
            sizeof(float) * numRowsJe * numColsJe;
        return (float *)bytes;
    }

    inline float * getEqualityResiduals(StateManager *state_mgr)
    {
        (void)state_mgr;
        uint8_t *bytes =
            // (uint8_t *)state_mgr->memoryRangePointer<SolverScratch256b>(prepMemory) +
            prepMem +
            sizeof(SparseBlkDiag::Blk) * numBodyGroups +
            sizeof(float) * totalNumDofs +
            sizeof(float) * totalNumDofs +
            sizeof(float) * numContactPts +
            sizeof(float) * numContactPts +
            sizeof(float) * numRowsJc * numColsJc +
            sizeof(float) * numRowsJc +
            sizeof(float) * numRowsJe * numColsJe +
            sizeof(float) * numRowsJe;
        return (float *)bytes;
    }
#endif
};

}

#include "cv.inl"
