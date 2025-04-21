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

#define MINVAL    1E-15f

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
                               TaskGraphNodeID solve,
                               bool replay_mode = false);

namespace tasks {
void initHierarchies(Context &ctx,
                     InitBodyGroup body_grp);
void destroyHierarchies(Context &ctx,
                        DestroyBodyGroup &body_grp);
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
                        BodyGroupProperties &p,
                        BodyGroupMemory &m);

void computePhi(DofType dof_type,
                BodyPhi& body_phi,
                math::Vector3 origin,
                float* S);
void computePhiTrans(DofType dof_type,
                     BodyPhi &body_phi,
                     math::Vector3 origin,
                     float *S);

void factorM(BodyGroupProperties prop,
             BodyGroupMemory mem);
void mulM(BodyGroupProperties prop,
        BodyGroupMemory mem,
        float *x, float *y);
void solveM(
        BodyGroupProperties prop,
        BodyGroupMemory mem, 
        float* x);
void convertPostSolve(
        Context &ctx,
        Entity e,
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
    uint32_t worldID;
    uint32_t enablePhysics;

    uint32_t numBodyGroups;
    uint32_t *dofOffsets;
    float h;

    // Total number of dofs, limit, contact, friction constraints
    uint32_t nv;
    uint32_t nl;
    uint32_t nc;
    uint32_t nf;

    // Values
    float *mass;
    float *freeAcc;
    float *vel;
    float *currAcc;
    float *J_c;
    float *J_l;
    float *J_f;
    float *mu;
    float *penetrations;
    float *limitResiduals;
    // Diagonal approximations of A = J * M^-1 * J^T
    float *diagApprox_c;
    float *diagApprox_l;
    float *diagApprox_f;
    float *floss;

    uint32_t muDim;
    uint32_t penetrationsDim;

    // Sum of diagonals of mass matrix
    float massDiagSum;

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

#ifdef MADRONA_GPU_MODE
    inline float * getContactAccRef(StateManager *state_mgr)
    {
        return (float *)accRefMem;
    }

    inline float * getLimitAccRef(StateManager *state_mgr)
    {
        return (float *)accRefMem + nc;
    }

    inline SparseBlkDiag::Blk * getMassBlks(StateManager *state_mgr)
    {
        (void)state_mgr;
        uint8_t *bytes =
            prepMem;
        return (SparseBlkDiag::Blk *)bytes;
    }

    inline float * getFullVel(StateManager *state_mgr)
    {
        (void)state_mgr;
        uint8_t *bytes =
            prepMem +
            sizeof(SparseBlkDiag::Blk) * numBodyGroups;
        return (float *)bytes;
    }

    inline float * getFreeAcc(StateManager *state_mgr)
    {
        (void)state_mgr;
        uint8_t *bytes =
            prepMem +
            sizeof(SparseBlkDiag::Blk) * numBodyGroups +
            sizeof(float) * nv;
        return (float *)bytes;
    }

    inline float * getMu(StateManager *state_mgr)
    {
        (void)state_mgr;
        uint8_t *bytes =
            prepMem +
            sizeof(SparseBlkDiag::Blk) * numBodyGroups +
            sizeof(float) * nv +
            sizeof(float) * nv;
        return (float *)bytes;
    }

    inline float * getPenetrations(StateManager *state_mgr)
    {
        (void)state_mgr;
        uint8_t *bytes =
            prepMem +
            sizeof(SparseBlkDiag::Blk) * numBodyGroups +
            sizeof(float) * nv +
            sizeof(float) * nv +
            sizeof(float) * nc;
        return (float *)bytes;
    }

    inline float * getContactJacobian(StateManager *state_mgr)
    {
        (void)state_mgr;
        uint8_t *bytes =
            prepMem +
            sizeof(SparseBlkDiag::Blk) * numBodyGroups +
            sizeof(float) * nv +
            sizeof(float) * nv +
            sizeof(float) * nc +
            sizeof(float) * nc;
        return (float *)bytes;
    }

    inline float * getContactDiagApprox(StateManager *state_mgr)
    {
        (void)state_mgr;
        uint8_t *bytes =
            prepMem +
            sizeof(SparseBlkDiag::Blk) * numBodyGroups +
            sizeof(float) * nv +
            sizeof(float) * nv +
            sizeof(float) * nc +
            sizeof(float) * nc +
            sizeof(float) * nc * nv;
        return (float *)bytes;
    }

    inline float * getLimitJacobian(StateManager *state_mgr)
    {
        (void)state_mgr;
        uint8_t *bytes =
            prepMem +
            sizeof(SparseBlkDiag::Blk) * numBodyGroups +
            sizeof(float) * nv +
            sizeof(float) * nv +
            sizeof(float) * nc +
            sizeof(float) * nc +
            sizeof(float) * nc * nv +
            sizeof(float) * nc;
        return (float *)bytes;
    }

    inline float * getEqualityDiagApprox(StateManager *state_mgr)
    {
        (void)state_mgr;
        uint8_t *bytes =
            prepMem +
            sizeof(SparseBlkDiag::Blk) * numBodyGroups +
            sizeof(float) * nv +
            sizeof(float) * nv +
            sizeof(float) * nc +
            sizeof(float) * nc +
            sizeof(float) * nc * nv +
            sizeof(float) * nc +
            sizeof(float) * nl * nv;
        return (float *)bytes;
    }

    inline float * getEqualityResiduals(StateManager *state_mgr)
    {
        (void)state_mgr;
        uint8_t *bytes =
            prepMem +
            sizeof(SparseBlkDiag::Blk) * numBodyGroups +
            sizeof(float) * nv +
            sizeof(float) * nv +
            sizeof(float) * nc +
            sizeof(float) * nc +
            sizeof(float) * nc * nv +
            sizeof(float) * nc +
            sizeof(float) * nl * nv +
            sizeof(float) * nl;
        return (float *)bytes;
    }

    inline float * getContactR(StateManager *state_mgr)
    {
        (void)state_mgr;
        uint8_t *bytes =
            prepMem +
            sizeof(SparseBlkDiag::Blk) * numBodyGroups +
            sizeof(float) * nv +
            sizeof(float) * nv +
            sizeof(float) * nc +
            sizeof(float) * nc +
            sizeof(float) * nc * nv +
            sizeof(float) * nc +
            sizeof(float) * nl * nv +
            sizeof(float) * nl +
            sizeof(float) * nl;
        return (float *)bytes;
    }

    inline float * getEqualityR(StateManager *state_mgr)
    {
        (void)state_mgr;
        uint8_t *bytes =
            prepMem +
            sizeof(SparseBlkDiag::Blk) * numBodyGroups +
            sizeof(float) * nv +
            sizeof(float) * nv +
            sizeof(float) * nc +
            sizeof(float) * nc +
            sizeof(float) * nc * nv +
            sizeof(float) * nc +
            sizeof(float) * nl * nv +
            sizeof(float) * nl +
            sizeof(float) * nl +
            sizeof(float) * nc;
        return (float *)bytes;
    }
#endif
};

}

#include "cv.inl"
