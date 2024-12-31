#include <algorithm>
#include <madrona/state.hpp>
#include <madrona/utils.hpp>
#include <madrona/physics.hpp>
#include <madrona/context.hpp>
#include <madrona/cvphysics.hpp>
#include <madrona/taskgraph.hpp>

#include "physics_impl.hpp"

#ifdef MADRONA_GPU_MODE
#include <madrona/mw_gpu/host_print.hpp>
#define LOG(...) mwGPU::HostPrint::log(__VA_ARGS__)
#else
#define LOG(...)
#endif

using namespace madrona::math;
using namespace madrona::base;

namespace madrona::phys::cv {

StateManager * getStateManager(Context &ctx)
{
#ifdef MADRONA_GPU_MODE
    return mwGPU::getStateManager();
#else
    return ctx.getStateManager();
#endif
}

StateManager * getStateManager()
{
#ifdef MADRONA_GPU_MODE
    return mwGPU::getStateManager();
#else
    assert(false);
#endif
}

struct MRElement128b {
    uint8_t d[128];
};

float * DofObjectTmpState::getPhiFull(Context &ctx)
{
    uint8_t *bytes = 
        (uint8_t *)ctx.memoryRangePointer<MRElement128b>(dynData) + 
        phiFullOffset;
    return (float *)bytes;
}

float * BodyGroupHierarchy::getMassMatrix(Context &ctx)
{
    uint8_t *bytes = 
        (uint8_t *)ctx.memoryRangePointer<MRElement128b>(dynData) + 
        massMatrixOffset;
    return (float *)bytes;
}

float * BodyGroupHierarchy::getMassMatrixLTDL(Context &ctx)
{
    uint8_t *bytes = 
        (uint8_t *)ctx.memoryRangePointer<MRElement128b>(dynData) + 
        massMatrixLTDLOffset;
    return (float *)bytes;
}

float * BodyGroupHierarchy::getBias(Context &ctx)
{
    uint8_t *bytes = 
        (uint8_t *)ctx.memoryRangePointer<MRElement128b>(dynData) + 
        biasOffset;
    return (float *)bytes;
}

int32_t * BodyGroupHierarchy::getExpandedParent(Context &ctx)
{
    uint8_t *bytes =
        (uint8_t *)ctx.memoryRangePointer<MRElement128b>(dynData) + 
        expandedParentOffset;
    return (int32_t *)bytes;
}

Entity * BodyGroupHierarchy::bodies(Context &ctx)
{
    uint8_t *bytes =
        (uint8_t *)ctx.memoryRangePointer<MRElement128b>(mrBodies);
    return (Entity *)bytes;
}

uint32_t * BodyGroupHierarchy::getDofPrefixSum(Context &ctx)
{
    uint8_t *bytes =
        (uint8_t *)ctx.memoryRangePointer<MRElement128b>(dynData) +
        dofPrefixSumOffset;
    return (uint32_t *)bytes;
}

struct Contact : Archetype<
    ContactConstraint,
    ContactTmpState
> {};

struct Joint : Archetype<
    JointConstraint
> {};

struct CVRigidBodyState : Bundle<
    CVPhysicalComponent
> {};

struct CVSolveData {
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
    float *mu;
    float *penetrations;

    uint32_t massDim;
    uint32_t freeAccDim;
    uint32_t velDim;
    uint32_t numRowsJc;
    uint32_t numColsJc;
    uint32_t muDim;
    uint32_t penetrationsDim;

    // This is going to be allocated during the solve
    float *aRef;

    CVXSolve *cvxSolve;
};

namespace tasks {

// Computes the expanded parent array (based on Table 6.4 of Featherstone)
inline void computeExpandedParent(Context &ctx,
                                  BodyGroupHierarchy &body_grp) {
    CountT numBodies = body_grp.numBodies;

    int32_t *expandedParent = body_grp.getExpandedParent(ctx);

    // Initialize n-N_B elements
    for(int32_t i = 0; i < body_grp.numDofs; ++i) {
        expandedParent[i] = i - 1;
    }

    // Create a mapping from body index to start of block
    int32_t *map = (int32_t *)ctx.tmpAlloc(sizeof(int32_t) * numBodies);

    map[0] = -1;

    for(int32_t i = 1; i < numBodies; ++i) {
        uint32_t n_i = ctx.get<DofObjectNumDofs>(body_grp.bodies(ctx)[i]).numDofs;
        map[i] = map[i - 1] + (int32_t) n_i;
    }
    // Finish expanded parent array
    for(int32_t i = 1; i < numBodies; ++i) {
        int32_t parent_idx = ctx.get<DofObjectHierarchyDesc>(
            body_grp.bodies(ctx)[i]).parentIndex;
        expandedParent[map[i - 1] + 1] = map[parent_idx];
    }
}

inline void forwardKinematics(Context &ctx,
                              BodyGroupHierarchy &body_grp)
{
    { // Set the parent's state
        auto &position = ctx.get<DofObjectPosition>(body_grp.bodies(ctx)[0]);
        auto &tmp_state = ctx.get<DofObjectTmpState>(body_grp.bodies(ctx)[0]);

        tmp_state.comPos = {
            position.q[0],
            position.q[1],
            position.q[2]
        };

        tmp_state.composedRot = {
            position.q[3],
            position.q[4],
            position.q[5],
            position.q[6]
        };

        // omega remains unchanged, and v only depends on the COM position
        tmp_state.phi.v[0] = tmp_state.comPos[0];
        tmp_state.phi.v[1] = tmp_state.comPos[1];
        tmp_state.phi.v[2] = tmp_state.comPos[2];
    }

    // Forward pass from parent to children
    for (int i = 1; i < body_grp.numBodies; ++i) {
        Entity body = body_grp.bodies(ctx)[i];
        auto &position = ctx.get<DofObjectPosition>(body);
        auto &num_dofs = ctx.get<DofObjectNumDofs>(body);
        auto &tmp_state = ctx.get<DofObjectTmpState>(body);
        auto &hier_desc = ctx.get<DofObjectHierarchyDesc>(body);

        Entity parent_e = hier_desc.parent;
        DofObjectTmpState &parent_tmp_state =
            ctx.get<DofObjectTmpState>(parent_e);

        // We can calculate our stuff.
        switch (num_dofs.numDofs) {
        case (uint32_t)DofType::Hinge: {
            // Find the hinge axis orientation in world space
            Vector3 rotated_hinge_axis =
                parent_tmp_state.composedRot.rotateVec(hier_desc.hingeAxis);

            // Calculate the composed rotation applied to the child entity.
            tmp_state.composedRot = parent_tmp_state.composedRot *
                Quat::angleAxis(position.q[0], hier_desc.hingeAxis);

            // Calculate the composed COM position of the child
            //  (parent COM + R_{parent} * (rel_pos_parent + R_{hinge} * rel_pos_local))
            tmp_state.comPos = parent_tmp_state.comPos +
                parent_tmp_state.composedRot.rotateVec(
                        hier_desc.relPositionParent +
                        Quat::angleAxis(position.q[0], hier_desc.hingeAxis).
                            rotateVec(hier_desc.relPositionLocal)
                );

            // All we are getting here is the position of the hinge point
            // which is relative to the parent's COM.
            tmp_state.anchorPos = parent_tmp_state.comPos +
                parent_tmp_state.composedRot.rotateVec(
                        hier_desc.relPositionParent);

            // Phi only depends on the hinge axis and the hinge point
            tmp_state.phi.v[0] = rotated_hinge_axis[0];
            tmp_state.phi.v[1] = rotated_hinge_axis[1];
            tmp_state.phi.v[2] = rotated_hinge_axis[2];
            tmp_state.phi.v[3] = tmp_state.anchorPos[0];
            tmp_state.phi.v[4] = tmp_state.anchorPos[1];
            tmp_state.phi.v[5] = tmp_state.anchorPos[2];

        } break;

        case (uint32_t)DofType::Ball: {
            Quat joint_rot = Quat{
                position.q[0], position.q[1], position.q[2], position.q[3] 
            };

            // Calculate the composed rotation applied to the child entity.
            tmp_state.composedRot = parent_tmp_state.composedRot * joint_rot;

            // Calculate the composed COM position of the child
            //  (parent COM + R_{parent} * (rel_pos_parent + R_{ball} * rel_pos_local))
            tmp_state.comPos = parent_tmp_state.comPos +
                parent_tmp_state.composedRot.rotateVec(
                        hier_desc.relPositionParent +
                        joint_rot.rotateVec(hier_desc.relPositionLocal)
                );

            // All we are getting here is the position of the ball point
            // which is relative to the parent's COM.
            tmp_state.anchorPos = parent_tmp_state.comPos +
                parent_tmp_state.composedRot.rotateVec(
                        hier_desc.relPositionParent);

            // Phi only depends on the hinge point and parent rotation
            tmp_state.phi.v[0] = tmp_state.anchorPos[0];
            tmp_state.phi.v[1] = tmp_state.anchorPos[1];
            tmp_state.phi.v[2] = tmp_state.anchorPos[2];
            tmp_state.phi.v[3] = parent_tmp_state.composedRot.w;
            tmp_state.phi.v[4] = parent_tmp_state.composedRot.x;
            tmp_state.phi.v[5] = parent_tmp_state.composedRot.y;
            tmp_state.phi.v[6] = parent_tmp_state.composedRot.z;
        } break;

        default: {
            // Only hinges have parents
            assert(false);
        } break;
        }
    }
}

// Init tasks
inline void initHierarchies(Context &ctx,
                            BodyGroupHierarchy &grp)
{
    CountT num_dofs = grp.numDofs;

    // We are going to do all dynamic allocations in one memory range to
    // minimize fragmentation.
    uint64_t required_bytes = 0;

    // Expanded parent arrary
    grp.expandedParentOffset = required_bytes;
    required_bytes += num_dofs * sizeof(int32_t);

    // Bias vector
    grp.biasOffset = (uint32_t)required_bytes;
    required_bytes += num_dofs * sizeof(float);

    // Mass matrix
    grp.massMatrixOffset = (uint32_t)required_bytes;
    required_bytes += num_dofs * num_dofs * sizeof(float);

    // LTDL mass matrix
    grp.massMatrixLTDLOffset = (uint32_t)required_bytes;
    required_bytes += num_dofs * num_dofs * sizeof(float);

    grp.dofPrefixSumOffset = (uint32_t)required_bytes;
    required_bytes += grp.numBodies * sizeof(uint32_t);

    // All the bodies' data
    for (CountT j = 0; j < grp.numBodies; ++j) {
        Entity body = grp.bodies(ctx)[j];
        auto &num_body_dofs = ctx.get<DofObjectNumDofs>(body);
        auto &tmp_state = ctx.get<DofObjectTmpState>(body);

        tmp_state.phiFullOffset = (uint32_t)required_bytes;

        // Space for both phi and phi_dot
        CountT num_phi_vals = 2 * 6 * num_body_dofs.numDofs;
        required_bytes += num_phi_vals * sizeof(float);
    }

    // Do memory range allocation now
    uint64_t num_elems = (required_bytes + sizeof(MRElement128b) - 1) /
        sizeof(MRElement128b);
    grp.dynData = ctx.allocMemoryRange<MRElement128b>(num_elems);

    uint32_t ps_offset = 0;
    uint32_t *prefix_sum = grp.getDofPrefixSum(ctx);
    for (CountT j = 0; j < grp.numBodies; ++j) {
        Entity body = grp.bodies(ctx)[j];
        auto &tmp_state = ctx.get<DofObjectTmpState>(body);
        auto &num_dofs = ctx.get<DofObjectNumDofs>(body);

        tmp_state.dynData = grp.dynData;
        prefix_sum[j] = ps_offset;

        ps_offset += num_dofs.numDofs;
    }
    
    // Now do some post-allocation computations
    tasks::computeExpandedParent(ctx, grp);

    // Forward kinematics to get positions
    tasks::forwardKinematics(ctx, grp);
}

// We are going to need a different way of solving on GPU mode
#ifdef MADRONA_GPU_MODE
struct GaussMinimizationNode : NodeBase {
    GaussMinimizationNode(StateManager *state_mgr);

    // Unsure how efficient this will be exactly, but we're not dealing
    // with HUGE matrices, just TONS of tiny ones. (WIP)
    template <typename DataT,
              uint32_t block_size>
    void gmmaWarpSmallSmem(
            DataT *res,
            DataT *a,
            DataT *b,
            uint32_t a_rows, uint32_t a_cols,
            uint32_t b_rows, uint32_t b_cols);

    template <typename DataT,
              uint32_t block_size,
              bool reset_res = false>
    void gmmaWarpSmallReg(
            DataT *res,
            DataT *a,
            DataT *b,
            uint32_t a_rows, uint32_t a_cols,
            uint32_t b_rows, uint32_t b_cols);

    template <typename DataT,
              uint32_t block_size>
    void copyToRegs(
            DataT (&blk_tmp)[block_size][block_size], // dst
            DataT *mtx,                               // src
            uint32_t mtx_rows, uint32_t mtx_cols,
            uint32_t blk_r, uint32_t blk_c);

    template <typename DataT,
              uint32_t block_size>
    void copyToMem(
            DataT *mtx,                               // dst
            DataT (&blk_tmp)[block_size][block_size], // src
            uint32_t mtx_rows, uint32_t mtx_cols,
            uint32_t blk_r, uint32_t blk_c);

    template <typename DataT,
              uint32_t block_size,
              bool reset_res = false>
    void gmmaBlockRegs(
            DataT (&res)[block_size][block_size],
            DataT (&a)[block_size][block_size],
            DataT (&b)[block_size][block_size]);

    template <typename DataT,
              uint32_t block_size>
    void setBlockZero(
            DataT (&blk)[block_size][block_size]);

    void computeARef(int32_t invocation_idx);

    static TaskGraph::NodeID addToGraph(
            TaskGraph::Builder &builder,
            Span<const TaskGraph::NodeID> dependencies);

    // Each `CVSolveData` contains the matrices / vectors we need.
    CVSolveData *solveDatas;
};

template <typename DataT,
          uint32_t block_size>
void GaussMinimizationNode::copyToRegs(
        DataT (&blk_tmp)[block_size][block_size],
        DataT *mtx,
        uint32_t mtx_rows, uint32_t mtx_cols,
        uint32_t blk_r, uint32_t blk_c)
{
    uint32_t col_start = blk_c * block_size;
    uint32_t row_start = blk_r * block_size;

    #pragma unroll
    for (uint32_t blk_row = 0; blk_row < block_size; ++blk_row) {
        #pragma unroll
        for (uint32_t blk_col = 0; blk_col < block_size; ++blk_col) {
            blk_tmp[blk_row][blk_col] = 
                (col_start + blk_col < mtx_cols &&
                 row_start + blk_row < mtx_rows) ?
                mtx[col_start + blk_col + mtx_cols * (row_start + blk_row)] :
                0.f;
        }
    }
}

template <typename DataT,
          uint32_t block_size>
void GaussMinimizationNode::copyToMem(
        DataT *mtx,
        DataT (&blk_tmp)[block_size][block_size],
        uint32_t mtx_rows, uint32_t mtx_cols,
        uint32_t blk_r, uint32_t blk_c)
{
    uint32_t col_start = blk_c * block_size;
    uint32_t row_start = blk_r * block_size;

    #pragma unroll
    for (uint32_t blk_row = 0; blk_row < block_size; ++blk_row) {
        #pragma unroll
        for (uint32_t blk_col = 0; blk_col < block_size; ++blk_col) {
            if (col_start + blk_col < mtx_cols &&
                 row_start + blk_row < mtx_rows) {
                mtx[col_start + blk_col + mtx_cols * (row_start + blk_row)] =
                    blk_tmp[blk_row][blk_col];
            }
        }
    }
}

template <typename DataT,
          uint32_t block_size,
          bool reset_res>
void GaussMinimizationNode::gmmaBlockRegs(
        DataT (&res)[block_size][block_size],
        DataT (&a)[block_size][block_size],
        DataT (&b)[block_size][block_size])
{
    #pragma unroll
    for (int i = 0; i < block_size; i++) {
        #pragma unroll
        for (int j = 0; j < block_size; j++) {
            if constexpr (reset_res) {
                res[i][j] = 0;
            }

            #pragma unroll
            for (int k = 0; k < block_size; k++) {
                res[i][j] += a[i][k] * b[k][j];
            }
        }
    }
}

template <typename DataT,
          uint32_t block_size>
void GaussMinimizationNode::setBlockZero(
        DataT (&blk)[block_size][block_size])
{
    #pragma unroll
    for (int i = 0; i < block_size; i++) {
        #pragma unroll
        for (int j = 0; j < block_size; j++) {
            blk[i][j] = 0.f;
        }
    }
}

template <typename DataT,
          uint32_t block_size>
void GaussMinimizationNode::gmmaWarpSmallSmem(
        DataT *res,
        DataT *a,
        DataT *b,
        uint32_t a_rows, uint32_t a_cols,
        uint32_t b_rows, uint32_t b_cols)
{
    const int32_t num_smem_bytes_per_warp =
        mwGPU::SharedMemStorage::numBytesPerWarp();
    auto smem_buf = (uint8_t *)mwGPU::SharedMemStorage::buffer +
                    num_smem_bytes_per_warp * warp_id;

    // This function requires (block_size x block_size) * 3 bytes of Smem
    assert(3 * (block_size * block_size) < num_smem_bytes_per_warp);

    DataT *a_blk_tmp = (DataT *)smem_buf;
    DataT *b_blk_tmp = (DataT *)smem_buf + (block_size * block_size);
    DataT *res_blk_tmp = (DataT *)smem_buf + (block_size * block_size) * 2;

    // Get value in block tmp matrix
    // TODO: Double check the compiler spits out not dumb stuff.
    auto bv = [](DataT *d, uint32_t row, uint32_t col) -> DataT & {
        return d[col + row * block_size];
    };

    
}

template <typename DataT,
          uint32_t block_size,
          bool reset_res>
void GaussMinimizationNode::gmmaWarpSmallReg(
        DataT *res,
        DataT *a,
        DataT *b,
        uint32_t a_rows, uint32_t a_cols,
        uint32_t b_rows, uint32_t b_cols)
{
    // TODO: Make sure non of this stuff spills
    DataT a_blk_tmp[block_size][block_size];
    DataT b_blk_tmp[block_size][block_size];
    DataT res_blk_tmp[block_size][block_size];

    uint32_t res_rows = a_rows;
    uint32_t res_cols = b_cols;

    uint32_t num_iters_r = (res_rows + block_size - 1) / block_size;
    uint32_t num_iters_c = (res_cols + block_size - 1) / block_size;
    uint32_t total_num_iters = num_iters_r * num_iters_c;

    uint32_t num_blks_a = (a_cols + block_size - 1) / block_size;
    uint32_t num_blks_b = (b_rows + block_size - 1) / block_size;

    uint32_t lane_id = threadIdx.x % 32;
    uint32_t curr_iter = lane_id;
    
    while (curr_iter < total_num_iters) {
        uint32_t res_blk_r = total_num_iters / num_iters_c;
        uint32_t res_blk_c = total_num_iters % num_iters_c;

        if constexpr (reset_res) {
            setBlockZero(res_blk_tmp);
        }

        for (uint32_t blk_i = 0; blk_i < num_blks_a; ++blk_i) {
            for (uint32_t blk_j = 0; blk_j < num_blks_b; ++blk_j) {
                copyToRegs(a_blk_tmp, a, a_rows, a_cols, res_blk_r, blk_i);
                copyToRegs(b_blk_tmp, b, b_rows, b_cols, blk_j, res_blk_c);
                gmmaBlockRegs(res_blk_tmp, a_blk_tmp, b_blk_tmp);
            }
        }

        copyToMem(res, res_blk_tmp, res_rows, 
                  res_cols, res_blk_r, res_blk_c);

        curr_iter += 32;
    }
}

GaussMinimizationNode::GaussMinimizationNode(
        StateManager *s)
    : solveDatas(s->getSingletonColumn<CVSolveData>())
{
}

// Might be overkill to allocate a warp per world but we can obviously
// experiment.
void GaussMinimizationNode::computeARef(int32_t invocation_idx)
{
    uint32_t total_resident_warps = (blockDim.x * gridDim.x) / 32;

    uint32_t total_num_worlds = mwGPU::GPUImplConsts::get().numWorlds;
    uint32_t world_idx = (blockDim.x * blockIdx.x + threadIdx.x) / 32;

    uint32_t warp_id = threadIdx.x / 32;
    uint32_t lane_id = threadIdx.x % 32;

    const int32_t num_smem_bytes_per_warp =
        mwGPU::SharedMemStorage::numBytesPerWarp();
    auto smem_buf = (uint8_t *)mwGPU::SharedMemStorage::buffer +
                    num_smem_bytes_per_warp * warp_id;

    assert(blockDim.x == consts::numMegakernelThreads);

    StateManager *state_mgr = getStateManager();

    uint32_t world_id = warp_id;
    while (world_id < total_num_worlds) {
        CVSolveData *curr_sd = &solveDatas[world_id];

        // Allocate space for the a_ref vector
        float *a_ref = nullptr;
        if (lane_id == 0) {
            if (curr_sd->numRowsJc > 0) {
                a_ref = (float *)mwGPU::TmpAllocator::get().alloc(
                    sizeof(float) * curr_sd->numRowsJc);
            }

            LOG("smem per warp: {}\n", num_smem_bytes_per_warp);
        }

        __syncwarp();
        a_ref = (float *)__shfl_sync(0xFFFF'FFFF, (uint64_t)a_ref, 0);

        // Just make sure this compiles
        gmmaWarpSmallReg<float, 4, true>(
                a_ref, 
                curr_sd->J_c,
                curr_sd->vel,
                curr_sd->numRowsJc,
                curr_sd->numColsJc,
                curr_sd->velDim,
                1);

        world_id += total_resident_warps;
    }
}

TaskGraph::NodeID GaussMinimizationNode::addToGraph(
        TaskGraph::Builder &builder,
        Span<const TaskGraph::NodeID> deps)
{
    using namespace mwGPU;

    StateManager *state_mgr = getStateManager();

    auto data_id = builder.constructNodeData<GaussMinimizationNode>(
            state_mgr);
    auto &gauss_data = builder.getDataRef(data_id);

    // For now, we are going with the persistent threads approach where each
    // thread block is going to process a world.
    uint32_t num_invocations = (uint32_t)gridDim.x;

    TaskGraph::NodeID a_ref_node = builder.addNodeFn<
        &GaussMinimizationNode::computeARef>(data_id, {},
                Optional<TaskGraph::NodeID>::none(),
                num_invocations,
                // This is the thread block dimension
                consts::numMegakernelThreads);

    return a_ref_node;
}
#else
inline void solveCPU(Context &ctx,
                     CVSolveData &cv_sing)
{
    uint32_t world_id = ctx.worldID().idx;

    StateManager *state_mgr = getStateManager(ctx);
    PhysicsSystemState &physics_state = ctx.singleton<PhysicsSystemState>();

    // Call the solver
    if (cv_sing.cvxSolve && cv_sing.cvxSolve->fn) {
        cv_sing.cvxSolve->totalNumDofs = cv_sing.totalNumDofs;
        cv_sing.cvxSolve->numContactPts = cv_sing.numContactPts;
        cv_sing.cvxSolve->h = cv_sing.h;
        cv_sing.cvxSolve->mass = cv_sing.mass;
        cv_sing.cvxSolve->free_acc = cv_sing.freeAcc;
        cv_sing.cvxSolve->vel = cv_sing.vel;
        cv_sing.cvxSolve->J_c = cv_sing.J_c;
        cv_sing.cvxSolve->mu = cv_sing.mu;
        cv_sing.cvxSolve->penetrations = cv_sing.penetrations;

        cv_sing.cvxSolve->callSolve.store_release(1);
        while (cv_sing.cvxSolve->callSolve.load_acquire() != 2);
        cv_sing.cvxSolve->callSolve.store_relaxed(0);

        float *res = cv_sing.cvxSolve->resPtr;

        if (res) {
            BodyGroupHierarchy *hiers = state_mgr->getWorldComponents<
                BodyGroup, BodyGroupHierarchy>(world_id);

            CountT num_grps = state_mgr->numRows<BodyGroup>(world_id);

            // Update the body accelerations
            uint32_t processed_dofs = 0;
            for (CountT i = 0; i < num_grps; ++i)
            {
                for (CountT j = 0; j < hiers[i].numBodies; j++)
                {
                    auto body = hiers[i].bodies(ctx)[j];
                    auto numDofs = ctx.get<DofObjectNumDofs>(body).numDofs;
                    auto &acceleration = ctx.get<DofObjectAcceleration>(body);
                    for (CountT k = 0; k < numDofs; k++) {
                        acceleration.dqv[k] = res[processed_dofs];
                        processed_dofs++;
                    }
                }
            }
        }
    }
}
#endif

inline Mat3x3 skewSymmetricMatrix(Vector3 v)
{
    return {
        {
            { 0.f, v.z, -v.y },
            { -v.z, 0.f, v.x },
            { v.y, -v.x, 0.f }
        }
    };
}

inline void computeBodyCOM(Context &ctx,
                           DofObjectTmpState &tmp_state,
                           const ObjectID obj_id,
                           const DofObjectNumDofs num_dofs)
{
    if (num_dofs.numDofs == (uint32_t)DofType::FixedBody) {
        tmp_state.scratch[0] =
            tmp_state.scratch[0] =
            tmp_state.scratch[0] =
            tmp_state.scratch[0] = 0.f;
    } else {
        const ObjectManager &obj_mgr = *ctx.singleton<ObjectData>().mgr;

        float mass = 1.f / obj_mgr.metadata[obj_id.idx].mass.invMass;

        tmp_state.scratch[0] = mass * tmp_state.comPos.x;
        tmp_state.scratch[1] = mass * tmp_state.comPos.y;
        tmp_state.scratch[2] = mass * tmp_state.comPos.z;
        tmp_state.scratch[3] = mass;
    }
}

inline void computeTotalCOM(Context &ctx,
                            BodyGroupHierarchy &body_grp)
{
    Vector3 hierarchy_com = Vector3::zero();
    float total_mass = 0.f;

    Entity *bodies = body_grp.bodies(ctx);

    for (int i = 0; i < body_grp.numBodies; ++i) {
        auto &tmp_state = ctx.get<DofObjectTmpState>(bodies[i]);

        hierarchy_com += Vector3 {
            tmp_state.scratch[0],
            tmp_state.scratch[1],
            tmp_state.scratch[2] 
        };

        total_mass += tmp_state.scratch[3];
    }

    body_grp.comPos = hierarchy_com / total_mass;
}

// Pre-CRB: compute the spatial inertia in common Plücker coordinates
#if 0
inline void computeSpatialInertia(Context &ctx,
                                  BodyGroupHierarchy &body_grp) {
    ObjectManager &obj_mgr = *ctx.singleton<ObjectData>().mgr;

    for(int i = 0; i < body_grp.numBodies; i++)
    {
        auto num_dofs = ctx.get<DofObjectNumDofs>(body_grp.bodies(ctx)[i]);
        auto obj_id = ctx.get<ObjectID>(body_grp.bodies(ctx)[i]);
        auto &tmp_state = ctx.get<DofObjectTmpState>(body_grp.bodies(ctx)[i]);

        if(num_dofs.numDofs == (uint32_t)DofType::FixedBody) {
            tmp_state.spatialInertia.mass = 0.f;
            continue;;
        }

        RigidBodyMetadata metadata = obj_mgr.metadata[obj_id.idx];
        Diag3x3 inertia = Diag3x3::fromVec(metadata.mass.invInertiaTensor).inv();
        float mass = 1.f / metadata.mass.invMass;

        // We need to find inertia tensor in world space orientation
        Mat3x3 rot_mat = Mat3x3::fromQuat(tmp_state.composedRot);
        // I_world = R * I * R^T (since R^T transforms from world to local)
        Mat3x3 i_world_frame = rot_mat * inertia * rot_mat.transpose();

        // Compute the 3x3 skew-symmetric matrix (r^x)
        // (where r is from Plücker origin to COM)
        Vector3 adjustedCom = tmp_state.comPos - body_grp.comPos;
        Mat3x3 sym_mat = skewSymmetricMatrix(adjustedCom);
        // (I_world - m r^x r^x)
        Mat3x3 inertia_mat = i_world_frame - (mass * sym_mat * sym_mat);

        // Take only the upper triangular part (since it's symmetric)
        tmp_state.spatialInertia.spatial_inertia[0] = inertia_mat[0][0];
        tmp_state.spatialInertia.spatial_inertia[1] = inertia_mat[1][1];
        tmp_state.spatialInertia.spatial_inertia[2] = inertia_mat[2][2];
        tmp_state.spatialInertia.spatial_inertia[3] = inertia_mat[1][0];
        tmp_state.spatialInertia.spatial_inertia[4] = inertia_mat[2][0];
        tmp_state.spatialInertia.spatial_inertia[5] = inertia_mat[2][1];

        // Rest of parameters
        tmp_state.spatialInertia.mass = mass;
        tmp_state.spatialInertia.mCom = mass * adjustedCom;
    }
}
#endif

inline void computeSpatialInertias(Context &ctx,
                                   DofObjectTmpState &tmp_state,
                                   const DofObjectHierarchyDesc hier_desc,
                                   const ObjectID obj_id,
                                   const DofObjectNumDofs num_dofs)
{
    ObjectManager &obj_mgr = *ctx.singleton<ObjectData>().mgr;

    if(num_dofs.numDofs == (uint32_t)DofType::FixedBody) {
        tmp_state.spatialInertia.mass = 0.f;
        return;
    }

    Vector3 body_grp_com_pos = ctx.get<BodyGroupHierarchy>(
            hier_desc.bodyGroup).comPos;

    RigidBodyMetadata metadata = obj_mgr.metadata[obj_id.idx];
    Diag3x3 inertia = Diag3x3::fromVec(metadata.mass.invInertiaTensor).inv();
    float mass = 1.f / metadata.mass.invMass;

    // We need to find inertia tensor in world space orientation
    Mat3x3 rot_mat = Mat3x3::fromQuat(tmp_state.composedRot);
    // I_world = R * I * R^T (since R^T transforms from world to local)
    Mat3x3 i_world_frame = rot_mat * inertia * rot_mat.transpose();

    // Compute the 3x3 skew-symmetric matrix (r^x)
    // (where r is from Plücker origin to COM)
    Vector3 adjustedCom = tmp_state.comPos - body_grp_com_pos;
    Mat3x3 sym_mat = skewSymmetricMatrix(adjustedCom);
    // (I_world - m r^x r^x)
    Mat3x3 inertia_mat = i_world_frame - (mass * sym_mat * sym_mat);

    // Take only the upper triangular part (since it's symmetric)
    tmp_state.spatialInertia.spatial_inertia[0] = inertia_mat[0][0];
    tmp_state.spatialInertia.spatial_inertia[1] = inertia_mat[1][1];
    tmp_state.spatialInertia.spatial_inertia[2] = inertia_mat[2][2];
    tmp_state.spatialInertia.spatial_inertia[3] = inertia_mat[1][0];
    tmp_state.spatialInertia.spatial_inertia[4] = inertia_mat[2][0];
    tmp_state.spatialInertia.spatial_inertia[5] = inertia_mat[2][1];

    // Rest of parameters
    tmp_state.spatialInertia.mass = mass;
    tmp_state.spatialInertia.mCom = mass * adjustedCom;
}

// Compute spatial inertia of subtree rooted at the body.
inline void combineSpatialInertias(Context &ctx,
                                   BodyGroupHierarchy &body_grp)
{
    // Backward pass from children to parent
    for (CountT i = body_grp.numBodies-1; i > 0; --i) {
        Entity body_i = body_grp.bodies(ctx)[i];

        auto &current_hier_desc = ctx.get<DofObjectHierarchyDesc>(body_i);
        auto &current_tmp_state = ctx.get<DofObjectTmpState>(body_i);

        auto &parent_tmp_state = ctx.get<DofObjectTmpState>(
                body_grp.bodies(ctx)[current_hier_desc.parentIndex]);

        parent_tmp_state.spatialInertia += current_tmp_state.spatialInertia;
    }
}

// Computes the Phi matrix from generalized velocities to Plücker coordinates
inline float* computePhi(Context &ctx,
                         const DofObjectNumDofs num_dofs,
                         DofObjectTmpState &tmp_state,
                         Vector3 origin)
{
    Phi phi = tmp_state.phi;

    float *S = tmp_state.getPhiFull(ctx);

    if (num_dofs.numDofs == (uint32_t)DofType::FreeBody) {
        // S = [1_3x3 r^x; 0 1_3x3], column-major
        memset(S, 0.f, 6 * 6 * sizeof(float));
        // Diagonal identity
        for(CountT i = 0; i < 6; ++i) {
            S[i * 6 + i] = 1.f;
        }
        // r^x Skew symmetric matrix
        Vector3 comPos = {phi.v[0], phi.v[1], phi.v[2]};
        comPos -= origin;
        S[0 + 6 * 4] = -comPos.z;
        S[0 + 6 * 5] = comPos.y;
        S[1 + 6 * 3] = comPos.z;
        S[1 + 6 * 5] = -comPos.x;
        S[2 + 6 * 3] = -comPos.y;
        S[2 + 6 * 4] = comPos.x;
    }
    else if (num_dofs.numDofs == (uint32_t)DofType::Hinge) {
        // S = [r \times hinge; hinge]
        Vector3 hinge = {phi.v[0], phi.v[1], phi.v[2]};
        Vector3 anchorPos = {phi.v[3], phi.v[4], phi.v[5]};
        anchorPos -= origin;
        Vector3 r_cross_hinge = anchorPos.cross(hinge);
        S[0] = r_cross_hinge.x;
        S[1] = r_cross_hinge.y;
        S[2] = r_cross_hinge.z;
        S[3] = hinge.x;
        S[4] = hinge.y;
        S[5] = hinge.z;
    }
    else if (num_dofs.numDofs == (uint32_t)DofType::Ball) {
        // This will just get right-multiplied by the angular velocity
        Vector3 anchor_pos = {phi.v[0], phi.v[1], phi.v[2]};
        anchor_pos -= origin;

        // We need to right multiply these by the parent composed rotation
        // matrix.
        Mat3x3 rx = skewSymmetricMatrix(anchor_pos);
        Quat parent_composed_rot = Quat{
            phi.v[3], phi.v[4], phi.v[5], phi.v[6]
        };
        Mat3x3 parent_rot = Mat3x3::fromQuat(parent_composed_rot);

        rx *= parent_rot;

        for (int col = 0; col < 3; ++col) {
            S[col * 6 + 0] = rx[col][0];
            S[col * 6 + 1] = rx[col][1];
            S[col * 6 + 2] = rx[col][2];

            S[col * 6 + 3] = parent_rot[col][0];
            S[col * 6 + 4] = parent_rot[col][1];
            S[col * 6 + 5] = parent_rot[col][2];
        }
    }
    else {
        MADRONA_UNREACHABLE();
    }
    return S;
}

inline float* computePhiDot(Context &ctx,
                            DofObjectNumDofs &num_dofs,
                            DofObjectTmpState &tmp_state,
                            SpatialVector &v_hat)
{
    float *S = tmp_state.getPhiFull(ctx);

    // Same storage location, but will need to be incremented based on size of S
    float *S_dot = S + 6 * num_dofs.numDofs;

    if (num_dofs.numDofs == (uint32_t)DofType::FreeBody) {
        // S_dot = [0_3x3 v^x; 0_3x3 0_3x3], column-major
        memset(S_dot, 0.f, 6 * 6 * sizeof(float));
        // v^x Skew symmetric matrix
        Vector3 v_trans = v_hat.linear;
        S_dot[0 + 6 * 4] = -v_trans.z;
        S_dot[0 + 6 * 5] = v_trans.y;
        S_dot[1 + 6 * 3] = v_trans.z;
        S_dot[1 + 6 * 5] = -v_trans.x;
        S_dot[2 + 6 * 3] = -v_trans.y;
        S_dot[2 + 6 * 4] = v_trans.x;
    }
    else if (num_dofs.numDofs == (uint32_t)DofType::Hinge) {
        // S_dot = v [spatial cross] S
        SpatialVector S_sv = SpatialVector::fromVec(S);
        SpatialVector S_dot_sv = v_hat.cross(S_sv);
        S_dot[0] = S_dot_sv.linear.x;
        S_dot[1] = S_dot_sv.linear.y;
        S_dot[2] = S_dot_sv.linear.z;
        S_dot[3] = S_dot_sv.angular.x;
        S_dot[4] = S_dot_sv.angular.y;
        S_dot[5] = S_dot_sv.angular.z;
    }
    else if (num_dofs.numDofs == (uint32_t)DofType::Ball) {
        // S_dot = v [spatial cross] S
        for (int i = 0; i < 3; ++i) {
            SpatialVector S_sv = SpatialVector::fromVec(S + (i * 6));
            SpatialVector S_dot_sv = v_hat.cross(S_sv);

            S_dot[i * 6 + 0] = S_dot_sv.linear.x;
            S_dot[i * 6 + 1] = S_dot_sv.linear.y;
            S_dot[i * 6 + 2] = S_dot_sv.linear.z;
            S_dot[i * 6 + 3] = S_dot_sv.angular.x;
            S_dot[i * 6 + 4] = S_dot_sv.angular.y;
            S_dot[i * 6 + 5] = S_dot_sv.angular.z;
        }
    }
    return S_dot;
}

// First pass to compute Phi with body COM as origin
inline void computePhiHierarchy(Context &ctx,
                                DofObjectTmpState &tmp_state,
                                const DofObjectNumDofs num_dofs,
                                const DofObjectHierarchyDesc &desc)
{
    if (num_dofs.numDofs > 0) {
        Vector3 com_pos = ctx.get<BodyGroupHierarchy>(desc.bodyGroup).comPos;
        computePhi(ctx, num_dofs, tmp_state, com_pos);
    }
}

inline float* computeContactJacobian(Context &ctx,
                                     BodyGroupHierarchy &body_grp,
                                     DofObjectHierarchyDesc &hier_desc,
                                     Mat3x3 &C,
                                     Vector3 &origin,
                                     float *J)
{
    uint32_t world_id = ctx.worldID().idx;
    StateManager *state_mgr = getStateManager(ctx);
    // J_C = C^T[e_{b1} S_1, e_{b2} S_2, ...], col-major
    //  where e_{bi} = 1 if body i is an ancestor of b
    //  C^T projects into the contact space
#if 0
    float *J = (float *) ctx.tmpAlloc(world_id,
        3 * body_grp.numDofs * sizeof(float));
#endif

    memset(J, 0.f, 3 * body_grp.numDofs * sizeof(float));

    // Compute prefix sum to determine the start of the block for each body
    uint32_t *block_start = body_grp.getDofPrefixSum(ctx); //[body_grp.numBodies];

#if 0
    uint32_t block_offset = 0;
    for (CountT i = 0; i < body_grp.numBodies; ++i) {
        block_start[i] = block_offset;
        block_offset += ctx.get<DofObjectNumDofs>(
                body_grp.bodies(ctx)[i]).numDofs;
    }
#endif

    // Populate J_C by traversing up the hierarchy
    int32_t curr_idx = hier_desc.index;
    while(curr_idx != -1) {
        Entity body = body_grp.bodies(ctx)[curr_idx];
        auto &curr_tmp_state = ctx.get<DofObjectTmpState>(body);
        auto &curr_num_dofs = ctx.get<DofObjectNumDofs>(body);
        auto &curr_hier_desc = ctx.get<DofObjectHierarchyDesc>(body);

        // Populate columns of J_C
        float *S = computePhi(ctx, curr_num_dofs, curr_tmp_state, origin);
        // Only use translational part of S
        for(CountT i = 0; i < curr_num_dofs.numDofs; ++i) {
            float *J_col = J + 3 * (block_start[curr_idx] + i);
            float *S_col = S + 6 * i;
            for(CountT j = 0; j < 3; ++j) {
                J_col[j] = S_col[j];
            }
        }
        curr_idx = curr_hier_desc.parentIndex;
    }

    // Multiply by C^T to project into contact space
    for(CountT i = 0; i < body_grp.numDofs; ++i) {
        float *J_col = J + 3 * i;
        Vector3 J_col_vec = { J_col[0], J_col[1], J_col[2] };
        J_col_vec = C.transpose() * J_col_vec;
        J_col[0] = J_col_vec.x;
        J_col[1] = J_col_vec.y;
        J_col[2] = J_col_vec.z;
    }
    return J;
}

// y = Mx. Based on Table 6.5 in Featherstone
inline void mulM(Context &ctx, BodyGroupHierarchy &body_grp,
    float *x, float *y) {
    CountT total_dofs = body_grp.numDofs;

    int32_t *expandedParent = body_grp.getExpandedParent(ctx);

    float *massMatrix = body_grp.getMassMatrix(ctx);

    auto M = [&](int32_t row, int32_t col) -> float& {
        return massMatrix[row + total_dofs * col];
    };

    for(int32_t i = 0; i < total_dofs; ++i) {
        y[i] = M(i, i) * x[i];
    }

    for(int32_t i = (int32_t) total_dofs - 1; i >= 0; --i) {
        int32_t j = expandedParent[i];
        while(j != -1) {
            y[i] += M(i, j) * x[j];
            y[j] += M(i, j) * x[i];
            j = expandedParent[j];
        }
    }
}

// Solves M^{-1}x, overwriting x. Based on Table 6.5 in Featherstone
inline void solveM(Context &ctx, BodyGroupHierarchy &body_grp, float *x) {
    CountT total_dofs = body_grp.numDofs;

    int32_t *expandedParent = body_grp.getExpandedParent(ctx);
    float *massMatrixLTDL = body_grp.getMassMatrixLTDL(ctx);

    auto ltdl = [&](int32_t row, int32_t col) -> float& {
        return massMatrixLTDL[row + total_dofs * col];
    };
    // M=L^TDL, so first solve L^{-T} x
    for (int32_t i = (int32_t) total_dofs - 1; i >= 0; --i) {
        int32_t j = expandedParent[i];
        while (j != -1) {
            x[j] -= ltdl(i, j) * x[i];
            j = expandedParent[j];
        }
    }
    // D^{-1} x
    for (int32_t i = 0; i < total_dofs; ++i) {
        x[i] /= ltdl(i, i);
    }
    // L^{-1} x
    for (int32_t i = 0; i < total_dofs; ++i) {
        int32_t j = expandedParent[i];
        while (j != -1) {
            x[i] -= ltdl(i, j) * x[j];
            j = expandedParent[j];
        }
    }
}


// CRB: Compute the Mass Matrix (n_dofs x n_dofs)
inline void compositeRigidBody(Context &ctx,
                               BodyGroupHierarchy &body_grp)
{
    uint32_t world_id = ctx.worldID().idx;
    StateManager *state_mgr = getStateManager(ctx);

    // Mass Matrix of this entire body group, column-major
    uint32_t total_dofs = body_grp.numDofs;

    float *M = body_grp.getMassMatrix(ctx);

    memset(M, 0.f, total_dofs * total_dofs * sizeof(float));

    // Compute prefix sum to determine the start of the block for each body
    uint32_t *block_start = body_grp.getDofPrefixSum(ctx);

    // Backward pass
    for (CountT i = body_grp.numBodies-1; i >= 0; --i) {
        Entity body = body_grp.bodies(ctx)[i];

        auto &i_hier_desc = ctx.get<DofObjectHierarchyDesc>(body);
        auto &i_tmp_state = ctx.get<DofObjectTmpState>(body);
        auto &i_num_dofs = ctx.get<DofObjectNumDofs>(body);

        float *S_i = i_tmp_state.getPhiFull(ctx);

        float *F = body_grp.scratch;

        for(CountT col = 0; col < i_num_dofs.numDofs; ++col) {
            float *S_col = S_i + 6 * col;
            float *F_col = F + 6 * col;
            i_tmp_state.spatialInertia.multiply(S_col, F_col);
        }

        // M_{ii} = S_i^T I_i^C S_i = F^T S_i
        float *M_ii = M + block_start[i] * total_dofs + block_start[i];
        for(CountT row = 0; row < i_num_dofs.numDofs; ++row) {
            float *F_col = F + 6 * row; // take col for transpose
            for(CountT col = 0; col < i_num_dofs.numDofs; ++col) {
                float *S_col = S_i + 6 * col;
                for(CountT k = 0; k < 6; ++k) {
                    M_ii[row + total_dofs * col] += F_col[k] * S_col[k];
                }
            }
        }

        // Traverse up hierarchy
        uint32_t j = i;
        auto parent_j = i_hier_desc.parent;
        while(parent_j != Entity::none()) {
            j = ctx.get<DofObjectHierarchyDesc>(
                body_grp.bodies(ctx)[j]).parentIndex;
            Entity body = body_grp.bodies(ctx)[j];
            auto &j_tmp_state = ctx.get<DofObjectTmpState>(body);
            auto &j_num_dofs = ctx.get<DofObjectNumDofs>(body);

            float *S_j = j_tmp_state.getPhiFull(ctx);

            // M_{ij} = M{ji} = F^T S_j
            float *M_ij = M + block_start[i] + total_dofs * block_start[j]; // row i, col j
            float *M_ji = M + block_start[j] + total_dofs * block_start[i]; // row j, col i
            for(CountT row = 0; row < i_num_dofs.numDofs; ++row) {
                float *F_col = F + 6 * row; // take col for transpose
                for(CountT col = 0; col < j_num_dofs.numDofs; ++col) {
                    float *S_col = S_j + 6 * col;
                    for(CountT k = 0; k < 6; ++k) {
                        M_ij[row + total_dofs * col] += F_col[k] * S_col[k];
                        M_ji[col + total_dofs * row] += F_col[k] * S_col[k];
                    }
                }
            }
            parent_j = ctx.get<DofObjectHierarchyDesc>(
                body_grp.bodies(ctx)[j]).parent;
        }
    }
}

// Computes the LTDL factorization of the mass matrix
inline void factorizeMassMatrix(Context &ctx,
                                BodyGroupHierarchy &body_grp) {
    uint32_t world_id = ctx.worldID().idx;
    StateManager *state_mgr = getStateManager(ctx);

    // First copy in the mass matrix
    uint32_t total_dofs = body_grp.numDofs;

    float *LTDL = body_grp.getMassMatrixLTDL(ctx);
    float *massMatrix = body_grp.getMassMatrix(ctx);

    memcpy(LTDL, massMatrix, total_dofs * total_dofs * sizeof(float));

    // Helper
    auto ltdl = [&](int32_t row, int32_t col) -> float& {
        return LTDL[row + total_dofs * col];
    };

    int32_t *expandedParent = body_grp.getExpandedParent(ctx);

    // Backward pass through DOFs
    for (int32_t k = (int32_t) total_dofs - 1; k >= 0; --k) {
        int32_t i = expandedParent[k];
        while (i != -1) {
            // Temporary storage
            float a = ltdl(k, i) / ltdl(k, k);
            int32_t j = i;
            while (j != -1) {
                ltdl(i, j) = ltdl(i, j) - a * ltdl(k, j);
                j = expandedParent[j];
            }
            ltdl(k, i) = a;
            i = expandedParent[i];
        }
    }
}

// Computes the unconstrained acceleration of each body
inline void computeFreeAcceleration(Context &ctx,
                                    BodyGroupHierarchy &body_grp) {
    // Negate the bias forces, solve
    CountT num_dofs = body_grp.numDofs;

    float *bias = body_grp.getBias(ctx);

    for (CountT i = 0; i < num_dofs; ++i) {
        bias[i] = -bias[i];
    }
    // This overwrites bias with the acceleration
    solveM(ctx, body_grp, bias);
}

// RNE: Compute bias forces and gravity
// May want to do a GPU specific version of this to extract some
// parallelism out of this
inline void recursiveNewtonEuler(Context &ctx,
                                BodyGroupHierarchy &body_grp)
{
    PhysicsSystemState &physics_state = ctx.singleton<PhysicsSystemState>();
    uint32_t total_dofs = body_grp.numDofs;

    // Forward pass. Find in Plücker coordinates:
    //  1. velocities. v_i = v_{parent} + S * \dot{q_i}
    //  2. accelerations. a_i = a_{parent} + \dot{S} * \dot{q_i} + S * \ddot{q_i}
    //  3. forces. f_i = I_i a_i + v_i [spatial star cross] I_i v_i
    for (int i = 0; i < body_grp.numBodies; ++i) {
        Entity body = body_grp.bodies(ctx)[i];

        auto num_dofs = ctx.get<DofObjectNumDofs>(body);
        auto velocity = ctx.get<DofObjectVelocity>(body);

        auto &tmp_state = ctx.get<DofObjectTmpState>(body);
        auto &hier_desc = ctx.get<DofObjectHierarchyDesc>(body);

        float *S = tmp_state.getPhiFull(ctx);

        if (num_dofs.numDofs == (uint32_t)DofType::FreeBody) {
            // Free bodies must be root of their hierarchy
            SpatialVector v_body = {
                {velocity.qv[0], velocity.qv[1], velocity.qv[2]},
                Vector3::zero()
            };
            float *S_dot = computePhiDot(ctx, num_dofs, tmp_state, v_body);

            // v_0 = 0, a_0 = -g (fictitious upward acceleration)
            tmp_state.sAcc = {-physics_state.g, Vector3::zero()};
            tmp_state.sVel = {Vector3::zero(), Vector3::zero()};

            // S\dot{q_i} and \dot{S}\dot{q_i}
            for (int j = 0; j < 6; ++j) {
                for (int k = 0; k < num_dofs.numDofs; ++k) {
                    tmp_state.sVel[j] += S[j + 6 * k] * velocity.qv[k];
                    tmp_state.sAcc[j] += S_dot[j + 6 * k] * velocity.qv[k];
                }
            }
        }
        else if (num_dofs.numDofs == (uint32_t)DofType::FixedBody) {
            // Fixeds bodies must also be root of their hierarchy
            tmp_state.sVel = {Vector3::zero(), Vector3::zero()};
            tmp_state.sAcc = {-physics_state.g, Vector3::zero()};
        }
        else if (num_dofs.numDofs == (uint32_t)DofType::Hinge) {
            assert(hier_desc.parent != Entity::none());
            auto parent_tmp_state = ctx.get<DofObjectTmpState>(hier_desc.parent);
            tmp_state.sVel = parent_tmp_state.sVel;
            tmp_state.sAcc = parent_tmp_state.sAcc;

            // v_i = v_{parent} + S * \dot{q_i}, compute S * \dot{q_i}
            // a_i = a_{parent} + \dot{S} * \dot{q_i} [+ S * \ddot{q_i}, which is 0]
            // Note: we are using the parent velocity here (for hinge itself)
            float *S_dot = computePhiDot(ctx, num_dofs, tmp_state,
                parent_tmp_state.sVel);

            float q_dot = velocity.qv[0];
            for (int j = 0; j < 6; ++j) {
                tmp_state.sVel[j] += S[j] * q_dot;
                tmp_state.sAcc[j] += S_dot[j] * q_dot;
            }
        }
        else if (num_dofs.numDofs == (uint32_t)DofType::Ball) {
            assert(hier_desc.parent != Entity::none());
            auto parent_tmp_state = ctx.get<DofObjectTmpState>(hier_desc.parent);
            tmp_state.sVel = parent_tmp_state.sVel;
            tmp_state.sAcc = parent_tmp_state.sAcc;

            // v_i = v_{parent} + S * \dot{q_i}, compute S * \dot{q_i}
            // a_i = a_{parent} + \dot{S} * \dot{q_i} [+ S * \ddot{q_i}, which is 0]
            // Note: we are using the parent velocity here (for hinge itself)
            float *S_dot = computePhiDot(ctx, num_dofs, tmp_state,
                parent_tmp_state.sVel);

            float *q_dot = velocity.qv;
            for (int j = 0; j < 6; ++j) {
                // TODO: Probably should switch to row major - this isn't
                // particularly cache-friendly.
                tmp_state.sVel[j] += S[j + 6 * 0] * q_dot[0] + 
                                     S[j + 6 * 1] * q_dot[1] +
                                     S[j + 6 * 2] * q_dot[2];

                tmp_state.sAcc[j] += S_dot[j + 6 * 0] * q_dot[0] +
                                     S_dot[j + 6 * 1] * q_dot[1] +
                                     S_dot[j + 6 * 2] * q_dot[2];
            }
        } else {
            MADRONA_UNREACHABLE();
        }

        // f_i = I_i a_i + v_i [spatial star cross] I_i v_i
        tmp_state.sForce = tmp_state.spatialInertia.multiply(tmp_state.sAcc);
        tmp_state.sForce += tmp_state.sVel.crossStar(
            tmp_state.spatialInertia.multiply(tmp_state.sVel));
    }

    // Backward pass to find bias forces
    float *tau = body_grp.getBias(ctx);

    memset(tau, 0, total_dofs * sizeof(float));

    CountT dof_index = total_dofs;
    for (CountT i = body_grp.numBodies-1; i >= 0; --i) {
        auto &hier_desc = ctx.get<DofObjectHierarchyDesc>(body_grp.bodies(ctx)[i]);
        auto num_dofs = ctx.get<DofObjectNumDofs>(body_grp.bodies(ctx)[i]);
        auto &tmp_state = ctx.get<DofObjectTmpState>(body_grp.bodies(ctx)[i]);

        // tau_i = S_i^T f_i
        dof_index -= num_dofs.numDofs;
        float *S = tmp_state.getPhiFull(ctx);
        for(CountT row = 0; row < num_dofs.numDofs; ++row) {
            float *S_col = S + 6 * row;
            for(CountT k = 0; k < 6; ++k) {
                tau[dof_index + row] += S_col[k] * tmp_state.sForce[k];
            }
        }

        // Add to parent's force
        if (hier_desc.parent != Entity::none()) {
            auto &parent_tmp_state = ctx.get<DofObjectTmpState>(hier_desc.parent);
            parent_tmp_state.sForce += tmp_state.sForce;
        }
    }
}

inline void processContacts(Context &ctx,
                            ContactConstraint &contact,
                            ContactTmpState &tmp_state)
{
    ObjectManager &obj_mgr = *ctx.singleton<ObjectData>().mgr;
    CVPhysicalComponent ref = ctx.get<CVPhysicalComponent>(
            contact.ref);
    CVPhysicalComponent alt = ctx.get<CVPhysicalComponent>(
            contact.alt);

    // If a parent collides with its direct child, unless the parent is a
    //  fixed body, we should ignore the contact.
    auto &refHier = ctx.get<DofObjectHierarchyDesc>(ref.physicsEntity);
    auto &altHier = ctx.get<DofObjectHierarchyDesc>(alt.physicsEntity);
    if (refHier.parent == alt.physicsEntity
        || altHier.parent == ref.physicsEntity) {
        auto &refNumDofs = ctx.get<DofObjectNumDofs>(ref.physicsEntity);
        auto &altNumDofs = ctx.get<DofObjectNumDofs>(alt.physicsEntity);
        if (refNumDofs.numDofs != (uint32_t)DofType::FixedBody
            && altNumDofs.numDofs != (uint32_t)DofType::FixedBody) {
            contact.numPoints = 0;
            return;
        }
    }

    CountT objID_i = ctx.get<ObjectID>(ref.physicsEntity).idx;
    CountT objID_j = ctx.get<ObjectID>(alt.physicsEntity).idx;

    // Create a coordinate system for the contact
    Vector3 n = contact.normal.normalize();
    Vector3 t{};

    Vector3 x_axis = {1.f, 0.f, 0.f};
    if(n.cross(x_axis).length() > 0.01f) {
        t = n.cross(x_axis).normalize();
    } else {
        t = n.cross({0.f, 1.f, 0.f}).normalize();
    }
    Vector3 s = n.cross(t).normalize();
    tmp_state.C[0] = n;
    tmp_state.C[1] = t;
    tmp_state.C[2] = s;

    // Get friction coefficient
    float mu = fminf(obj_mgr.metadata[objID_i].friction.muS,
                        obj_mgr.metadata[objID_j].friction.muS);
    tmp_state.mu = mu;
}

// TODO: This will require a complete rewrite for GPU version.
// Will keep for now for the sake of having first iteration
// of the numerical solver written.
//
// Renaming to brobdingnag for the time beingbecause it does a 
// ton of crap - will need to separate.
//
// (https://en.wikipedia.org/wiki/Brobdingnag)
inline void brobdingnag(Context &ctx,
                        CVSolveData &cv_sing)
{
    uint32_t world_id = ctx.worldID().idx;

    StateManager *state_mgr = getStateManager(ctx);
    PhysicsSystemState &physics_state = ctx.singleton<PhysicsSystemState>();

    // Recover necessary pointers.
    BodyGroupHierarchy *hiers = state_mgr->getWorldComponents<
        BodyGroup, BodyGroupHierarchy>(world_id);
    CountT num_grps = state_mgr->numRows<BodyGroup>(world_id);

    // Create complete mass matrix
    uint32_t total_num_dofs = 0;
    for (CountT i = 0; i < num_grps; ++i) {
        total_num_dofs += hiers[i].numDofs;
    }

    CountT num_mass_mat_bytes = sizeof(float) *
        total_num_dofs * total_num_dofs;
    // Row-major
    float *total_mass_mat = (float *)ctx.tmpAlloc(
            num_mass_mat_bytes);
    memset(total_mass_mat, 0, num_mass_mat_bytes);

    CountT num_full_dofs_bytes = sizeof(float) * total_num_dofs;
    float *full_free_acc = (float *)ctx.tmpAlloc(
            total_num_dofs * sizeof(float));
    float *full_vel = (float *)ctx.tmpAlloc(
            num_full_dofs_bytes);
    memset(full_free_acc, 0, num_full_dofs_bytes);
    memset(full_vel, 0, num_full_dofs_bytes);

    uint32_t processed_dofs = 0;
    for (CountT i = 0; i < num_grps; ++i) {
        float *local_mass = hiers[i].getMassMatrix(ctx);

        for (CountT row = 0; row < hiers[i].numDofs; ++row) {
            float *freeAcceleration = hiers[i].getBias(ctx);
            full_free_acc[row + processed_dofs] = freeAcceleration[row];

            for (CountT col = 0; col < hiers[i].numDofs; ++col) {
                uint32_t mi = row + processed_dofs;
                uint32_t mj = col + processed_dofs;

                // The total mass matrix is row major but the
                // local mass matrix is column major.
                total_mass_mat[mj + mi * total_num_dofs] =
                    local_mass[row + hiers[i].numDofs * col];

            }
        }

        processed_dofs += hiers[i].numDofs;
    }

    // Full velocity
    processed_dofs = 0;
    for (CountT i = 0; i < num_grps; ++i) {
        for (CountT j = 0; j < hiers[i].numBodies; ++j) {
            DofObjectVelocity vel = ctx.get<DofObjectVelocity>(
                    hiers[i].bodies(ctx)[j]);
            DofObjectNumDofs num_dofs = ctx.get<DofObjectNumDofs>(
                    hiers[i].bodies(ctx)[j]);
            for (CountT k = 0; k < num_dofs.numDofs; ++k) {
                full_vel[processed_dofs] = vel.qv[k];
                processed_dofs++;
            }
        }
    }

    // Create the contact Jacobian
    ContactConstraint *contacts = state_mgr->getWorldComponents<
        Contact, ContactConstraint>(world_id);
    ContactTmpState *contacts_tmp_state = state_mgr->getWorldComponents<
        Contact, ContactTmpState>(world_id);
    // Count contacts
    CountT num_contacts = state_mgr->numRows<Contact>(world_id);
    CountT total_contact_pts = 0;
    for (int i = 0; i < num_contacts; ++i) {
        total_contact_pts += contacts[i].numPoints;
    }

    // Process mu and penetrations for each point
    CountT num_full_contact_bytes = sizeof(float) * total_contact_pts;
    float *full_mu = (float *)ctx.tmpAlloc(
            num_full_contact_bytes);
    float *full_penetration = (float *)ctx.tmpAlloc(
            num_full_contact_bytes);
    CountT processed_pts = 0;
    for (int i = 0; i < num_contacts; ++i) {
        ContactTmpState &tmp_state = contacts_tmp_state[i];
        for (int j = 0; j < contacts[i].numPoints; ++j) {
            full_mu[processed_pts] = tmp_state.mu;
            full_penetration[processed_pts] = contacts[i].points[j].w;
            processed_pts++;
        }
    }

    uint32_t max_dofs = 0;

    // Prefix sum for each of the body groups
    uint32_t *block_start = (uint32_t *)ctx.tmpAlloc( 
            num_grps * sizeof(uint32_t));
    uint32_t block_offset = 0;

    for (CountT i = 0; i < num_grps; ++i) {
        block_start[i] = block_offset;
        block_offset += hiers[i].numDofs;
        hiers[i].tmpIdx = i;

        max_dofs = std::max(max_dofs, hiers[i].numDofs);
    }

    // Jacobian is size 3n_c x n_dofs, column-major
    uint32_t J_rows = 3 * total_contact_pts;
    uint32_t J_cols = total_num_dofs;

    float *J_c = (float *) ctx.tmpAlloc(
        J_rows * J_cols * sizeof(float));

    memset(J_c, 0, J_rows * J_cols * sizeof(float));

    float *J_c_body_scratch = (float *)ctx.tmpAlloc(
            3 * max_dofs * sizeof(float));

    CountT jac_row = 0;

    for(CountT ct_idx = 0; ct_idx < num_contacts; ++ct_idx) {
        ContactConstraint contact = contacts[ct_idx];
        ContactTmpState &tmp_state = contacts_tmp_state[ct_idx];

        auto ref = ctx.get<CVPhysicalComponent>(contact.ref);
        auto alt = ctx.get<CVPhysicalComponent>(contact.alt);
        auto &ref_num_dofs = ctx.get<DofObjectNumDofs>(ref.physicsEntity);
        auto &alt_num_dofs = ctx.get<DofObjectNumDofs>(alt.physicsEntity);

        bool ref_fixed = ref_num_dofs.numDofs == (uint32_t)DofType::FixedBody;
        bool alt_fixed = alt_num_dofs.numDofs == (uint32_t)DofType::FixedBody;

        // Each of the contact points
        for(CountT pt_idx = 0; pt_idx < contact.numPoints; pt_idx++) {
            Vector3 contact_pt = contact.points[pt_idx].xyz();

            // Compute the Jacobians for each body at the contact point
            if(!ref_fixed) {
                DofObjectHierarchyDesc &ref_hier_desc = ctx.get<DofObjectHierarchyDesc>(
                        ref.physicsEntity);
                BodyGroupHierarchy &ref_grp = ctx.get<BodyGroupHierarchy>(
                        ref_hier_desc.bodyGroup);

                float* J_ref = computeContactJacobian(ctx, ref_grp,
                    ref_hier_desc, tmp_state.C, contact_pt, J_c_body_scratch);

                for(CountT i = 0; i < ref_grp.numDofs; ++i) {

                    float *J_col = J_c +
                        J_rows * (block_start[ref_grp.tmpIdx] + i) + jac_row;

                    float *J_ref_col = J_ref + 3 * i;

                    for(CountT j = 0; j < 3; ++j) {
                        J_col[j] -= J_ref_col[j];
                    }
                }
            }
            if(!alt_fixed) {
                auto &alt_hier_desc = ctx.get<DofObjectHierarchyDesc>(
                        alt.physicsEntity);
                auto &alt_grp = ctx.get<BodyGroupHierarchy>(
                        alt_hier_desc.bodyGroup);
                
                float *J_alt = computeContactJacobian(ctx, alt_grp,
                    alt_hier_desc, tmp_state.C, contact_pt, J_c_body_scratch);

                for(CountT i = 0; i < alt_grp.numDofs; ++i) {
                    float *J_col = J_c +
                        J_rows * (block_start[alt_grp.tmpIdx] + i) + jac_row;

                    float *J_alt_col = J_alt + 3 * i;

                    for(CountT j = 0; j < 3; ++j) {
                        J_col[j] += J_alt_col[j];
                    }
                }
            }
            jac_row += 3;
        }
    }

    cv_sing.totalNumDofs = total_num_dofs;
    cv_sing.numContactPts = total_contact_pts;
    cv_sing.h = physics_state.h;
    cv_sing.mass = total_mass_mat;
    cv_sing.freeAcc = full_free_acc;
    cv_sing.vel = full_vel;
    cv_sing.J_c = J_c;
    cv_sing.mu = full_mu;
    cv_sing.penetrations = full_penetration;
    cv_sing.dofOffsets = block_start;
    cv_sing.numBodyGroups = num_grps;

    cv_sing.massDim = total_num_dofs;
    cv_sing.freeAccDim = total_num_dofs;
    cv_sing.velDim = total_num_dofs;
    cv_sing.numRowsJc = J_rows;
    cv_sing.numColsJc = J_cols;
    cv_sing.muDim = total_contact_pts;
    cv_sing.penetrationsDim = total_contact_pts;
}

inline void integrationStep(Context &ctx,
                            DofObjectPosition &position,
                            DofObjectVelocity &velocity,
                            DofObjectAcceleration &acceleration,
                            DofObjectNumDofs &numDofs)
{
    PhysicsSystemState &physics_state = ctx.singleton<PhysicsSystemState>();
    float h = physics_state.h;

    if (numDofs.numDofs == (uint32_t)DofType::FreeBody) {
        // Symplectic Euler
        for (int i = 0; i < 6; ++i) {
            velocity.qv[i] += h * acceleration.dqv[i];
        }
        for (int i = 0; i < 3; ++i) {
            position.q[i] += h * velocity.qv[i];
        }

        // From angular velocity to quaternion [Q_w, Q_x, Q_y, Q_z]
        Vector3 omega = { velocity.qv[3], velocity.qv[4], velocity.qv[5] };
        Quat rot_quat = { position.q[3], position.q[4], position.q[5], position.q[6], };
        Quat new_rot = {rot_quat.w, rot_quat.x, rot_quat.y, rot_quat.z};
        new_rot.w += 0.5f * h * (-rot_quat.x * omega.x - rot_quat.y * omega.y - rot_quat.z * omega.z);
        new_rot.x += 0.5f * h * (rot_quat.w * omega.x + rot_quat.z * omega.y - rot_quat.y * omega.z);
        new_rot.y += 0.5f * h * (-rot_quat.z * omega.x + rot_quat.w * omega.y + rot_quat.x * omega.z);
        new_rot.z += 0.5f * h * (rot_quat.y * omega.x - rot_quat.x * omega.y + rot_quat.w * omega.z);
        new_rot = new_rot.normalize();
        position.q[3] = new_rot.w;
        position.q[4] = new_rot.x;
        position.q[5] = new_rot.y;
        position.q[6] = new_rot.z;
    }
    else if (numDofs.numDofs == (uint32_t)DofType::Hinge) {
        velocity.qv[0] += h * acceleration.dqv[0];
        position.q[0] += h * velocity.qv[0];
    }
    else if (numDofs.numDofs == (uint32_t)DofType::FixedBody) {
        // Do nothing
    }
    else if (numDofs.numDofs == (uint32_t)DofType::Ball) {
        // Symplectic Euler
        for (int i = 0; i < 3; ++i) {
            velocity.qv[i] += h * acceleration.dqv[i];
        }

        // From angular velocity to quaternion [Q_w, Q_x, Q_y, Q_z]
        Vector3 omega = { velocity.qv[0], velocity.qv[1], velocity.qv[2] };
        Quat rot_quat = { position.q[0], position.q[1], position.q[2], position.q[3] };
        Quat new_rot = {rot_quat.w, rot_quat.x, rot_quat.y, rot_quat.z};
        new_rot.w += 0.5f * h * (-rot_quat.x * omega.x - rot_quat.y * omega.y - rot_quat.z * omega.z);
        new_rot.x += 0.5f * h * (rot_quat.w * omega.x + rot_quat.z * omega.y - rot_quat.y * omega.z);
        new_rot.y += 0.5f * h * (-rot_quat.z * omega.x + rot_quat.w * omega.y + rot_quat.x * omega.z);
        new_rot.z += 0.5f * h * (rot_quat.y * omega.x - rot_quat.x * omega.y + rot_quat.w * omega.z);
        new_rot = new_rot.normalize();
        position.q[0] = new_rot.w;
        position.q[1] = new_rot.x;
        position.q[2] = new_rot.y;
        position.q[3] = new_rot.z;
    }
    else {
        MADRONA_UNREACHABLE();
    }
}

// Convert all the generalized coordinates here.
inline void convertPostSolve(
        Context &ctx,
        Position &position,
        Rotation &rotation,
        const CVPhysicalComponent &phys)
{
    // TODO: use some forward kinematics results here
    Entity physical_entity = phys.physicsEntity;

    DofObjectNumDofs num_dofs = ctx.get<DofObjectNumDofs>(physical_entity);
    DofObjectTmpState tmp_state = ctx.get<DofObjectTmpState>(physical_entity);

    if (num_dofs.numDofs == (uint32_t)DofType::FreeBody) {
        position = tmp_state.comPos;
        rotation = tmp_state.composedRot;
    }
    else if (num_dofs.numDofs == (uint32_t)DofType::Hinge) {
        position = tmp_state.comPos;
        rotation = tmp_state.composedRot;
    }
    else if (num_dofs.numDofs == (uint32_t)DofType::FixedBody) {
        // Do nothing
    }
    else if (num_dofs.numDofs == (uint32_t)DofType::Ball) {
        position = tmp_state.comPos;
        rotation = tmp_state.composedRot;
    }
    else {
        MADRONA_UNREACHABLE();
    }
}

}

void registerTypes(ECSRegistry &registry)
{
    registry.registerComponent<CVPhysicalComponent>();

    registry.registerSingleton<CVSolveData>();

    registry.registerComponent<DofObjectPosition>();
    registry.registerComponent<DofObjectVelocity>();
    registry.registerComponent<DofObjectAcceleration>();
    registry.registerComponent<DofObjectNumDofs>();
    registry.registerComponent<DofObjectTmpState>();
    registry.registerComponent<DofObjectHierarchyDesc>();
    registry.registerComponent<ContactTmpState>();

    registry.registerArchetype<DofObjectArchetype>();
    registry.registerArchetype<Contact>();
    registry.registerArchetype<Joint>();

    registry.registerComponent<BodyGroupHierarchy>();
    registry.registerArchetype<BodyGroup>();

    registry.registerBundle<CVRigidBodyState>();
    registry.registerBundleAlias<SolverBundleAlias, CVRigidBodyState>();

    registry.registerMemoryRangeElement<MRElement128b>();

#if 0
    registry.registerMemoryRangeElement<PhiUnit>();
    registry.registerMemoryRangeElement<MassMatrixUnit>();
    registry.registerMemoryRangeElement<ParentArrayUnit>();
    registry.registerMemoryRangeElement<BodyFloatUnit>();
#endif
}

void setCVGroupRoot(Context &ctx,
                    Entity body_group,
                    Entity body)
{
    Entity physics_entity =
        ctx.get<CVPhysicalComponent>(body).physicsEntity;

    auto &hierarchy = ctx.get<DofObjectHierarchyDesc>(physics_entity);
    hierarchy.leaf = true;
    hierarchy.index = 0;
    hierarchy.parentIndex = -1;
    hierarchy.parent = Entity::none();
    hierarchy.bodyGroup = body_group;

    auto &body_grp_hier = ctx.get<BodyGroupHierarchy>(body_group);

    body_grp_hier.numBodies = 1;
    body_grp_hier.bodies(ctx)[0] = physics_entity;
    body_grp_hier.numDofs = ctx.get<DofObjectNumDofs>(physics_entity).numDofs;
}

void makeCVPhysicsEntity(Context &ctx, 
                         Entity e,
                         Position position,
                         Rotation rotation,
                         ObjectID obj_id,
                         DofType dof_type)
{
    Entity physical_entity = ctx.makeEntity<DofObjectArchetype>();

    auto &pos = ctx.get<DofObjectPosition>(physical_entity);
    auto &vel = ctx.get<DofObjectVelocity>(physical_entity);
    auto &acc = ctx.get<DofObjectAcceleration>(physical_entity);
    auto &tmp_state = ctx.get<DofObjectTmpState>(physical_entity);

#if 0
    auto &hierarchy = ctx.get<DofObjectHierarchyDesc>(physical_entity);
    hierarchy.leaf = true;
#endif

    switch (dof_type) {
    case DofType::FreeBody: {
        pos.q[0] = position.x;
        pos.q[1] = position.y;
        pos.q[2] = position.z;

        pos.q[3] = rotation.w;
        pos.q[4] = rotation.x;
        pos.q[5] = rotation.y;
        pos.q[6] = rotation.z;

        for(int i = 0; i < 6; i++) {
            vel.qv[i] = 0.f;
            acc.dqv[i] = 0.f;
        }
    } break;

    case DofType::Hinge: {
        pos.q[0] = 0.0f;
        vel.qv[0] = 0.f;
    } break;
    
    case DofType::Ball: {
        pos.q[0] = rotation.w;
        pos.q[1] = rotation.x;
        pos.q[2] = rotation.y;
        pos.q[3] = rotation.z;

        for(int i = 0; i < 3; i++) {
            vel.qv[i] = 0.f;
            acc.dqv[i] = 0.f;
        }
    } break;

    case DofType::FixedBody: {
        // Keep these around for forward kinematics
        pos.q[0] = position.x;
        pos.q[1] = position.y;
        pos.q[2] = position.z;

        pos.q[3] = rotation.w;
        pos.q[4] = rotation.x;
        pos.q[5] = rotation.y;
        pos.q[6] = rotation.z;

        for(int i = 0; i < 6; i++) {
            vel.qv[i] = 0.f;
            acc.dqv[i] = 0.f;
        }
    } break;
    }


    ctx.get<ObjectID>(physical_entity) = obj_id;
    ctx.get<DofObjectNumDofs>(physical_entity).numDofs = (uint32_t)dof_type;

    ctx.get<CVPhysicalComponent>(e) = {
        .physicsEntity = physical_entity,
    };

#if 0
#ifdef MADRONA_GPU_MODE
    static_assert(false, "Need to implement GPU DOF object hierarchy")
#else
    // By default, no parent
    hierarchy.parent = Entity::none();
#endif
#endif
}

void cleanupPhysicalEntity(Context &ctx, Entity e)
{
    CVPhysicalComponent physical_comp = ctx.get<CVPhysicalComponent>(e);
    ctx.destroyEntity(physical_comp.physicsEntity);
}

TaskGraphNodeID setupCVInitTasks(
        TaskGraphBuilder &builder,
        Span<const TaskGraphNodeID> deps)
{
    auto node = builder.addToGraph<ParallelForNode<Context,
         tasks::initHierarchies,
         BodyGroupHierarchy
     >>(deps);

    return node;
}

TaskGraphNodeID setupCVSolverTasks(TaskGraphBuilder &builder,
                                   TaskGraphNodeID broadphase,
                                   CountT num_substeps)
{
    auto cur_node = broadphase;

#ifdef MADRONA_GPU_MODE
    auto sort_sys = builder.addToGraph<
        SortArchetypeNode<BodyGroup, WorldID>>({cur_node});
    sort_sys = builder.addToGraph<
        SortArchetypeNode<DofObjectArchetype, WorldID>>({sort_sys});
    cur_node = builder.addToGraph<ResetTmpAllocNode>({sort_sys});
#endif

    for (CountT i = 0; i < num_substeps; ++i) {
        auto run_narrowphase = narrowphase::setupTasks(builder, {cur_node});

#ifdef MADRONA_GPU_MODE
        // We need to sort the contacts by world.
        run_narrowphase = builder.addToGraph<
            SortArchetypeNode<Contact, WorldID>>(
                {run_narrowphase});
        run_narrowphase = builder.addToGraph<ResetTmpAllocNode>(
                {run_narrowphase});
#endif

        auto compute_body_coms = builder.addToGraph<ParallelForNode<Context,
             tasks::computeBodyCOM,
                DofObjectTmpState,
                ObjectID,
                DofObjectNumDofs
            >>({run_narrowphase});

        auto compute_total_com = builder.addToGraph<ParallelForNode<Context,
             tasks::computeTotalCOM,
                BodyGroupHierarchy
            >>({run_narrowphase});

        auto compute_spatial_inertia = builder.addToGraph<ParallelForNode<Context,
             tasks::computeSpatialInertias,
                DofObjectTmpState,
                DofObjectHierarchyDesc,
                ObjectID,
                DofObjectNumDofs
            >>({compute_total_com});

        auto compute_phi = builder.addToGraph<ParallelForNode<Context,
             tasks::computePhiHierarchy,
                DofObjectTmpState,
                DofObjectNumDofs,
                DofObjectHierarchyDesc
            >>({compute_spatial_inertia});

        auto recursive_newton_euler = builder.addToGraph<ParallelForNode<Context,
             tasks::recursiveNewtonEuler,
                BodyGroupHierarchy
            >>({compute_phi});

        auto combine_spatial_inertia = builder.addToGraph<ParallelForNode<Context,
             tasks::combineSpatialInertias,
                BodyGroupHierarchy
            >>({recursive_newton_euler});

        auto composite_rigid_body = builder.addToGraph<ParallelForNode<Context,
             tasks::compositeRigidBody,
                BodyGroupHierarchy
            >>({combine_spatial_inertia});

        auto factorize_mass_matrix = builder.addToGraph<ParallelForNode<Context,
             tasks::factorizeMassMatrix,
                BodyGroupHierarchy
            >>({composite_rigid_body});

        auto compute_free_acc = builder.addToGraph<ParallelForNode<Context,
             tasks::computeFreeAcceleration,
                BodyGroupHierarchy
            >>({factorize_mass_matrix});

        auto contact_node = builder.addToGraph<ParallelForNode<Context,
             tasks::processContacts,
                ContactConstraint,
                ContactTmpState
            >>({compute_free_acc});

        auto thing = builder.addToGraph<ParallelForNode<Context,
             tasks::brobdingnag,
                CVSolveData
            >>({contact_node});

#ifdef MADRONA_GPU_MODE
        auto solve = builder.addToGraph<tasks::GaussMinimizationNode>(
                {thing});
#else
        auto solve = builder.addToGraph<ParallelForNode<Context,
             tasks::solveCPU,
                CVSolveData
            >>({thing});
#endif

        auto int_node = builder.addToGraph<ParallelForNode<Context,
             tasks::integrationStep,
                 DofObjectPosition,
                 DofObjectVelocity,
                 DofObjectAcceleration,
                 DofObjectNumDofs
            >>({solve});

        auto post_forward_kinematics = builder.addToGraph<ParallelForNode<Context,
             tasks::forwardKinematics,
                BodyGroupHierarchy
            >>({int_node});

        cur_node =
            builder.addToGraph<ParallelForNode<Context, tasks::convertPostSolve,
                Position,
                Rotation,
                CVPhysicalComponent
            >>({post_forward_kinematics});

        cur_node = builder.addToGraph<
            ClearTmpNode<Contact>>({cur_node});

        cur_node = builder.addToGraph<ResetTmpAllocNode>({cur_node});
    }

    auto clear_broadphase = builder.addToGraph<
        ClearTmpNode<CandidateTemporary>>({cur_node});
    
    return clear_broadphase;
}



void getSolverArchetypeIDs(uint32_t *contact_archetype_id,
                           uint32_t *joint_archetype_id)
{
    *contact_archetype_id = TypeTracker::typeID<Contact>();
    *joint_archetype_id = TypeTracker::typeID<Joint>();
}

void init(Context &ctx, CVXSolve *cvx_solve)
{
    ctx.singleton<CVSolveData>().cvxSolve = cvx_solve;
}

void setCVEntityParentHinge(Context &ctx,
                            Entity body_grp,
                            Entity parent, Entity child,
                            Vector3 rel_pos_parent,
                            Vector3 rel_pos_child,
                            Vector3 hinge_axis)
{
    Entity child_physics_entity =
        ctx.get<CVPhysicalComponent>(child).physicsEntity;
    Entity parent_physics_entity =
        ctx.get<CVPhysicalComponent>(parent).physicsEntity;

    BodyGroupHierarchy &grp = ctx.get<BodyGroupHierarchy>(body_grp);

    auto &hier_desc = ctx.get<DofObjectHierarchyDesc>(child_physics_entity);
    auto &parent_hier_desc =
        ctx.get<DofObjectHierarchyDesc>(parent_physics_entity);

    hier_desc.parent = parent_physics_entity;
    hier_desc.relPositionParent = rel_pos_parent;
    hier_desc.relPositionLocal = rel_pos_child;

    hier_desc.hingeAxis = hinge_axis;

    hier_desc.leaf = true;
    hier_desc.bodyGroup = body_grp;

    hier_desc.index = grp.numBodies;
    hier_desc.parentIndex = parent_hier_desc.index;


    grp.bodies(ctx)[grp.numBodies] = child_physics_entity;

    // Make the parent no longer a leaf
    ctx.get<DofObjectHierarchyDesc>(parent_physics_entity).leaf = false;

    ++grp.numBodies;
    grp.numDofs += ctx.get<DofObjectNumDofs>(child_physics_entity).numDofs;
}

void setCVEntityParentBall(Context &ctx,
                           Entity body_grp,
                           Entity parent, Entity child,
                           Vector3 rel_pos_parent,
                           Vector3 rel_pos_child)
{
    Entity child_physics_entity =
        ctx.get<CVPhysicalComponent>(child).physicsEntity;
    Entity parent_physics_entity =
        ctx.get<CVPhysicalComponent>(parent).physicsEntity;

    BodyGroupHierarchy &grp = ctx.get<BodyGroupHierarchy>(body_grp);

    auto &hier_desc = ctx.get<DofObjectHierarchyDesc>(child_physics_entity);
    auto &parent_hier_desc =
        ctx.get<DofObjectHierarchyDesc>(parent_physics_entity);

    hier_desc.parent = parent_physics_entity;
    hier_desc.relPositionParent = rel_pos_parent;
    hier_desc.relPositionLocal = rel_pos_child;

    hier_desc.leaf = true;
    hier_desc.bodyGroup = body_grp;

    hier_desc.index = grp.numBodies;
    hier_desc.parentIndex = parent_hier_desc.index;


    grp.bodies(ctx)[grp.numBodies] = child_physics_entity;

    // Make the parent no longer a leaf
    ctx.get<DofObjectHierarchyDesc>(parent_physics_entity).leaf = false;

    ++grp.numBodies;
    grp.numDofs += ctx.get<DofObjectNumDofs>(child_physics_entity).numDofs;
}

Entity makeCVBodyGroup(Context &ctx, uint32_t num_bodies)
{
    Entity e = ctx.makeEntity<BodyGroup>();

    auto &hier = ctx.get<BodyGroupHierarchy>(e);
    hier.numBodies = 0;

    uint64_t mr_num_bytes = num_bodies * sizeof(Entity);
    uint32_t num_elems = (mr_num_bytes + sizeof(MRElement128b) - 1) /
        sizeof(MRElement128b);
    hier.mrBodies = ctx.allocMemoryRange<MRElement128b>(num_elems);

    return e;
}

}
