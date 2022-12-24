#include <madrona/taskgraph.hpp>

#include "megakernel_consts.hpp"

#include <cub/block/block_radix_sort.cuh>
#include <cub/block/block_scan.cuh>

namespace madrona {

TaskGraph::Builder::Builder(int32_t max_num_nodes,
                            int32_t max_node_datas,
                            int32_t max_num_dependencies)
    : staged_((StagedNode *)rawAlloc(sizeof(StagedNode) * max_num_nodes)),
      num_nodes_(0),
      node_datas_((NodeData *)rawAlloc(sizeof(NodeData) * max_node_datas)),
      num_datas_(0),
      all_dependencies_((NodeID *)rawAlloc(sizeof(NodeID) * max_num_dependencies)),
      num_dependencies_(0)
{}

TaskGraph::Builder::~Builder()
{
    rawDealloc(all_dependencies_);
    rawDealloc(node_datas_),
    rawDealloc(staged_);
}

TaskGraph::NodeID TaskGraph::Builder::registerNode(
    uint32_t data_idx,
    uint32_t fixed_count,
    uint32_t func_id,
    Span<const TaskGraph::NodeID> dependencies)
{
    uint32_t offset = num_dependencies_;
    uint32_t num_deps = dependencies.size();

    num_dependencies_ += num_deps;

    for (int i = 0; i < (int)num_deps; i++) {
        all_dependencies_[offset + i] = dependencies[i];
    }

    int32_t node_idx = num_nodes_++;

    new (&staged_[node_idx]) StagedNode {
        {
            data_idx,
            fixed_count,
            func_id,
            0,
            0,
            0,
        },
        offset,
        num_deps,
    };

    return NodeID {
        node_idx,
    };
}

void TaskGraph::Builder::build(TaskGraph *out)
{
    assert(staged_[0].numDependencies == 0);
    Node *sorted_nodes = 
        (Node *)rawAlloc(sizeof(Node) * num_nodes_);
    bool *queued = (bool *)rawAlloc(num_nodes_ * sizeof(bool));

    uint32_t sorted_idx = 0;
    auto enqueueInSorted = [&](const Node &node) {
        new (&sorted_nodes[sorted_idx++]) Node {
            node.dataIDX,
            node.fixedCount,
            node.funcID,
            0, 0, 0,
        };
    };

    enqueueInSorted(staged_[0].node);

    queued[0] = true;

    uint32_t num_remaining_nodes = num_nodes_ - 1;
    uint32_t *remaining_nodes =
        (uint32_t *)rawAlloc(num_remaining_nodes * sizeof(uint32_t));

    for (int64_t i = 1; i < (int64_t)num_nodes_; i++) {
        queued[i]  = false;
        remaining_nodes[i - 1] = i;
    }

    while (num_remaining_nodes > 0) {
        uint32_t cur_node_idx;
        for (cur_node_idx = 0; queued[cur_node_idx]; cur_node_idx++) {}

        StagedNode &cur_staged = staged_[cur_node_idx];

        bool dependencies_satisfied = true;
        for (uint32_t dep_offset = 0; dep_offset < cur_staged.numDependencies;
             dep_offset++) {
            uint32_t dep_node_idx =
                all_dependencies_[cur_staged.dependencyOffset + dep_offset].id;
            if (!queued[dep_node_idx]) {
                dependencies_satisfied = false;
                break;
            }
        }

        if (dependencies_satisfied) {
            queued[cur_node_idx] = true;
            enqueueInSorted(cur_staged.node);
            num_remaining_nodes--;
        }
    }

    rawDealloc(remaining_nodes);
    rawDealloc(queued);

    auto tg_datas = (NodeData *)rawAlloc(sizeof(NodeData) * num_datas_);
    memcpy(tg_datas, node_datas_, sizeof(NodeData) * num_datas_);

    new (out) TaskGraph(sorted_nodes, num_nodes_, tg_datas);
}

ClearTmpNodeBase::ClearTmpNodeBase(uint32_t archetype_id)
    : NodeBase(),
      archetypeID(archetype_id)
{}

void ClearTmpNodeBase::run(int32_t)
{
    StateManager *state_mgr = mwGPU::getStateManager();
    state_mgr->clearTemporaries(archetypeID);
}

TaskGraph::NodeID ClearTmpNodeBase::addToGraph(
    TaskGraph::Builder &builder,
    Span<const TaskGraph::NodeID> dependencies,
    uint32_t archetype_id)
{
    return builder.addOneOffNode<ClearTmpNodeBase>(dependencies, archetype_id);
}

RecycleEntitiesNode::RecycleEntitiesNode()
    : NodeBase(),
      recycleBase(0)
{}

void RecycleEntitiesNode::run(int32_t invocation_idx)
{
    mwGPU::getStateManager()->recycleEntities(
        invocation_idx, recycleBase);
}

uint32_t RecycleEntitiesNode::numInvocations()
{
    auto [recycle_base, num_deleted] =
        mwGPU::getStateManager()->fetchRecyclableEntities();

    if (num_deleted > 0) {
        recycleBase = recycle_base;
    }

    return num_deleted;
}

TaskGraph::NodeID RecycleEntitiesNode::addToGraph(
    TaskGraph::Builder &builder,
    Span<const TaskGraph::NodeID> dependencies)
{
    return builder.addDynamicCountNode<RecycleEntitiesNode>(dependencies);
}

void ResetTmpAllocNode::run(int32_t)
{
    TmpAllocator::get().reset();
}

TaskGraph::NodeID ResetTmpAllocNode::addToGraph(
    TaskGraph::Builder &builder,
    Span<const TaskGraph::NodeID> dependencies)
{
    return builder.addOneOffNode<ResetTmpAllocNode>(dependencies);
}

CompactArchetypeNodeBase::CompactArchetypeNodeBase(uint32_t archetype_id)
    : NodeBase(),
      archetypeID(archetype_id)
{}

void CompactArchetypeNodeBase::run(int32_t invocation_idx)
{
#if 0
    uint32_t archetype_id = data.compactArchetype.archetypeID;
    StateManager *state_mgr = mwGPU::getStateManager();
#endif

    // Actually compact
    assert(false);
}

uint32_t CompactArchetypeNodeBase::numInvocations()
{
    assert(false);
    return mwGPU::getStateManager()->numArchetypeRows(archetypeID);
}

TaskGraph::NodeID CompactArchetypeNodeBase::addToGraph(
    TaskGraph::Builder &builder,
    Span<const TaskGraph::NodeID> dependencies,
    uint32_t archetype_id)
{
    return builder.addDynamicCountNode<CompactArchetypeNodeBase>(
        dependencies, archetype_id);
}

#if __CUDA_ARCH__ < 800
static uint32_t __reduce_add_sync(uint32_t mask, uint32_t val)
{
    uint32_t lane_id = threadIdx.x % 32;
#pragma unroll
    for (int i = 16; i > 0; i /= 2) {
        uint32_t read_lane = lane_id ^ i;

        bool other_active = mask & (1 << read_lane);

        if (!other_active) {
            read_lane = lane_id;
        }

        uint32_t other = __shfl_sync(mask, val, read_lane);

        if (other_active) {
            val += other;
        }
    }

    return val;
}
#endif

namespace sortConsts {
inline constexpr int RADIX_BITS = 8;
inline constexpr int RADIX_DIGITS = 1 << RADIX_BITS;
inline constexpr int ALIGN_BYTES = 256;
inline constexpr int MAX_NUM_PASSES =
    (sizeof(uint32_t) * 8 + RADIX_BITS - 1) / RADIX_BITS;
}

SortArchetypeNodeBase::SortArchetypeNodeBase(uint32_t archetype_id,
                                             int32_t num_passes,
                                             uint32_t *keys_col)
    :  NodeBase {},
       archetypeID(archetype_id),
       numPasses(num_passes),
       keysCol(keys_col)
{}

void SortArchetypeNodeBase::sortSetup(int32_t)
{
    using namespace sortConsts;

    StateManager *state_mgr = mwGPU::getStateManager();

    if (!state_mgr->archetypeNeedsSort(archetypeID)) {
        numDynamicInvocations = 0;
        return;
    }

    int num_rows = state_mgr->numArchetypeRows(archetypeID);

    int32_t num_threads =
        num_rows / num_elems_per_sort_thread_;

    uint32_t num_blocks = utils::divideRoundUp((uint32_t)num_threads,
        consts::numMegakernelThreads);

    uint32_t rounded_num_threads = num_blocks * consts::numMegakernelThreads;

    constexpr uint64_t onesweep_tile_items =
        consts::numMegakernelThreads * num_elems_per_sort_thread_;

    uint64_t bins_offset = 0;
    uint64_t total_bytes = bins_offset +
        uint64_t(numPasses * RADIX_DIGITS) * sizeof(uint32_t);

    uint64_t lookback_offset = utils::roundUpPow2(total_bytes, ALIGN_BYTES);
    total_bytes = lookback_offset + onesweep_tile_items;
    uint64_t keys_alt_offset = utils::roundUpPow2(total_bytes, ALIGN_BYTES);
    total_bytes = keys_alt_offset + uint64_t(num_rows) * sizeof(uint32_t);
    uint64_t indices_offset = utils::roundUpPow2(total_bytes, ALIGN_BYTES);
    total_bytes = indices_offset + uint64_t(num_rows) * sizeof(int);
    uint64_t indices_alt_offset =
        utils::roundUpPow2(total_bytes, ALIGN_BYTES);
    total_bytes = indices_alt_offset + uint64_t(num_rows) * sizeof(int);
    uint64_t counters_offset = utils::roundUpPow2(total_bytes, ALIGN_BYTES);
    total_bytes = counters_offset + uint64_t(numPasses) * sizeof(uint32_t);

    char *tmp_buffer = (char *)TmpAllocator::get().alloc(total_bytes);

    numRows = num_rows;
    numSortBlocks = num_blocks;
    numSortThreads = rounded_num_threads;
    bins = (uint32_t *)(tmp_buffer + bins_offset);
    lookback = (uint32_t *)(tmp_buffer + lookback_offset);
    keysAlt = (uint32_t *)(tmp_buffer + keys_alt_offset);
    indices = (int *)(tmp_buffer + indices_offset);
    indicesAlt = (int *)(tmp_buffer + indices_alt_offset);
    counters = (uint32_t *)(tmp_buffer + counters_offset);

    // Set launch count for next node that computes the histogram
    numDynamicInvocations = numSortThreads;
    // Zero counters
    for (int i = 0; i < numPasses; i++) {
        counters[i] = 0;
    }
}

void SortArchetypeNodeBase::zeroBins(int32_t invocation_idx)
{
    bins[invocation_idx] = 0;
}

void SortArchetypeNodeBase::histogram(int32_t invocation_idx)
{
    using namespace sortConsts;

    struct HistogramSMem {
        uint32_t bins[MAX_NUM_PASSES][RADIX_DIGITS];
    };

    auto smem_tmp = (HistogramSMem *)mwGPU::SharedMemStorage::buffer;

    constexpr int32_t block_items =
        consts::numMegakernelThreads * num_elems_per_sort_thread_;
    const int32_t block_idx = invocation_idx / consts::numMegakernelThreads;

    for (int pass = 0; pass < numPasses; pass++) {
        smem_tmp->bins[pass][threadIdx.x] = 0;
    }

    __syncthreads();

#pragma unroll
    for (int i = 0; i < num_elems_per_sort_thread_; i++) {
        int32_t row_idx = block_idx * block_items +
            i * consts::numMegakernelThreads + threadIdx.x;

        if (row_idx < numRows) {
            // Initialize indices while we're here
            indices[row_idx] = row_idx;

            int current_bit = 0;
            for (int pass = 0; pass < numPasses; pass++) {
                cub::ShiftDigitExtractor<uint32_t> digit_extractor(
                    current_bit, RADIX_BITS);

                int bin = digit_extractor.Digit(keysCol[row_idx]);
                atomicAdd(&smem_tmp->bins[pass][bin], 1);

                current_bit += RADIX_BITS;
            }
        }
    }

    __syncthreads();

    for (int pass = 0; pass < numPasses; pass++) {
        int32_t bin_count = smem_tmp->bins[pass][threadIdx.x];

        int smem_sum = __reduce_add_sync(0xFFFF'FFFF, bin_count);
        if (threadIdx.x % 32 == 0) {
            atomicAdd(&bins[pass * RADIX_DIGITS + threadIdx.x], smem_sum);
        }
    }
}

void SortArchetypeNodeBase::binScan(int32_t invocation_idx)
{
    using BlockScanT = cub::BlockScan<uint32_t, consts::numMegakernelThreads>;
    using SMemTmpT = typename BlockScanT::TempStorage;
    auto smem_tmp = (SMemTmpT *)mwGPU::SharedMemStorage::buffer;

    uint32_t bin_vals[1];
    bin_vals[0] = bins[invocation_idx];

    BlockScanT(*smem_tmp).ExclusiveSum(bin_vals, bin_vals);

    bins[invocation_idx] = bin_vals[0];

    // Setup for prepareOnesweep
    if (invocation_idx == 0) {
        numDynamicInvocations = numSortBlocks;
    }
}

void SortArchetypeNodeBase::prepareOnesweep(int32_t invocation_idx)
{
    // Zero out the lookback counters
    lookback[invocation_idx]  = 0;
}

TaskGraph::NodeID SortArchetypeNodeBase::addToGraph(
    TaskGraph::Builder &builder,
    Span<const TaskGraph::NodeID> dependencies,
    uint32_t archetype_id,
    int32_t component_id)
{
    using namespace mwGPU;

    StateManager *state_mgr = getStateManager();
    auto keys_col = 
        (uint32_t *)state_mgr->getArchetypeColumn(archetype_id, component_id);

    // Optimize for sorts on the WorldID column, where the 
    // max # of worlds is known
    int32_t num_passes;
    if (component_id == TypeTracker::typeID<WorldID>()) {
        int32_t num_worlds = GPUImplConsts::get().numWorlds;
        // num_worlds + 1 to leave room for columns with WorldID == -1
        int32_t num_bits = 32 - __clz(num_worlds + 1);

        num_passes = utils::divideRoundUp(num_bits, 8);
    } else {
        num_passes = 4;
    }

    uint32_t num_histogram_bins = num_passes * consts::numMegakernelThreads;

    auto &node = builder.constructNodeData<SortArchetypeNodeBase>(
        archetype_id, num_passes, keys_col);

    TaskGraph::NodeID setup = builder.addNodeFn<
        &SortArchetypeNodeBase::sortSetup>(node, dependencies, 1);

    auto zero_bins = builder.addNodeFn<
        &SortArchetypeNodeBase::zeroBins>(node, {setup}, num_histogram_bins);

    auto compute_histogram = builder.addNodeFn<
        &SortArchetypeNodeBase::histogram>(node, {zero_bins});

    auto binscan = builder.addNodeFn<&SortArchetypeNodeBase::binScan>(
        node, {compute_histogram}, num_histogram_bins);

    return binscan;
}

}
