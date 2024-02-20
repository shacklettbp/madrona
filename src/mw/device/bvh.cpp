// Dummy stuff we don't need but certain headers need
#define MADRONA_MWGPU_MAX_BLOCKS_PER_SM 4
#define MADRONA_WARP_SIZE 32

#define MADRONA_DEBUG_TEST
#define MADRONA_DEBUG_TEST_NUM_LEAVES 11
#define MADRONA_TREELET_SIZE 7

#include <atomic>
#include <algorithm>
#include <madrona/bvh.hpp>
#include <madrona/math.hpp>
#include <madrona/memory.hpp>

using namespace madrona;

namespace sm {
extern __shared__ uint8_t buffer[];

// For the initial BVH build.
struct BuildFastBuffer {
    uint32_t blockNodeOffset;
    uint32_t totalNumInstances;

    // This holds the range of morton codes which are in memory.
    uint32_t mortonCodesStart;
    uint32_t mortonCodesEnd;

    char buffer[1];
};

// For optimizing the initial BVH build with treelet rotations.
struct OptFastBuffer {
    uint32_t blockNodeOffset;
    uint32_t totalNumInstances;
    char buffer[1];
};

struct OptFastBufferTreelets {
    uint32_t blockNodeOffset;
    uint32_t totalNumInstances;
    AtomicU32 treeletCounter;

    // Stores the treelets
    char buffer[1];
};

// After the initial treelets get formed, we might have to prune certain
// leaves because each treelet may initially have more than
// MADRONA_TREELET_SIZE nodes.
struct InitialTreelet {
    int32_t rootIndex;
};

}

extern "C" {
    __constant__ BVHParams bvhParams;
}

extern "C" __global__ void bvhInit()
{
}

// Because the number of instances / views isn't going to be known when the
// CPU launches this kernel (it just gets put into a CUgraph and run every
// frame), we are just going to max out the GPU and use a persistent thread
// approach to make sure all the work gets done.

// Stages of the top-level BVH build and ray cast
// 1) Generate the internal nodes
// 2) Optimize the BVH
extern "C" __global__ void bvhAllocInternalNodes()
{
    BVHInternalData *internal_data = bvhParams.internalData;

    // We need to make sure we have enough internal nodes for the initial
    // 2-wide BVH which gets constructed before the optimized tree
    uint32_t num_instances = bvhParams.instanceOffsets[bvhParams.numWorlds-1] +
                             bvhParams.instanceCounts[bvhParams.numWorlds-1];

    // For the 2-wide tree, we need about num_instances internal nodes
    uint32_t num_required_nodes = num_instances;
    uint32_t num_bytes = num_required_nodes * 
        (2 * sizeof(LBVHNode) + sizeof(TreeletFormationNode));

    mwGPU::TmpAllocator *allocator = (mwGPU::TmpAllocator *)
        bvhParams.tmpAllocator;

    auto *ptr = allocator->alloc(num_bytes);
    printf("From allocInternalNode: tmp allocated: %p\n", ptr);
    printf("From allocInternalNode: internal data at: %p\n", internal_data);
    printf("There are %d total instances\n", (int32_t)num_instances);

    internal_data->internalNodes = (LBVHNode *)ptr;
    internal_data->numAllocatedNodes = num_required_nodes;
    internal_data->buildFastAccumulator.store_relaxed(0);

    internal_data->leaves = (LBVHNode *)((char *)ptr +
            num_required_nodes * sizeof(LBVHNode));
    internal_data->numAllocatedLeaves = num_required_nodes;
    internal_data->optFastAccumulator.store_relaxed(0);

    internal_data->treeletFormNodes = (TreeletFormationNode *)
        ((char *)ptr + 2 * num_required_nodes * sizeof(LBVHNode))

#if defined(MADRONA_DEBUG_TEST)
    // We are going to set up a test case here from the paper
    uint32_t *codes = bvhParams.mortonCodes;

    codes[0] = 1;
    codes[1] = 3;
    codes[2] = 11;
    codes[3] = 19;
    codes[4] = 25;
    codes[5] = 39;
    codes[6] = 57;
    codes[7] = 90;
    codes[8] = 121;
    codes[9] = 251;
    codes[10] = 253;

    codes[MADRONA_DEBUG_TEST_NUM_LEAVES+0] = 1;
    codes[MADRONA_DEBUG_TEST_NUM_LEAVES+1] = 3;
    codes[MADRONA_DEBUG_TEST_NUM_LEAVES+2] = 11;
    codes[MADRONA_DEBUG_TEST_NUM_LEAVES+3] = 19;
    codes[MADRONA_DEBUG_TEST_NUM_LEAVES+4] = 25;
    codes[MADRONA_DEBUG_TEST_NUM_LEAVES+5] = 39;
    codes[MADRONA_DEBUG_TEST_NUM_LEAVES+6] = 57;
    codes[MADRONA_DEBUG_TEST_NUM_LEAVES+7] = 90;
    codes[MADRONA_DEBUG_TEST_NUM_LEAVES+8] = 121;
    codes[MADRONA_DEBUG_TEST_NUM_LEAVES+9] = 251;
    codes[MADRONA_DEBUG_TEST_NUM_LEAVES+10] = 253;

    for (int i = 0; i < MADRONA_DEBUG_TEST_NUM_LEAVES; ++i) {
        uint32_t code = codes[i];
        printf(USHORT_TO_BINARY_PATTERN " ", USHORT_TO_BINARY((code>>16)));
        printf(USHORT_TO_BINARY_PATTERN " \t", USHORT_TO_BINARY((code)));

        printf("(Leaf node %d): \t (%d)\n", 
                i, 
                codes[i]);
    }
#endif
}

namespace bits {
// length of the longest common prefix
int32_t llcp(uint32_t a, uint32_t b)
{
    return (int32_t)__clz(a ^ b);
}

int32_t sign(int32_t x)
{
    uint32_t x_u32 = *((uint32_t *)&x);
    return 1 - (int32_t)(x_u32 >> 31) * 2;
}

int32_t ceil_div(int32_t a, int32_t b)
{
    return (a + b-1) / b;
}
}

// We are going to have each thread be responsible for a single internal node
extern "C" __global__ void bvhBuildFast()
{
    BVHInternalData *internal_data = bvhParams.internalData;
    sm::BuildFastBuffer *smem = (sm::BuildFastBuffer *)sm::buffer;

    const uint32_t threads_per_block = blockDim.x;

    if (threadIdx.x == 0) {
        uint32_t node_offset = internal_data->buildFastAccumulator.fetch_add<
            sync::memory_order::relaxed>(threads_per_block);
        uint32_t num_instances = bvhParams.instanceOffsets[bvhParams.numWorlds-1] +
                                 bvhParams.instanceCounts[bvhParams.numWorlds-1];

        smem->blockNodeOffset = node_offset;
        smem->totalNumInstances = num_instances;
    }

    __syncthreads();

    // For now, each thread is individually processing an internal node
    uint32_t thread_offset = smem->blockNodeOffset + threadIdx.x;
    if (thread_offset >= smem->totalNumInstances) {
        return;
    }

#if defined(MADRONA_DEBUG_TEST)
    if (thread_offset >= 2 * MADRONA_DEBUG_TEST_NUM_LEAVES) {
        return;
    }
#endif

    // Load a bunch of morton codes into shared memory.
    //
    // TODO: Profile the difference between directly loading the morton codes
    // from memory vs first loading them into shared memory and operating on
    // them there. Problem is, would need an edge case for access morton codes
    // outside of bounds.
    //
    // We'd have to get the min/max world ID handled by this thread block and
    // load all the morton codes from the worlds within that range.
    //
    // However, the complication lies in the case where 

    // Number of leaves in the world of this thread
    //
    struct {
        uint32_t idx;
        uint32_t numInternalNodes;
        uint32_t internalNodesOffset;
        uint32_t numLeaves;
    } world_info;

    world_info.idx = bvhParams.instances[thread_offset].worldIDX;
    world_info.numLeaves = bvhParams.instanceCounts[world_info.idx];
    world_info.numInternalNodes = world_info.numLeaves - 1;
    world_info.internalNodesOffset = bvhParams.instanceOffsets[world_info.idx];

#if defined(MADRONA_DEBUG_TEST)
    world_info.idx = 0;
    world_info.numLeaves = MADRONA_DEBUG_TEST_NUM_LEAVES;
    world_info.numInternalNodes = MADRONA_DEBUG_TEST_NUM_LEAVES-1;
    world_info.internalNodesOffset = 0;
    
    if (thread_offset >= MADRONA_DEBUG_TEST_NUM_LEAVES) {
        world_info.internalNodesOffset = MADRONA_DEBUG_TEST_NUM_LEAVES;
        world_info.idx = 1;
    }
#endif

    // The offset into the nodes of the world this thread is dealing with
    int32_t tn_offset = thread_offset - world_info.internalNodesOffset;

    // Both the nodes and the leaves use the same offset
    LBVHNode *nodes = internal_data->internalNodes +
                      world_info.internalNodesOffset;
    LBVHNode *leaves = internal_data->internalNodes + 
                       world_info.internalNodesOffset;

    if (tn_offset >= world_info.numInternalNodes) {
        nodes[tn_offset].left = -1;
        nodes[tn_offset].right = -1;
        return;
    }

    printf("(thread_offset = %d) tn_offset = %d | internalNodesOffset = %d | numInternalNodes = %d\n",
            thread_offset, tn_offset, world_info.internalNodesOffset, world_info.numInternalNodes);

    // For now, we load things directly from global memory which sucks.
    // Need to try the strategy from the TODO
    auto llcp_nodes = [&world_info](int32_t i, int32_t j) {
        if (j >= world_info.numLeaves || j < 0) {
            return -1;
        }

        int32_t llcp = bits::llcp(bvhParams.mortonCodes[i+world_info.internalNodesOffset],
                bvhParams.mortonCodes[j+world_info.internalNodesOffset]);

        if (llcp == 8 * sizeof(uint32_t)) {
            return llcp + bits::llcp(i, j);
        } else {
            return llcp;
        }
    };

    int32_t direction = bits::sign(llcp_nodes(tn_offset, tn_offset+1) -
                                   llcp_nodes(tn_offset, tn_offset-1));
    int32_t llcp_min = llcp_nodes(tn_offset, tn_offset - direction);
    int32_t length_max = 2;

    while (llcp_nodes(tn_offset, tn_offset + length_max * direction) > llcp_min) {
        length_max *= 2;
    }

    int32_t true_length = 0;

    for (int32_t t = length_max / 2; t >= 1; t /= 2) {
        if (llcp_nodes(tn_offset, 
                       tn_offset + (true_length + t) * direction) > llcp_min) {
            true_length += t;
        }
    }

    int32_t other_end = tn_offset + true_length * direction;

#if defined (MADRONA_DEBUG_TEST)
    printf("tn_offset %d: direction=%d = %d - %d | true_length %d | other_end %d\n", 
            tn_offset, direction,
            llcp_nodes(tn_offset, tn_offset+1),
            llcp_nodes(tn_offset, tn_offset-1),
            true_length,
            other_end);
#endif

    // The number of common bits for all leaves coming out of this node
    int32_t node_llcp = llcp_nodes(tn_offset, other_end);

    // Relative to the tn_offset
    int32_t rel_split_offset = 0;
    
    for (int32_t divisor = 2, t = bits::ceil_div(true_length, divisor);
            t >= 1; (divisor *= 2), t = bits::ceil_div(true_length, divisor)) {
        if (llcp_nodes(tn_offset, tn_offset + 
                    (rel_split_offset + t) * direction) > node_llcp) {
            rel_split_offset += t;
        }
    }

    int32_t split_index = tn_offset + rel_split_offset * direction +
        std::min(direction, 0);

    int32_t left_index = std::min(tn_offset, other_end);
    int32_t right_index = std::max(tn_offset, other_end);

    if (left_index == split_index) {
        // The left node is a leaf and the leaf's index is split_index
        nodes[tn_offset].left = -split_index;

        // TODO: Calculate the AABB
        leaves[split_index].aabb = {};
        leaves[split_index].parent = tn_offset;
    } else {
        // The left node is an internal node and its index is split_index
        nodes[tn_offset].left = split_index;
    }
    
    if (right_index == split_index + 1) {
        // The right node is a leaf and the leaf's index is split_index+1
        nodes[tn_offset].right = -(split_index+1);

        // TODO: Calculate the AABB
        leaves[split_index+1].aabb = {};
        leaves[split_index+1].parent = tn_offset;
    } else {
        // The right node is an internal node and its index is split_index+1
        nodes[tn_offset].right = split_index+1;
    }
}

// Each warp will maintain a single treelet
extern "C" __global__ void bvhOptimizeLBVH()
{
    BVHInternalData *internal_data = bvhParams.internalData;

    // Phase 1 shared memory layout
    sm::OptFastBuffer *smem_p1 = (sm::OptFastBuffer *)sm::buffer;
    
    const uint32_t threads_per_block = blockDim.x;
    const uint32_t warps_per_block = threads_per_block / MADRONA_WARP_SIZE;

    if (threadIdx.x == 0) {
        uint32_t node_offset = internal_data->optFastAccumulator.fetch_add<
            sync::memory_order::relaxed>(threads_per_block);
        uint32_t num_instances = bvhParams.instanceOffsets[bvhParams.numWorlds-1] +
                                 bvhParams.instanceCounts[bvhParams.numWorlds-1];

        smem_p1->blockNodeOffset = node_offset;
        smem_p1->totalNumInstances = num_instances;
    }

    __syncthreads();

    if (threadIdx.x < smem_p1->totalNumInstances) {
        internal_data->treeletFormNodes[threadIdx.x].numLeaves.store<
            sync::memory_order::relaxed>(0);
        internal_data->treeletFormNodes[threadIdx.x].numReached.store<
            sync::memory_order::relaxed>(0);
    }

    __syncthreads();

    uint32_t thread_offset = smem_p1->blockNodeOffset + threadIdx.x;

    struct {
        uint32_t idx;
        uint32_t numInternalNodes;
        uint32_t internalNodesOffset;
        uint32_t numLeaves;
    } world_info;

    // Fill in world info:
    world_info.idx = bvhParams.instances[thread_offset].worldIDX;
    world_info.numLeaves = bvhParams.instanceCounts[world_info.idx];
    world_info.numInternalNodes = world_info.numLeaves - 1;
    world_info.internalNodesOffset = bvhParams.instanceOffsets[world_info.idx];

    // TODO: Need to merge the LeafNode and LBVHNode (internal node)
    // definitions into one such that we can repeat multiple iterations of the
    // treelet formation algorithm.

    // We first designate treelet roots
    sm::TreeletFormationTmp *treelet_nodes = 
        internal_data->treeletFormNodes + world_info.internalNodesOffset;

    // Each thread starts at a given leaf node
    LBVHNode *world_leaves = internal_data->leaves +
        world_info.internalNodesOffset;
    LBVHNode *world_inodes = internal_data->internalNodes +
        world_info.internalNodesOffset;

    // This thread's leaf (tn_offset = thread node offset)
    uint32_t tn_offset = thread_offset - world_info.internalNodesOffset;
    LBVHNode *current = &world_leaves[tn_offset];
    uint32_t num_leaves = 1;

    // TODO: Find break condition here
    while (num_leaves >= MADRONA_TREELET_SIZE) {
        int32_t parent = current->parent;

        sm::TreeletFormationTmp *parent_form = treelet_nodes + parent;
        parent_form->numLeaves.fetch_add_release(num_leaves);

        if (parent_form->numReached.exchange<
                sync::memory_order::relaxed>(1) == 0) {
            // Suspend this thread
            goto suspended_threads;
        } else {
            num_leaves += parent_form->numLeaves.load_acquire();
            current = &world_inodes[parent];
        }
    }

    // Only the threads which survived push their treelets to shared memory
    // (phase 2 shared memory layout).
    sm::OptFastBufferTreelets *smem_p2 = 
        (sm::OptFastBufferTreelets *)sm::buffer;
    sm::InitialTreelet *treelets_buffer = (sm::InitialTreelet *)
        smem_p2->buffer;

    uint32_t treelet_idx = smem_p2->treeletCounter.fetch_add_relaxed(1);

    sm::InitialTreelet initial_treelet = {
        // `current` will be relative to the world_inodes array after treelet
        // formation happens
        .rootIndex = current - world_inodes
    };

    treelets_buffer[treelet_idx] = initial_treelet;

suspended_threads:
    __syncthreads();

    // Now, we post process the treelets to prune leaves if needed (pruning
    // will happen based on surface area heuristic - we remove all leaves with
    // the lowest cost until treelet size is MADRONA_TREELET_SIZE)
}

extern "C" __global__ void bvhDebug()
{
#if 1
    BVHInternalData *internal_data = bvhParams.internalData;

    uint32_t num_instances = bvhParams.instanceOffsets[bvhParams.numWorlds-1] +
                             bvhParams.instanceCounts[bvhParams.numWorlds-1];

#if defined(MADRONA_DEBUG_TEST)
    for (int i = 0; i < MADRONA_DEBUG_TEST_NUM_LEAVES; ++i) {
#else
    for (int i = 0; i < num_instances; ++i) {
#endif
        render::InstanceData &instance_data = bvhParams.instances[i];
        LBVHNode *node = &internal_data->internalNodes[i];
        uint32_t offset = bvhParams.instanceOffsets[instance_data.worldIDX];

        printf("(Internal node %d) %d: left: %d, right: %d\n",
               i - offset,
               instance_data.worldIDX,
               node->left, node->right);
    }
#endif
}
