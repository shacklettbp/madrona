// Dummy stuff we don't need but certain headers need
#define MADRONA_MWGPU_MAX_BLOCKS_PER_SM 4
#define MADRONA_WARP_SIZE 32

// #define MADRONA_DEBUG_TEST
#define MADRONA_DEBUG_TEST_NUM_LEAVES 11
#define MADRONA_TREELET_SIZE 7

#include <atomic>
#include <algorithm>
#include <madrona/bvh.hpp>
#include <madrona/math.hpp>
#include <madrona/memory.hpp>
#include <madrona/mesh_bvh.hpp>

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

struct ConstructAABBBuffer {
    uint32_t blockNodeOffset;
    uint32_t totalNumInstances;

    char buffer[1];
};

// After the initial treelets get formed, we might have to prune certain
// leaves because each treelet may initially have more than
// MADRONA_TREELET_SIZE nodes.
struct InitialTreelet {
    int32_t rootIndex;
    int32_t worldIndex;
    int32_t numLeaves;
};

struct FormedTreelet {
    uint32_t rootIndex;
    uint32_t worldIndex;
    int32_t leaves[MADRONA_TREELET_SIZE];
};

union Treelet {
    InitialTreelet initial;
    FormedTreelet formed;
};

struct OptFastBufferTreelets {
    uint32_t blockNodeOffset;
    uint32_t totalNumInstances;
    AtomicU32 treeletCounter;

    // Stores the treelets
    Treelet buffer[1];
};

}

extern "C" {
    __constant__ BVHParams bvhParams;
}

extern "C" __global__ void bvhInit()
{
    // bvhParams.internal_data->numFrames = 0;
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
    printf("render resolution %d\n", bvhParams.renderOutputResolution);
    printf("pixels are at %p\n", bvhParams.renderOutput);

    bvhParams.timingInfo->timingCounts.store_relaxed(0);
    bvhParams.timingInfo->tlasTime.store_relaxed(0);
    bvhParams.timingInfo->blasTime.store_relaxed(0);

    BVHInternalData *internal_data = bvhParams.internalData;

    // We need to make sure we have enough internal nodes for the initial
    // 2-wide BVH which gets constructed before the optimized tree
    uint32_t num_instances = bvhParams.instanceOffsets[bvhParams.numWorlds-1] +
                             bvhParams.instanceCounts[bvhParams.numWorlds-1];

    // For the 2-wide tree, we need about num_instances internal nodes
    uint32_t num_required_nodes = num_instances;
    uint32_t num_bytes = num_required_nodes * 
        (2 * sizeof(LBVHNode) + sizeof(TreeletFormationNode) +
         sizeof(uint32_t));

    mwGPU::TmpAllocator *allocator = (mwGPU::TmpAllocator *)
        bvhParams.tmpAllocator;

    auto *ptr = allocator->alloc(num_bytes);

    internal_data->internalNodes = (LBVHNode *)ptr;
    internal_data->numAllocatedNodes = num_required_nodes;
    internal_data->buildFastAccumulator.store_relaxed(0);
    internal_data->constructAABBsAccumulator.store_relaxed(0);

    internal_data->leaves = (LBVHNode *)((char *)ptr +
            num_required_nodes * sizeof(LBVHNode));
    internal_data->numAllocatedLeaves = num_required_nodes;
    internal_data->optFastAccumulator.store_relaxed(0);

    internal_data->treeletFormNodes = (TreeletFormationNode *)
        ((char *)ptr + 2 * num_required_nodes * sizeof(LBVHNode));

    internal_data->treeletRootIndices = (uint32_t *)
        ((char *)ptr + 2 * num_required_nodes * sizeof(LBVHNode)
                     + num_required_nodes * sizeof(TreeletFormationNode));

    internal_data->treeletRootIndexCounter.store_relaxed(0);

    bvhParams.internalData->numFrames++;
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

    // The offset into the nodes of the world this thread is dealing with
    int32_t tn_offset = thread_offset - world_info.internalNodesOffset;

    // Both the nodes and the leaves use the same offset
    LBVHNode *nodes = internal_data->internalNodes +
                      world_info.internalNodesOffset;
    LBVHNode *leaves = internal_data->leaves + 
                       world_info.internalNodesOffset;
    TreeletFormationNode *treelet_form_nodes = internal_data->treeletFormNodes +
                                               world_info.internalNodesOffset;

    nodes[tn_offset].left = 0;
    nodes[tn_offset].right = 0;
    nodes[tn_offset].parent = 0;

    if (tn_offset == 0) {
        nodes[tn_offset].parent = 0xFFFF'FFFF;
        // printf("Root at %p\n", &nodes[tn_offset]);
    }

    if (tn_offset >= world_info.numInternalNodes) {
        return;
    }

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
        nodes[tn_offset].left = LBVHNode::childIdxToStoreIdx(split_index, true);
        nodes[tn_offset].reachedCount.store_relaxed(0);

        leaves[split_index].parent = tn_offset;
        uint32_t instance_idx = (leaves[split_index].instanceIdx = split_index +
            world_info.internalNodesOffset);

        leaves[split_index].aabb = bvhParams.aabbs[instance_idx].aabb;
        leaves[split_index].reachedCount.store_relaxed(0);
    } else {
        // The left node is an internal node and its index is split_index
        nodes[tn_offset].left = LBVHNode::childIdxToStoreIdx(split_index, false);
        nodes[tn_offset].reachedCount.store_relaxed(0);
        nodes[split_index].parent = tn_offset;
    }
    
    if (right_index == split_index + 1) {
        // The right node is a leaf and the leaf's index is split_index+1
        nodes[tn_offset].right = LBVHNode::childIdxToStoreIdx(split_index + 1, true);
        nodes[tn_offset].reachedCount.store_relaxed(0);

        leaves[split_index+1].parent = tn_offset;
        uint32_t instance_idx = (leaves[split_index+1].instanceIdx = split_index + 1 +
            world_info.internalNodesOffset);
        leaves[split_index+1].aabb = bvhParams.aabbs[instance_idx].aabb;
        leaves[split_index+1].reachedCount.store_relaxed(0);
    } else {
        // The right node is an internal node and its index is split_index+1
        nodes[tn_offset].right = LBVHNode::childIdxToStoreIdx(split_index + 1, false);
        nodes[tn_offset].reachedCount.store_relaxed(0);
        nodes[split_index + 1].parent = tn_offset;
    }

    treelet_form_nodes[tn_offset].numLeaves.store_relaxed(0);
    treelet_form_nodes[tn_offset].numReached.store_relaxed(0);
}

// Constructs the internal nodes' AABBs
//
// TODO: Create outer loop for persistent threads.
extern "C" __global__ void bvhConstructAABBs()
{
    BVHInternalData *internal_data = bvhParams.internalData;
    sm::ConstructAABBBuffer *smem = (sm::ConstructAABBBuffer *)sm::buffer;

    const uint32_t threads_per_block = blockDim.x;

    if (threadIdx.x == 0) {
        uint32_t node_offset = internal_data->constructAABBsAccumulator.fetch_add<
            sync::memory_order::relaxed>(threads_per_block);
        uint32_t num_instances = bvhParams.instanceOffsets[bvhParams.numWorlds-1] +
                                 bvhParams.instanceCounts[bvhParams.numWorlds-1];

        smem->blockNodeOffset = node_offset;
        smem->totalNumInstances = num_instances;
    }

    __syncthreads();

    uint32_t thread_offset = smem->blockNodeOffset + threadIdx.x;

    if (thread_offset >= smem->totalNumInstances) {
        return;
    }

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

    int32_t tn_offset = thread_offset - world_info.internalNodesOffset;

    LBVHNode *internal_nodes = internal_data->internalNodes +
                               world_info.internalNodesOffset;
    LBVHNode *leaves = internal_data->leaves +
                       world_info.internalNodesOffset;

    LBVHNode *current = &leaves[tn_offset];

    if (current->isInvalid() || tn_offset >= world_info.numLeaves) {
        return;
    }

    current = &internal_nodes[current->parent];

    // If the value before the add is 0, then we are the first to hit this node.
    // => we need to break out of the loop.
    while (current->reachedCount.fetch_add_release(1) == 1) {
        // Merge the AABBs of the children nodes. (Store like this for now to
        // help when we make the tree 4 wide or 8 wide.
        LBVHNode *children[2];
        bool are_leaves[2];

        if (current->left == 0) {
            children[0] = nullptr;
        } else {
            bool is_leaf;
            int32_t node_idx = LBVHNode::storeIdxToChildIdx(
                    current->left, is_leaf);

            if (is_leaf) {
                children[0] = &leaves[node_idx];
                are_leaves[0] = true;
            } else {
                children[0] = &internal_nodes[node_idx];
                are_leaves[0] = false;
            }
        }

        if (current->right == 0) {
            children[1] = nullptr;
        } else {
            bool is_leaf;
            int32_t node_idx = LBVHNode::storeIdxToChildIdx(
                    current->right, is_leaf);

            if (is_leaf) {
                children[1] = &leaves[node_idx];
                are_leaves[1] = true;
            } else {
                children[1] = &internal_nodes[node_idx];
                are_leaves[1] = false;
            }
        }

        bool got_valid_node = false;
        math::AABB merged_aabb;

        // Merge the AABBs
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            LBVHNode *node = children[i];

            if (node) {
                if (!got_valid_node) {
                    got_valid_node = true;
                    merged_aabb = node->aabb;
                } else {
                    merged_aabb = math::AABB::merge(node->aabb, merged_aabb);
                }
            }
        }

        current->aabb = merged_aabb;

        if (current->parent == 0xFFFF'FFFF) {
            break;
        } else {
            // Parent is 0 indexed so no problem just indexing 
            // at current->parent.
            current = &internal_nodes[current->parent];
        }
    }
}

#if 0
// Phase 1 of the optimization kernel:
// Find the first treelet roots which expand to treelets with at least 7
// leaf nodes. All these potential treelets will be pushed to a global
// buffer which will then be pulled from in the next stage by each warp
// for processing.
// Takes in the node at which this thread will start searching upwards the tree
static __device__ inline void pushPotentialRoots(uint32_t start_search_idx,
                                                 uint32_t total_num_instances,
                                                 uint32_t num_resident_threads)
{
    BVHInternalData *internal_data = bvhParams.internalData;

    struct {
        uint32_t idx;
        uint32_t numInternalNodes;
        uint32_t internalNodesOffset;
        uint32_t numLeaves;
        LBVHNode *leaves;
        LBVHNode *internalNodes;
        TreeletFormationNode *treeletFormNodes;
    } world_info;

    auto update_world_info = [&world_info](uint32_t start_search_idx) {
        world_info.idx = bvhParams.instances[start_search_idx].worldIDX;
        world_info.numLeaves = bvhParams.instanceCounts[world_info.idx];
        world_info.numInternalNodes = world_info.numLeaves - 1;
        world_info.internalNodesOffset = bvhParams.instanceOffsets[world_info.idx];
        world_info.leaves = internal_data->leaves + world_info.internalNodesOffset;
        world_info.internalNodes = internal_data->internalNodes +
            world_info.internalNodesOffset;
        world_info.treeletFormNodes = internal_data->treeletFormNodes +
            world_info.internalNodesOffset;
    };

    update_world_info(start_search_idx);

    // This thread's leaf (tn_offset = thread node offset)
    uint32_t tn_offset = start_search_idx - world_info.internalNodesOffset;
    LBVHNode *current = &world_info.leaves[tn_offset];
    uint32_t num_leaves = 1;

    // Only the threads which survived push their treelets to shared memory
    // (phase 2 shared memory layout).
    sm::OptFastBufferTreelets *smem_p2 = 
        (sm::OptFastBufferTreelets *)sm::buffer;
    sm::Treelet *treelets_buffer = (sm::Treelet *)
        smem_p2->buffer;

    // TODO: Find break condition here
    while (start_search_idx < total_num_instances) {
        int32_t parent = current->parent;

        LBVHNode *parent_node = world_info.internalNodes + parent;
        TreeletFormationNode *parent_form = world_info.treeletFormNodes + parent;

        parent_form->numLeaves.fetch_add_release(num_leaves);

        if (parent_form->numReached.exchange<
                sync::memory_order::relaxed>(1) == 0) {
            // Suspend this thread if this is the first thread to reach this node.
            // However, if this is the first thread to reach this node but this
            // node only has one child, don't suspend.
            if (parent_node->numChildren() == 2) {
                start_search_idx += num_resident_threads;
                update_world_info(start_search_idx);

                tn_offset = start_search_idx - world_info.internalNodesOffset;
                current = &world_info.leaves[tn_offset];
                num_leaves = 1;
                
                continue;
            }
        }

        // When adding the amount of leaves, exclude what was just
        // added by this thread
        num_leaves += (parent_form->numLeaves.load_acquire() - num_leaves);
        current = &world_info.internalNodes[parent];

        if (num_leaves > MADRONA_TREELET_SIZE) {
            // Push a potential treelet!
            sm::InitialTreelet initial_treelet = {
                .rootIndex = tn_offset,
                .worldIndex = world_info.idx,
                .numLeaves = num_leaves
            };

            int32_t treelet_idx =
                (int32_t)smem_p2->treeletCounter.fetch_add_relaxed(1);

            treelets_buffer[treelet_idx].initial = initial_treelet;

            // Start a new search
            start_search_idx += num_resident_threads;
            update_world_info(start_search_idx);

            tn_offset = start_search_idx - world_info.internalNodesOffset;
            current = &world_info.leaves[tn_offset];
            num_leaves = 1;
        }
    }
}

static __device__ inline void formTreelet(uint32_t treelet_idx)
{
    BVHInternalData *internal_data = bvhParams.internalData;

    uint32_t lane_idx = threadIdx.x % MADRONA_WARP_SIZE;

    sm::OptFastBufferTreelets *smem = (sm::OptFastBufferTreelets *)sm::buffer;
    sm::Treelet *treelets = (sm::Treelet *)smem->buffer;

    // Only one thread of the warp actually does the treelet formation
    if (lane_idx == 0) {
        sm::InitialTreelet *initial = &treelets[treelet_idx].initial;

        uint32_t internal_nodes_offset = 
            bvhParams.instanceOffsets[initial->worldIndex];
        LBVHNode *internal_nodes = internal_data->internalNodes +
            internal_nodes_offset;
        LBVHNode *leaf_nodes = internal_data->leaves + internal_nodes_offset;

        sm::FormedTreelet formed_treelet = {
            .rootIndex = initial->rootIndex,
            .worldIndex = initial->worldIndex
        };

        LBVHNode *current_node = &internal_nodes[formed_treelet.rootIndex];
        uint32_t num_leaves = 2;
        formed_treelet.leaves[0] = current_node->left;
        formed_treelet.leaves[1] = current_node->right;
        
        while (num_leaves < MADRONA_TREELET_SIZE) {
            int32_t argmax_sah = -1;
            float max_sah = -FLT_MAX;

            // Loop through the leaves and figure out which has the largest SAH
            // NOTE: you can only replace nodes which are in reality internal
            // nodes. Actual leaves cannot be replaced by their children
            // because they don't have children.
            for (int i = 0; i < num_leaves; ++i) {
                bool is_leaf;
                int32_t child_idx = LBVHNode::storeIdxToChildIdx(
                        formed_treelet.leaves[i], is_leaf);

                if (is_leaf)
                    continue;

                LBVHNode *node = &internal_nodes[child_idx];
                if (node->sah() > max_sah) {
                    argmax_sah = i;
                    max_sah = node->sah();
                }
            }

            // Now, replace the node with maximum sah with its 2 children.
            // This should NEVER fail
            assert(argmax_sah != -1);

            LBVHNode *replaced_node = &internal_nodes[
                formed_treelet.leaves[argmax_sah]];

            // Normally, after each iteration of the while loop, the number of
            // leaves should increase by 1 (until we reach MADRONA_TREELET_SIZE)
            formed_treelet.leaves[argmax_sah] = replaced_node->left;
            formed_treelet.leaves[num_leaves++] = repalced_node->right;
        }

        // Override the bytes in the Treelet struct to reflect the final
        // formed treelet.
        treelets[treelet_idx].formed = formed_treelet;
    }
}

template <typename T, uint32_t N>
struct WarpRegisterFile
{
    static constexpr uint32_t kNumItemsPerLane = 
        (N + MADRONA_WARP_SIZE - 1) / MADRONA_WARP_SIZE;

    T items[kNumItemsPerLane];

    T operator[](uint32_t index)
    {
        const uint32_t lane_id = index / kNumItemsPerLane;
        const uint32_t sub_array_idx = index % kNumItemsPerLane;
        return __shlf_sync(0xFFFF'FFFF, items[sub_array_idx], lane_id);
    }
};
#endif

// Each warp will maintain a single treelet
extern "C" __global__ void bvhOptimizeLBVH()
{
#if 0
    BVHInternalData *internal_data = bvhParams.internalData;

    // Phase 1 shared memory layout
    sm::OptFastBufferTreelets *smem = (sm::OptFastBufferTreelets *)sm::buffer;
    
    const uint32_t threads_per_block = blockDim.x;
    const uint32_t warps_per_block = threads_per_block / MADRONA_WARP_SIZE;
    const uint32_t num_resident_blocks = gridDim.x;
    const uint32_t num_resident_threads = threads_per_block * num_resident_blocks;

    if (threadIdx.x == 0) {
        uint32_t num_instances = bvhParams.instanceOffsets[bvhParams.numWorlds-1] +
                                 bvhParams.instanceCounts[bvhParams.numWorlds-1];
        smem->totalNumInstances = num_instances;
        smem->treeletCounter.store_relaxed(0);
        smem->consumerCounter.store_relaxed(0);
    }

    __syncthreads();

    // For this section, we want all threads who's `thread_inst_offset` is
    // beyond the range of allocated instances to lay dormant and wait
    // until all the treelets have been formed.
    uint32_t thread_inst_offset = blockIdx.x * threads_per_block + threadIdx.x;
    uint32_t total_num_instances = smem->totalNumInstances;

    // Push potential roots to shared memory buffer.
    pushPotentialRoots(thread_inst_offset, 
                       total_num_instances,
                       num_resident_threads);

    __syncthreads();

    // Now, each warp is going to process a single treelet
    uint32_t warp_idx = threadIdx.x / MADRONA_WARP_SIZE;
    uint32_t lane_idx = threadIdx.x % MADRONA_WARP_SIZE;
    uint32_t num_treelets = smem->treeletCounter.load_relaxed();

    for (uint32_t treelet_idx = warp_idx; 
            treelet_idx < num_treelets;
            treelet_idx += warps_per_block) {
        // First, form the treelet of size MADRONA_TREELET_SIZE
        formTreelet(treelet_idx);

        // Make the treelet as optimal as possible


        __syncwarp();
    }

#endif
}

extern "C" __global__ void bvhDebug()
{
}

