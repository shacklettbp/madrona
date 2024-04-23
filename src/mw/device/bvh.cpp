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
#include <madrona/mw_gpu/host_print.hpp>

#define LOG(...) mwGPU::HostPrint::log(__VA_ARGS__)

using namespace madrona;

static constexpr uint32_t kNumBuckets = 12;

struct BinnedSAHJob {
    enum class Direction {
        None, Left, Right
    };

    // Start index of the instances for the current job.
    uint32_t start;

    // End index of the instances for the current job.
    uint32_t end;

    // Parent index
    uint32_t parent;

    // Is this job for the left or right child of the parent.
    Direction currentDir;
};

struct SAHBucket {
    uint32_t count;
    math::AABB bounds;
};

namespace sm {

// We cannot do a true lock-free stack because the items of the stack are
// stored in a contiguous array.
//
// We require the the default constructor of T creates an invalid item.
template <typename T, int MaxN>
struct Stack {
    void init()
    {
        lock_.store(0, std::memory_order_relaxed);
        offset = 0;
    }

    void push(T item)
    {
        while (lock_.exchange(1, std::memory_order_acquire) == 1) {
            while (lock_.load(std::memory_order_relaxed) == 1);
        }

        assert(offset < MaxN);

        items[offset++] = item;

        lock_.store(0, std::memory_order_release);
    }

    T pop()
    {
        while (lock_.exchange(1, std::memory_order_acquire) == 1) {
            while (lock_.load(std::memory_order_relaxed) == 1);
        }

        T item = {};

        if (offset > 0) {
            item = items[--offset];
        }

        lock_.store(0, std::memory_order_release);

        return item;
    }

    [[maybe_unused]] bool tryPop(T *item)
    {
        int32_t is_locked = lock_.load(std::memory_order_relaxed);
        if (is_locked == 1) return false;

        int32_t prev_locked = lock_.exchange(1, std::memory_order_relaxed);
        if (prev_locked) return false;

        if (offset > 0) {
            *item = items[--offset];
        }

        lock_.store(0, std::memory_order_release);

        return true;
    }

private:
    cuda::atomic<int32_t, cuda::thread_scope_block> lock_;
    uint32_t offset;
    T items[MaxN];
};

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

struct BuildSlowBuffer {
    uint32_t worldIdx;
    uint32_t instancesOffset;
    uint32_t numInstances;
    LBVHNode *internalNodesPtr;
    LBVHNode *leafNodesPtr;
    render::InstanceData *instances;
    render::TLBVHNode *aabbs;
    // We use these indirect indices to refer to instances.
    uint32_t *indirectIndices;
    uint32_t *originalIndicesPtr;

    // For each world, we keep looping for work until numJobs hits 0
    cuda::atomic<int32_t, cuda::thread_scope_block> numJobs;
    cuda::atomic<int32_t, cuda::thread_scope_block> leafCounter;
    cuda::atomic<int32_t, cuda::thread_scope_block> internalNodeCounter;
    cuda::atomic<int32_t, cuda::thread_scope_block> processedJobsCounter;

    Stack<BinnedSAHJob, 96> stack;

    char buffer[1];



    math::AABB getAABB(uint32_t idx) {
        return aabbs[indirectIndices[idx]].aabb;
    }
};

struct WidenJob {
    // Index of the LBVH internal node
    uint32_t lbvhNodeIndex;

    // Index of the QBVH internal node
    uint32_t qbvhNodeIndex;
};

struct WidenBuffer {
    uint32_t worldIdx;
    uint32_t instancesOffset;
    uint32_t numInstances;
    LBVHNode *internalNodesPtr;
    LBVHNode *leafNodesPtr;
    render::InstanceData *instances;
    render::TLBVHNode *aabbs;
    QBVHNode *traversalNodes;

    cuda::atomic<int32_t, cuda::thread_scope_block> numJobs;
    cuda::atomic<int32_t, cuda::thread_scope_block> traversalNodesCounter;
    cuda::atomic<int32_t, cuda::thread_scope_block> processedJobsCounter;

    // Stack containing the indices to the internal nodes
    Stack<WidenJob, 64> stack;

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
#if 0
    LOG("\n\n\n\nrender resolution {}\n", bvhParams.renderOutputResolution);
    LOG("pixels are at {}\n", bvhParams.renderOutput);
#endif

    bvhParams.timingInfo->timingCounts.store_relaxed(0);
    bvhParams.timingInfo->tlasTime.store_relaxed(0);
    bvhParams.timingInfo->blasTime.store_relaxed(0);
    bvhParams.timingInfo->numTLASTraces.store_relaxed(0);
    bvhParams.timingInfo->numBLASTraces.store_relaxed(0);

    BVHInternalData *internal_data = bvhParams.internalData;

    // We need to make sure we have enough internal nodes for the initial
    // 2-wide BVH which gets constructed before the optimized tree
    uint32_t num_instances = bvhParams.instanceOffsets[bvhParams.numWorlds-1] +
                             bvhParams.instanceCounts[bvhParams.numWorlds-1];

    // For the 2-wide tree, we need about num_instances internal nodes
    uint32_t num_required_nodes = num_instances;
    uint32_t num_bytes = num_required_nodes * 
        (2 * sizeof(LBVHNode) + 
         sizeof(uint32_t) +
         sizeof(QBVHNode));

    mwGPU::TmpAllocator *allocator = (mwGPU::TmpAllocator *)
        bvhParams.tmpAllocator;

    auto *ptr = allocator->alloc(num_bytes);

    internal_data->internalNodes = (LBVHNode *)ptr;

    internal_data->numAllocatedNodes = num_required_nodes;
    internal_data->buildFastAccumulator.store_relaxed(0);
    internal_data->buildSlowAccumulator.store_relaxed(0);
    internal_data->constructAABBsAccumulator.store_relaxed(0);

    internal_data->leaves = (LBVHNode *)((char *)ptr +
            num_required_nodes * sizeof(LBVHNode));

    internal_data->numAllocatedLeaves = num_required_nodes;
    internal_data->optFastAccumulator.store_relaxed(0);

    internal_data->indirectIndices = (uint32_t *)
        ((char *)ptr + 
         num_required_nodes * (2 * sizeof(LBVHNode)));

    internal_data->traversalNodes = (QBVHNode *)
        ((char *)ptr + num_required_nodes * 
                       (2 * sizeof(LBVHNode) + sizeof(uint32_t)));

#if 0
    internal_data->indirectIndices = (uint32_t *)
        ((char *)ptr + 2 * num_required_nodes * sizeof(LBVHNode)
                     + num_required_nodes * sizeof(TreeletFormationNode)
                     + num_required_nodes * sizeof(uint32_t));
#endif

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

inline __device__ BinnedSAHJob getBinnedSAHJob(sm::BuildSlowBuffer *smem)
{
    BinnedSAHJob current_job = { 0, 0 };
    smem->stack.tryPop(&current_job);
    return current_job;
}

inline __device__ math::AABB getBounds(bool &is_leaf,
                                       sm::BuildSlowBuffer *smem,
                                       const BinnedSAHJob &job,
                                       uint32_t num_instances)
{
    // Only the first thread calculates the bounds
    math::AABB bounds = math::AABB::invalid();
    is_leaf = false;

    bounds = smem->getAABB(job.start);

    for (int i = job.start + 1; i < job.end; ++i) {
        math::AABB current_aabb = smem->getAABB(i);
        bounds = math::AABB::merge(bounds, current_aabb);
    }

    if (num_instances == 1) {
        is_leaf = true;
    }


    return bounds;
}

inline __device__ void updateParent(sm::BuildSlowBuffer *smem,
                                    const BinnedSAHJob &current_job,
                                    bool is_leaf,
                                    int32_t current_node_idx)
{
    if (current_job.parent == 0xFFFF'FFFF) {
        return;
    }

    LBVHNode *parent_node = smem->internalNodesPtr + current_job.parent;
    
    switch (current_job.currentDir) {

    case BinnedSAHJob::Direction::Left: {
        parent_node->left = LBVHNode::childIdxToStoreIdx(
                current_node_idx, is_leaf);
    } break;

    case BinnedSAHJob::Direction::Right: {
        parent_node->right = LBVHNode::childIdxToStoreIdx(
                current_node_idx, is_leaf);
    } break;

    case BinnedSAHJob::Direction::None: {
        // This should never ever happen.
        assert(false);
    } break;

    }
}

inline __device__ void pushLeaf(sm::BuildSlowBuffer *smem,
                                const BinnedSAHJob &current_job,
                                const math::AABB &bounds)
{
    int32_t current_leaf_idx = (int32_t)smem->leafCounter.fetch_add(1,
            std::memory_order_relaxed);

    LBVHNode *leaf_node = smem->leafNodesPtr + current_leaf_idx;

    leaf_node->left = leaf_node->right = 0;

    // Just get only instance index in this span of instances
    leaf_node->instanceIdx = 
        smem->indirectIndices[current_job.start];

    leaf_node->parent = current_job.parent;
    leaf_node->aabb = bounds;

    updateParent(smem, current_job, true, current_leaf_idx);
}

inline __device__ math::AABB getCentroidBounds(sm::BuildSlowBuffer *smem,
                                               const BinnedSAHJob &job)
{
    math::AABB centroid_bounds = math::AABB::invalid();

    centroid_bounds = math::AABB::point(
            smem->getAABB(job.start).centroid());

    for (int i = job.start + 1; i < job.end; ++i) {
        math::Vector3 current_centroid = smem->getAABB(i).centroid();
        centroid_bounds.expand(current_centroid);
    }

    return centroid_bounds;
}

inline __device__ uint32_t pushInternalNode(sm::BuildSlowBuffer *smem,
                                            const BinnedSAHJob &current_job,
                                            const math::AABB &bounds)
{
    int32_t current_node_idx = (int32_t)
        smem->internalNodeCounter.fetch_add(1, std::memory_order_relaxed);

    LBVHNode *node = smem->internalNodesPtr + current_node_idx;

    // For now, just initialize to 0 (this will be updated by later recursive
    // calls.
    node->left = node->right = 0;
    node->instanceIdx = 0xFFFF'FFFF;
    node->parent = current_job.parent;
    node->aabb = bounds;
    
    updateParent(smem, current_job, false, current_node_idx);

    return (uint32_t)current_node_idx;
}

inline __device__ void updateJobCount(sm::BuildSlowBuffer *smem,
                                      int32_t job_count_diff)
{
    int32_t new_job_count = 
        smem->numJobs.fetch_add(job_count_diff, std::memory_order_relaxed);
}

inline __device__ int32_t pushJobs(sm::BuildSlowBuffer *smem,
                                   const BinnedSAHJob &current_job,
                                   uint32_t mid,
                                   uint32_t current_node_idx)
{
    int32_t job_count_diff = -1;

    // Push the left job if there is one to push
    if (mid - current_job.start > 0) {
        BinnedSAHJob job = {
            .start = current_job.start,
            .end = mid,
            .parent = current_node_idx,
            .currentDir = BinnedSAHJob::Direction::Left,
        };

        smem->stack.push(job);

        ++job_count_diff;
    }

    // Push the right job if there is one to push
    if (current_job.end - mid > 0) {
        BinnedSAHJob job = {
            .start = mid,
            .end = current_job.end,
            .parent = current_node_idx,
            .currentDir = BinnedSAHJob::Direction::Right,
        };

        smem->stack.push(job);

        ++job_count_diff;
    }

    return job_count_diff;
}

inline __device__ int getBucket(sm::BuildSlowBuffer *smem,
                                uint32_t prim_idx,
                                const math::AABB &centroid_bounds,
                                int max_dim)
{
    math::AABB aabb = smem->getAABB(prim_idx);
    int b = kNumBuckets * centroid_bounds.offset(aabb.centroid())[max_dim];

    if (b >= kNumBuckets) {
        b = kNumBuckets - 1;
    }

    return b;
}

inline __device__ uint32_t partitionIndices(
        uint32_t start, uint32_t end,
        SAHBucket *buckets,
        sm::BuildSlowBuffer *smem,
        uint32_t min_cost_split_bucket,
        const math::AABB &centroid_bounds,
        int max_dim)
{
    uint32_t first_idx = [&]() {
        for (uint32_t i = start; i < end; ++i) {
            if (getBucket(smem, i, centroid_bounds, max_dim) >
                min_cost_split_bucket) {
                return i;
            }
        }

        return end;
    }();

    // Every thing is before the split
    if (first_idx == end) {
        return first_idx;
    }

    for (int i = first_idx + 1; i < end; ++i) {
        if (getBucket(smem, i, centroid_bounds, max_dim) <=
                min_cost_split_bucket) {
            std::swap(smem->indirectIndices[first_idx],
                      smem->indirectIndices[i]);
            ++first_idx;
        }
    }

    return first_idx;
}

// This builds a high quality tree but slowly
extern "C" __global__ void bvhBuildSlow()
{
    BVHInternalData *internal_data = bvhParams.internalData;
    sm::BuildSlowBuffer *smem = (sm::BuildSlowBuffer *)sm::buffer;

    uint32_t current_world_idx = blockIdx.x;
    uint32_t lane_idx = threadIdx.x % MADRONA_WARP_SIZE;

    while (current_world_idx < bvhParams.numWorlds) {
        // Get the instance offsets and stuff.
        __syncthreads();

        if (threadIdx.x == 0) {
            smem->worldIdx = current_world_idx;
            smem->instancesOffset = bvhParams.instanceOffsets[current_world_idx];
            smem->numInstances = bvhParams.instanceCounts[current_world_idx];
            smem->internalNodesPtr = internal_data->internalNodes +
                                     smem->instancesOffset;
            smem->leafNodesPtr = internal_data->leaves +
                                 smem->instancesOffset;
            smem->instances = bvhParams.instances + smem->instancesOffset;
            smem->aabbs = bvhParams.aabbs + smem->instancesOffset;
            smem->stack.init();
            smem->stack.push({ 0, smem->numInstances, 0xFFFF'FFFF, 
                               BinnedSAHJob::Direction::None });
            smem->numJobs.store(1, std::memory_order_relaxed);
            smem->indirectIndices = internal_data->indirectIndices + 
                smem->instancesOffset;
            smem->originalIndicesPtr = smem->indirectIndices;
            smem->processedJobsCounter.store(0, std::memory_order_relaxed);
            smem->leafCounter.store(0, std::memory_order_relaxed);
            smem->internalNodeCounter.store(0, std::memory_order_relaxed);
            smem->processedJobsCounter.store(0, std::memory_order_relaxed);
        }

        __syncthreads();

        // Set the indirect indices to just a simple iota
        for (int i = threadIdx.x; i < smem->numInstances; i += blockDim.x) {
            smem->indirectIndices[i] = i;
        }

        __syncthreads();

        if (lane_idx == 0) {
            while (smem->numJobs.load(std::memory_order_relaxed) > 0) {
                // Try to pop a job from the stack.
                BinnedSAHJob current_job = getBinnedSAHJob(smem);

                // If the range of the instances for this job is 0, that means
                // that we didn't manage to get a job.
                uint32_t num_instances = current_job.end - current_job.start;
                if (num_instances == 0) {
                    // For whatever reason, we didn't get a job (either lock failed,
                    // or there weren't any jobs in the stack).
                    continue;
                }

                int32_t processed_jobs_count = 
                    smem->processedJobsCounter.fetch_add(
                            1, std::memory_order_relaxed);

                // Only the first thread calculates the bounds (of the AABBs)
                bool is_leaf = false;
                math::AABB bounds = getBounds(is_leaf, smem, 
                        current_job, num_instances);

                // If we are a leaf, push leaf and just continue.
                if (is_leaf) {
                    // Push this as a leaf.
                    pushLeaf(smem, current_job, bounds);

                    // No jobs were pushed, this is just removing a job
                    updateJobCount(smem, -1);

                    continue;
                } 

                // If we hit this point, we aren't at a leaf.
                math::AABB centroid_bounds = getCentroidBounds(smem, current_job);

                int max_dim = centroid_bounds.maxDimension();

                // The case where num_instances is 1 is already handled
                if (num_instances == 2) {
                    int mid = current_job.start + 1;
                    if (smem->getAABB(current_job.start).centroid()[max_dim] >
                            smem->getAABB(current_job.start+1).centroid()[max_dim]) {
                        std::swap(smem->indirectIndices[current_job.start],
                                smem->indirectIndices[current_job.start+1]);
                    }

                    // Push the new internal node
                    uint32_t current_node_idx = pushInternalNode(
                            smem, current_job, bounds);

                    int32_t job_count_diff = pushJobs(
                            smem, current_job, mid, current_node_idx);

                    updateJobCount(smem, job_count_diff);

                    continue;
                }

                // Now, handle the case where there are more than 2 instances.
                // i.e., do the binned SAH method.
                // Each thread in the warp has a bucket
                {
                    SAHBucket buckets[kNumBuckets];

                    // Initialize the buckets
                    for (uint32_t i = 0; i < kNumBuckets; ++i) {
                        buckets[i].count = 0;
                        buckets[i].bounds = math::AABB::invalid();
                    }

                    // Create the bounds
                    for (uint32_t prim_idx = current_job.start;
                            prim_idx < current_job.end;
                            prim_idx++) {
                        math::AABB aabb = smem->getAABB(prim_idx);
                        int b = kNumBuckets * centroid_bounds.offset(
                                aabb.centroid())[max_dim];

                        if (b >= kNumBuckets) {
                            b = kNumBuckets - 1;
                        }

                        buckets[b].count++;
                        buckets[b].bounds = math::AABB::merge(buckets[b].bounds, 
                                aabb);
                    }

                    static constexpr uint32_t kNumSplits = kNumBuckets-1;
                    float costs[kNumSplits] = {};

                    int count_below = 0;
                    math::AABB bound_below = math::AABB::invalid();

                    for (int i = 0; i < kNumSplits; ++i) {
                        bound_below = math::AABB::merge(bound_below, buckets[i].bounds);
                        count_below += buckets[i].count;
                        costs[i] += count_below * bound_below.surfaceArea();
                    }

                    int count_above = 0;
                    math::AABB bound_above = math::AABB::invalid();

                    for (int i = kNumSplits; i >= 1; --i) {
                        bound_above = math::AABB::merge(bound_above, buckets[i].bounds);
                        count_above += buckets[i].count;
                        costs[i - 1] += count_above * bound_above.surfaceArea();
                    }

                    int min_cost_split_bucket = -1;
                    float min_cost = FLT_MAX;

                    for (int i = 0; i < kNumSplits; ++i) {
                        if (costs[i] < min_cost) {
                            min_cost = costs[i];
                            min_cost_split_bucket = i;
                        }
                    }

                    min_cost = 1.f / 2.f + min_cost / bounds.surfaceArea();
                    uint32_t mid_idx = 0;

                    { // Partition the array in-place
                        mid_idx = partitionIndices(
                                current_job.start, current_job.end,
                                buckets,
                                smem,
                                min_cost_split_bucket,
                                centroid_bounds,
                                max_dim);

                        for (int i = current_job.start; i < mid_idx; ++i) {
                            assert(getBucket(smem, i, centroid_bounds, max_dim) <= 
                                    min_cost_split_bucket);
                        }
                    }

                    uint32_t current_node_idx = pushInternalNode(
                            smem, current_job, bounds);

                    int32_t job_count_diff = pushJobs(
                            smem, current_job, mid_idx, current_node_idx);

                    updateJobCount(smem, job_count_diff);
                }

                __syncwarp();
            }
        }

        __syncwarp();


        // Increment the world that this thread block is working on
        current_world_idx += gridDim.x;
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
        nodes[tn_offset].instanceIdx = 0xFFFF'FFFF;

        leaves[split_index].parent = tn_offset;
        leaves[split_index].instanceIdx = split_index;

        uint32_t global_instance_idx = (split_index +
            world_info.internalNodesOffset);

        leaves[split_index].aabb = bvhParams.aabbs[global_instance_idx].aabb;
        leaves[split_index].reachedCount.store_relaxed(0);
    } else {
        // The left node is an internal node and its index is split_index
        nodes[tn_offset].left = LBVHNode::childIdxToStoreIdx(split_index, false);
        nodes[tn_offset].reachedCount.store_relaxed(0);
        nodes[tn_offset].instanceIdx = 0xFFFF'FFFF;
        nodes[split_index].parent = tn_offset;
    }
    
    if (right_index == split_index + 1) {
        // The right node is a leaf and the leaf's index is split_index+1
        nodes[tn_offset].right = LBVHNode::childIdxToStoreIdx(split_index + 1, true);
        nodes[tn_offset].reachedCount.store_relaxed(0);
        nodes[tn_offset].instanceIdx = 0xFFFF'FFFF;

        leaves[split_index+1].parent = tn_offset;
        leaves[split_index+1].instanceIdx = split_index+1;

        uint32_t global_instance_idx = (split_index + 1 +
            world_info.internalNodesOffset);

        leaves[split_index+1].aabb = bvhParams.aabbs[global_instance_idx].aabb;
        leaves[split_index+1].reachedCount.store_relaxed(0);
    } else {
        // The right node is an internal node and its index is split_index+1
        nodes[tn_offset].right = LBVHNode::childIdxToStoreIdx(split_index + 1, false);
        nodes[tn_offset].reachedCount.store_relaxed(0);
        nodes[tn_offset].instanceIdx = 0xFFFF'FFFF;
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

inline __device__ sm::WidenJob getWidenJob(sm::WidenBuffer *smem)
{
    sm::WidenJob stored_job = {};
    smem->stack.tryPop(&stored_job);
    return stored_job;
}

// For now, we're just doing a very naive widening of the initial binary tree.
extern "C" __global__ void bvhWidenTree()
{
    BVHInternalData *internal_data = bvhParams.internalData;
    sm::WidenBuffer *smem = (sm::WidenBuffer *)sm::buffer;

    uint32_t current_world_idx = blockIdx.x;
    uint32_t lane_idx = threadIdx.x % MADRONA_WARP_SIZE;
    uint32_t warp_idx = threadIdx.x / MADRONA_WARP_SIZE;

    while (current_world_idx < bvhParams.numWorlds) {
        __syncthreads();

        if (threadIdx.x == 0) {
            smem->worldIdx = current_world_idx;
            smem->instancesOffset = bvhParams.instanceOffsets[current_world_idx];
            smem->numInstances = bvhParams.instanceCounts[current_world_idx];
            smem->internalNodesPtr = internal_data->internalNodes +
                                     smem->instancesOffset;
            smem->leafNodesPtr = internal_data->leaves +
                                 smem->instancesOffset;
            smem->traversalNodes = internal_data->traversalNodes +
                                 smem->instancesOffset;

            smem->instances = bvhParams.instances + smem->instancesOffset;
            smem->aabbs = bvhParams.aabbs + smem->instancesOffset;

            smem->stack.init();
            // Push the root node (indices start at 1)
            smem->stack.push({ 1, 1 });

            smem->numJobs.store(1, std::memory_order_relaxed);
            smem->processedJobsCounter.store(0, std::memory_order_relaxed);
            smem->traversalNodesCounter.store(1, std::memory_order_relaxed);
        }

        __syncthreads();

        if (lane_idx == 0) {
            while (smem->numJobs.load(std::memory_order_relaxed) > 0) {
                auto stored_job = getWidenJob(smem);

                if (stored_job.lbvhNodeIndex == 0) {
                    continue;
                }

#if 0
                LOG("Got job lbvhNodeIndex = {}; qbvhNodeIndex = {}\n",
                    stored_job.lbvhNodeIndex, stored_job.qbvhNodeIndex);
#endif

                int32_t processed_jobs_count = 
                    smem->processedJobsCounter.fetch_add(
                        1, std::memory_order_relaxed);

                int32_t lbvh_node_idx = stored_job.lbvhNodeIndex - 1;

                LBVHNode *current_node = &smem->internalNodesPtr[lbvh_node_idx];

                uint32_t num_children = 0;
                QBVHNode::NodeIndexT children_indices[QBVHNode::NodeWidth];
                math::AABB children_aabbs[QBVHNode::NodeWidth];

                // Push the children
                if (current_node->left < 0) {
                    children_indices[num_children++] = current_node->left;
                } else {
                    LBVHNode *child = 
                        &smem->internalNodesPtr[current_node->left - 1];

                    children_indices[num_children++] = child->left;
                    children_indices[num_children++] = child->right;
                }

                if (current_node->right < 0) {
                    children_indices[num_children++] = current_node->right;
                } else {
                    LBVHNode *child = 
                        &smem->internalNodesPtr[current_node->right - 1];

                    children_indices[num_children++] = child->left;
                    children_indices[num_children++] = child->right;
                }

                for (int i = 0; i < num_children; ++i) {
                    if (children_indices[i] < 0) {
                        LBVHNode *leaf_node = 
                            &smem->leafNodesPtr[-children_indices[i] - 1];

                        // Store the instance index as negative
                        children_indices[i] =
                            -(QBVHNode::NodeIndexT)leaf_node->instanceIdx - 1;
                        children_aabbs[i] = leaf_node->aabb;
                    } else {
                        // Store the indices and aabb, but also start new jobs
                        LBVHNode *inode =
                            &smem->internalNodesPtr[children_indices[i] - 1];

                        // Create a new QBVHNode
                        uint32_t qbvh_node_idx =
                            smem->traversalNodesCounter.fetch_add(1,
                                    std::memory_order_relaxed);

                        sm::WidenJob new_job = {
                            .lbvhNodeIndex = (uint32_t)children_indices[i],
                            .qbvhNodeIndex = qbvh_node_idx + 1
                        };

                        smem->stack.push(new_job);
                        smem->numJobs.fetch_add(1, std::memory_order_relaxed);

                        children_indices[i] = 
                            (QBVHNode::NodeIndexT)qbvh_node_idx + 1;
                        children_aabbs[i] = inode->aabb;
                    }
                }

                QBVHNode *current_qbvh_node =
                    &smem->traversalNodes[stored_job.qbvhNodeIndex - 1];

                *current_qbvh_node = QBVHNode::construct(
                        num_children,
                        children_aabbs,
                        children_indices);

                smem->numJobs.fetch_add(-1, std::memory_order_relaxed);
            }
        }

        __syncwarp();

        current_world_idx += gridDim.x;
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

