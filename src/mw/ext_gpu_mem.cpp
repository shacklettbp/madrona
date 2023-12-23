#include <cassert>
#include <cstdlib>
#include <madrona/mw_ext_gpu_mem.hpp>

namespace madrona {

namespace {
constexpr uint32_t maxVMAllocators = 64;
constexpr uint32_t mappingNodesDefaultSize = 64;
constexpr uint32_t invalidNodeIndex = 0xFFFF'FFFF;
}

struct GPUMappingNode {
    GPUMapping data;

    // Depending on whether this is a free node or not, this will hold either
    // the index to the next free node, or to the next node in the queue.
    uint32_t nextIdx;
};

using InstanceQueue = std::pair<uint32_t, uint32_t>;

struct GPUExternalVM::Impl {
    uint32_t allocatorCounter;

    uint32_t bufferSize;
    uint32_t reachedNodes;
    uint32_t firstFreeNodeIdx;
    GPUMappingNode *mappingNodesBuffer;

    // Each one of these pairs contain the tail and head of the queue.
    InstanceQueue instances[maxVMAllocators];

    Impl();
    ~Impl();
};

GPUExternalVM::Impl::Impl()
    : allocatorCounter(0),
      bufferSize(mappingNodesDefaultSize),
      reachedNodes(0),
      firstFreeNodeIdx(invalidNodeIndex),
      mappingNodesBuffer((GPUMappingNode *)malloc(
                         sizeof(GPUMappingNode) * bufferSize))
{
}

GPUExternalVM::Impl::~Impl()
{
    free(mappingNodesBuffer);
}

GPUExternalVMInstance GPUExternalVM::registerInstance()
{
    assert(impl_->allocatorCounter < maxVMAllocators);
    uint32_t inst = impl_->allocatorCounter++;

    impl_->instances[inst].first = invalidNodeIndex;
    impl_->instances[inst].second = invalidNodeIndex;

    return inst;
}

GPUMapping GPUExternalVM::dequeueMapping(GPUExternalVMInstance inst) const
{
    InstanceQueue *q = &impl_->instances[inst];

    if (q->first == invalidNodeIndex && q->second == invalidNodeIndex) {
        return { nullptr, 0 };
    } else {
        uint32_t old_tail_idx = q->first;
        GPUMappingNode *old_tail = &impl_->mappingNodesBuffer[old_tail_idx];

        // Update the tail
        q->first = old_tail->nextIdx;

        // Once we've reached the end, uninitialize the queue
        if (q->first == invalidNodeIndex) {
            q->first = q->second = invalidNodeIndex;
        }

        // Add old_tail to the list of free nodes
        old_tail->data = { nullptr, 0 };
        old_tail->nextIdx = impl_->firstFreeNodeIdx;

        impl_->firstFreeNodeIdx = old_tail_idx;

        return old_tail->data;
    }
}

void GPUExternalVM::queueMapping(GPUExternalVMInstance inst,
                                 const GPUMapping &mapping)
{
    // Find a free GPU mapping node
    uint32_t freeNodeIdx = invalidNodeIndex;
    if (impl_->firstFreeNodeIdx != invalidNodeIndex) {
        freeNodeIdx = impl_->firstFreeNodeIdx;

        // Update the first free node index in the free list.
        GPUMappingNode *free_node = &impl_->mappingNodesBuffer[freeNodeIdx];
        impl_->firstFreeNodeIdx = free_node->nextIdx;
    } else if (impl_->reachedNodes < impl_->bufferSize) {
        // We can extend the reached nodes to peel off the next node
        freeNodeIdx = impl_->reachedNodes++;
    } else {
        // All the nodes are full, we must do a realloc of the nodes buffer.
        impl_->bufferSize *= 2;

        impl_->mappingNodesBuffer = (GPUMappingNode *)realloc(
            impl_->mappingNodesBuffer, sizeof(GPUMappingNode) * impl_->bufferSize);

        assert(impl_->mappingNodesBuffer);

        freeNodeIdx = impl_->reachedNodes++;
    }

    GPUMappingNode *node = &impl_->mappingNodesBuffer[freeNodeIdx];
    node->data = mapping;
    node->nextIdx = invalidNodeIndex;

    InstanceQueue *q = &impl_->instances[inst];
    if (q->second == invalidNodeIndex) {
        // This queue is uninitialized
        q->first = q->second = freeNodeIdx;
    } else {
        // Get the head
        GPUMappingNode *old_head = &impl_->mappingNodesBuffer[q->second];
        old_head->nextIdx = freeNodeIdx;

        q->second = freeNodeIdx;
    }
}

GPUExternalVM::GPUExternalVM()
    : impl_(std::make_unique<Impl>())
{
}

GPUExternalVM::~GPUExternalVM()
{
}
    
}
