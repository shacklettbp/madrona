#include <madrona/memory.hpp>
#include <madrona/table.hpp>
#include <madrona/utils.hpp>

#include <madrona/mw_gpu/host_print.hpp>

#if defined (MADRONA_MWGPU_BVH_MODULE)
#define MADRONA_MWGPU_MAX_BLOCKS_PER_SM 4
#include <madrona/bvh.hpp>
#endif

#if defined (MADRONA_MWGPU_BVH_MODULE)
extern "C" __constant__ madrona::BVHParams bvhParams;
#endif

namespace madrona {

namespace mwGPU {

HostAllocator::HostAllocator(HostAllocInit init)
    : channel_(init.channel),
      device_lock_(),
      host_page_size_(init.pageSize),
      alloc_granularity_(init.allocGranularity)
{}

static void submitRequest(HostChannel *channel)
{
    using namespace cuda::std;

    channel->ready.store(1, memory_order_release);

    while (channel->finished.load(memory_order_acquire) != 1) {}

    channel->finished.store(0, memory_order_relaxed);
}

void * HostAllocator::reserveMemory(uint64_t max_bytes,
                                    uint64_t init_num_bytes)
{
    device_lock_.lock();

    channel_->op = HostChannel::Op::Reserve;
    channel_->reserve.maxBytes = max_bytes;
    channel_->reserve.initNumBytes = init_num_bytes;

    submitRequest(channel_);

    void *result = channel_->reserve.result;

    device_lock_.unlock();

    return result;
}

void * HostAllocator::allocMemory(uint64_t num_bytes)
{
    device_lock_.lock();

    channel_->op = HostChannel::Op::Alloc;
    channel_->alloc.numBytes = num_bytes;

    submitRequest(channel_);

    void *result = channel_->alloc.result;

    device_lock_.unlock();

    return result;
}

void HostAllocator::mapMemory(void *addr, uint64_t num_bytes)
{
    device_lock_.lock();

    channel_->op = HostChannel::Op::Map;
    channel_->map.addr = addr;
    channel_->map.numBytes = num_bytes;

    submitRequest(channel_);

    device_lock_.unlock();
}

void HostAllocator::reserveFree(void *addr, uint64_t num_bytes,
                                uint64_t num_reserve_bytes)
{
    device_lock_.lock();

    channel_->op = HostChannel::Op::ReserveFree;
    channel_->reserveFree.addr = addr;
    channel_->reserveFree.numBytes = num_bytes;
    channel_->reserveFree.numReserveBytes = num_reserve_bytes;

    submitRequest(channel_);

    device_lock_.unlock();
}

void HostAllocator::allocFree(void *addr)
{
    device_lock_.lock();

    channel_->op = HostChannel::Op::AllocFree;
    channel_->allocFree.addr = addr;

    submitRequest(channel_);

    device_lock_.unlock();
}

uint64_t HostAllocator::roundUpReservation(uint64_t num_bytes)
{
    return utils::roundUp(num_bytes, host_page_size_);
}

uint64_t HostAllocator::roundUpAlloc(uint64_t num_bytes)
{
    return utils::roundUp(num_bytes, alloc_granularity_);
}

namespace SharedMemStorage {
__shared__ Chunk buffer[numSMemBytes / sizeof(Chunk)];
}

TmpAllocator::TmpAllocator()
    : base_(mwGPU::getHostAllocator()->reserveMemory(Table::maxReservedBytesPerTable, 0)),
      offset_(0),
      num_mapped_bytes_(0),
      grow_lock_()
{}

void * TmpAllocator::alloc(uint64_t num_bytes)
{
    num_bytes = utils::roundUpPow2(num_bytes, 256);
    uint64_t alloc_offset = offset_.fetch_add_relaxed(num_bytes);

    uint64_t required_bytes = alloc_offset + num_bytes;

    if (required_bytes > num_mapped_bytes_) {
        grow_lock_.lock();

        uint64_t cur_mapped_bytes = num_mapped_bytes_;
        if (required_bytes > cur_mapped_bytes) {

#if defined (MADRONA_MWGPU_BVH_MODULE)
            HostAllocator *host_alloc = (HostAllocator *)bvhParams.hostAllocator;
#else
            auto *host_alloc = mwGPU::getHostAllocator();
#endif

            uint64_t min_grow = required_bytes - cur_mapped_bytes;

            // Double by default
            uint64_t num_added_bytes = 
                max(max(cur_mapped_bytes, min_grow), uint64_t(1024 * 1024));

            constexpr uint64_t max_normal_grow = 256 * 1024 * 1024;
            if (min_grow < max_normal_grow) {
                num_added_bytes = min(num_added_bytes, max_normal_grow);
            }

            num_added_bytes = host_alloc->roundUpAlloc(num_added_bytes);
            
            host_alloc->mapMemory(
                (char *)base_ + cur_mapped_bytes, num_added_bytes);

            num_mapped_bytes_ = cur_mapped_bytes + num_added_bytes;
        }

        grow_lock_.unlock();
    }

    return (char *)base_ + alloc_offset;
}

}
}
