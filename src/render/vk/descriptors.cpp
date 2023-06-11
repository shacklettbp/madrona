#include "descriptors.hpp"

#include <cassert>
#include <iostream>

using namespace std;

namespace madrona {
namespace render {
namespace vk {

DescriptorManager::DescriptorManager(const Device &d,
                                     const PipelineShaders &shader,
                                     uint32_t set_id)
    : dev(d),
      shader_(shader),
      set_id_(set_id),
      layout_(shader_.getLayout(set_id)),
      free_pools_(),
      used_pools_()
{}

DescriptorManager::~DescriptorManager()
{
    for (PoolState &pool_state : free_pools_) {
        dev.dt.destroyDescriptorPool(dev.hdl, pool_state.pool, nullptr);
        assert(pool_state.numActive.load_relaxed() == 0);
    }

    for (PoolState &pool_state : used_pools_) {
        dev.dt.destroyDescriptorPool(dev.hdl, pool_state.pool, nullptr);
        assert(pool_state.numActive.load_relaxed() == 0);
        assert(pool_state.numUsed == VulkanConfig::descriptor_pool_size);
    }
}

DescriptorSet DescriptorManager::makeSet()
{
    if (layout_ == VK_NULL_HANDLE) {
        return DescriptorSet(VK_NULL_HANDLE, nullptr);
    }

    if (free_pools_.empty()) {
        auto iter = used_pools_.begin();
        while (iter != used_pools_.end()) {
            auto next_iter = next(iter);
            if (iter->numActive.load_relaxed() == 0) {
                iter->numUsed = 0;
                REQ_VK(dev.dt.resetDescriptorPool(dev.hdl, iter->pool, 0));
                free_pools_.splice(free_pools_.end(), used_pools_, iter);
            }
            iter = next_iter;
        }
        if (free_pools_.empty()) {
            free_pools_.emplace_back(
                shader_.makePool(set_id_, VulkanConfig::descriptor_pool_size));
        }
    }

    PoolState &cur_pool = free_pools_.front();

    VkDescriptorSet desc_set = makeDescriptorSet(dev, cur_pool.pool, layout_);

    cur_pool.numUsed++;
    cur_pool.numActive.fetch_add_relaxed(1);

    if (cur_pool.numUsed == VulkanConfig::descriptor_pool_size) {
        used_pools_.splice(used_pools_.end(), free_pools_,
                           free_pools_.begin());
    }

    return DescriptorSet(desc_set, &cur_pool);
}

FixedDescriptorPool::FixedDescriptorPool(const Device &d,
                                         const PipelineShaders &shader,
                                         uint32_t set_id,
                                         uint32_t pool_size)
    : dev(d),
      layout_(shader.getLayout(set_id)),
      pool_(shader.makePool(set_id, pool_size))
{}

FixedDescriptorPool::FixedDescriptorPool(FixedDescriptorPool &&o)
    : dev(o.dev),
      layout_(o.layout_),
      pool_(o.pool_)
{
    o.pool_ = VK_NULL_HANDLE;
}

FixedDescriptorPool::~FixedDescriptorPool()
{
    if (pool_ == VK_NULL_HANDLE) return;
    dev.dt.destroyDescriptorPool(dev.hdl, pool_, nullptr);
}

VkDescriptorSet FixedDescriptorPool::makeSet()
{
    return makeDescriptorSet(dev, pool_, layout_);
}

}
}
}
