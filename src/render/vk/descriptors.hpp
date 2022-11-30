#pragma once

#include <atomic>
#include <list>

#include "config.hpp"
#include "core.hpp"
#include "utils.hpp"
#include "shader.hpp"

namespace madrona {
namespace render {
namespace vk {

struct PoolState {
    PoolState(VkDescriptorPool p) : pool(p), numUsed(0), numActive(0) {}

    VkDescriptorPool pool;
    uint32_t numUsed;
    std::atomic_uint32_t numActive;
};

struct DescriptorSet {
    DescriptorSet(VkDescriptorSet d, PoolState *p) : hdl(d), pool(p) {}

    DescriptorSet(const DescriptorSet &) = delete;

    DescriptorSet(DescriptorSet &&o) : hdl(o.hdl), pool(o.pool)
    {
        o.hdl = VK_NULL_HANDLE;
    }

    ~DescriptorSet()
    {
        if (hdl == VK_NULL_HANDLE) return;
        pool->numActive--;
    };

    VkDescriptorSet hdl;
    PoolState *pool;
};

class DescriptorManager {
public:
    DescriptorManager(const DeviceState &dev,
                      const PipelineShaders &shader,
                      uint32_t set_id);

    DescriptorManager(const DescriptorManager &) = delete;
    DescriptorManager(DescriptorManager &&) = default;

    ~DescriptorManager();

    DescriptorSet makeSet();

private:
    const DeviceState &dev;
    const PipelineShaders &shader_;
    uint32_t set_id_;
    VkDescriptorSetLayout layout_;

    std::list<PoolState> free_pools_;
    std::list<PoolState> used_pools_;
};

class FixedDescriptorPool {
public:
    FixedDescriptorPool(const DeviceState &dev,
                        const PipelineShaders &shader,
                        uint32_t set_id,
                        uint32_t pool_size);

    FixedDescriptorPool(const FixedDescriptorPool &) = delete;
    FixedDescriptorPool(FixedDescriptorPool &&);

    ~FixedDescriptorPool();

    VkDescriptorSet makeSet();

private:
    const DeviceState &dev;
    VkDescriptorSetLayout layout_;
    VkDescriptorPool pool_;
};

class DescriptorUpdates {
public:
    inline DescriptorUpdates(uint32_t max_updates);

    inline void textures(VkDescriptorSet desc_set,
                         VkDescriptorImageInfo *views,
                         uint32_t num_textures,
                         uint32_t binding,
                         uint32_t arr_elem = 0);

    inline void buffer(VkDescriptorSet desc_set,
                       const VkDescriptorBufferInfo *buf,
                       uint32_t binding,
                       VkDescriptorType type);

    inline void uniform(VkDescriptorSet desc_set,
                        const VkDescriptorBufferInfo *buf,
                        uint32_t binding);

    inline void storage(VkDescriptorSet desc_set,
                        const VkDescriptorBufferInfo *buf,
                        uint32_t binding);

    inline void update(const DeviceState &dev);

private:
    std::vector<VkWriteDescriptorSet> updates_;
};

}
}
}

#include "descriptors.inl"
