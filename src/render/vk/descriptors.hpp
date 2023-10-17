#pragma once

#include <atomic>
#include <list>

#include <madrona/render/vk/device.hpp>

#include "config.hpp"
#include "utils.hpp"
#include "pipeline_shaders.hpp"

namespace madrona {
namespace render {
namespace vk {

struct PoolState {
    PoolState(VkDescriptorPool p) : pool(p), numUsed(0), numActive(0) {}

    VkDescriptorPool pool;
    uint32_t numUsed;
    AtomicU32 numActive;
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
        pool->numActive.fetch_sub_relaxed(1);
    };

    VkDescriptorSet hdl;
    PoolState *pool;
};

class DescriptorManager {
public:
    DescriptorManager(const Device &dev,
                      const PipelineShaders &shader,
                      uint32_t set_id);

    DescriptorManager(const DescriptorManager &) = delete;
    DescriptorManager(DescriptorManager &&) = default;

    ~DescriptorManager();

    DescriptorSet makeSet();

private:
    const Device &dev;
    const PipelineShaders &shader_;
    uint32_t set_id_;
    VkDescriptorSetLayout layout_;

    std::list<PoolState> free_pools_;
    std::list<PoolState> used_pools_;
};

class FixedDescriptorPool {
public:
    FixedDescriptorPool(const Device &dev,
                        const PipelineShaders &shader,
                        uint32_t set_id,
                        uint32_t pool_size);

    FixedDescriptorPool(const FixedDescriptorPool &) = delete;
    FixedDescriptorPool(FixedDescriptorPool &&);

    ~FixedDescriptorPool();

    VkDescriptorSet makeSet();

private:
    const Device &dev;
    VkDescriptorSetLayout layout_;
    VkDescriptorPool pool_;
};

class DescHelper {
public:
    static inline void textures(VkWriteDescriptorSet &update,
                                VkDescriptorSet desc_set,
                                VkDescriptorImageInfo *views,
                                uint32_t num_textures,
                                uint32_t binding,
                                uint32_t arr_offset = 0);

    static inline void buffer(VkWriteDescriptorSet &update,
                              VkDescriptorSet desc_set,
                              const VkDescriptorBufferInfo *buf,
                              uint32_t binding,
                              VkDescriptorType type);

    static inline void uniform(VkWriteDescriptorSet &update,
                               VkDescriptorSet desc_set,
                               const VkDescriptorBufferInfo *buf,
                               uint32_t binding);

    static inline void storage(VkWriteDescriptorSet &update,
                               VkDescriptorSet desc_set,
                               const VkDescriptorBufferInfo *buf,
                               uint32_t binding);

    static inline void storageImage(VkWriteDescriptorSet &update,
                                    VkDescriptorSet desc_set,
                                    const VkDescriptorImageInfo *img,
                                    uint32_t binding,
                                    uint32_t arr_offset = 0);

    static inline void accelStructs(
        VkWriteDescriptorSet &update,
        VkWriteDescriptorSetAccelerationStructureKHR &as_update,
        VkDescriptorSet desc_set,
        const VkAccelerationStructureKHR *accel_structs,
        uint32_t num_accel_structs,
        uint32_t binding,
        uint32_t arr_offset = 0);

    static inline void update(const Device &dev,
                              const VkWriteDescriptorSet *updates,
                              uint32_t num_desc_updates);
};

}
}
}

#include "descriptors.inl"
