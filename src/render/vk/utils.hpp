#pragma once

#include <deque>
#include <mutex>
#include <string>

#include <madrona/crash.hpp>

#include <madrona/render/vk/device.hpp>
#include "pipeline_shaders.hpp"

namespace madrona::render::vk {

class QueueState {
public:
    inline QueueState(VkQueue queue_hdl, bool shared);

    inline void submit(const Device &dev,
                       uint32_t submit_count,
                       const VkSubmitInfo *pSubmits,
                       VkFence fence) const;

    inline void bindSubmit(const Device &dev,
                           uint32_t submit_count,
                           const VkBindSparseInfo *pSubmits,
                           VkFence fence) const;

    // Returns true if successful. Returns false if failed (for instance,
    // in case of needing swapchain resizing.
    inline bool presentSubmit(const Device &dev,
                              const VkPresentInfoKHR *present_info) const;

private:
    VkQueue queue_hdl_;
    bool shared_;
    mutable std::mutex mutex_;
};

struct GPURunUtil {
    VkCommandPool pool;
    VkCommandBuffer cmd;
    VkQueue queue;
    VkFence fence;

    void begin(const Device &dev) const;
    void submit(const Device &dev) const;
};

inline VkDeviceAddress getDevAddr(const Device &dev, VkBuffer buf);

inline VkCommandPool makeCmdPool(const Device &dev, uint32_t qf_idx);

inline VkCommandBuffer makeCmdBuffer(
    const Device &dev,
    VkCommandPool pool,
    VkCommandBufferLevel level = VK_COMMAND_BUFFER_LEVEL_PRIMARY);

inline VkQueue makeQueue(const Device &dev,
                         uint32_t qf_idx,
                         uint32_t queue_idx);
inline VkSemaphore makeBinarySemaphore(const Device &dev);

inline VkSemaphore makeBinaryExternalSemaphore(const Device &dev);
int exportBinarySemaphore(const Device &dev, VkSemaphore semaphore);

inline VkFence makeFence(const Device &dev, bool pre_signal = false);

VkSampler makeImmutableSampler(const Device &dev,
                               VkSamplerAddressMode mode);

inline void waitForFenceInfinitely(const Device &dev, VkFence fence);

inline void resetFence(const Device &dev, VkFence fence);

inline VkDescriptorSet makeDescriptorSet(const Device &dev,
                                         VkDescriptorPool pool,
                                         VkDescriptorSetLayout layout);

inline VkDeviceSize alignOffset(VkDeviceSize offset, VkDeviceSize alignment);

template <typename T>
inline T divideRoundUp(T a, T b);

inline uint32_t getWorkgroupSize(uint32_t num_items);

void printVkError(VkResult res, const char *msg);

static inline VkResult checkVk(VkResult res,
                               const char *compiler_name,
                               const char *file,
                               int line,
                               const char *msg,
                               bool is_fatal = true) noexcept
{
    if (res != VK_SUCCESS) {
        printVkError(res, msg);
        if (is_fatal) {
            fatal(file, line, compiler_name, msg);
        }
    }

    return res;
}

#define STRINGIFY_HELPER(m) #m
#define STRINGIFY(m) STRINGIFY_HELPER(m)

#define LOC_APPEND(m) m ": " __FILE__ " @ " STRINGIFY(__LINE__)
#define REQ_VK(expr) ::madrona::render::vk::checkVk((expr), MADRONA_COMPILER_FUNCTION_NAME, __FILE__, __LINE__, #expr)
#define CHK_VK(expr) ::madrona::render::vk::checkVk((expr), MADRONA_COMPILER_FUNCTION_NAME, __FILE__, __LINE__, #expr, false)

}

#include "utils.inl"
