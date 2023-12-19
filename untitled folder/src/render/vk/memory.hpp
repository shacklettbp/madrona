#pragma once

#include <atomic>
#include <utility>
#include <optional>

#include <madrona/render/vk/backend.hpp>

#include "utils.hpp"
#include "vulkan/vulkan_core.h"

namespace madrona {
namespace render {
namespace vk {

class MemoryAllocator;

template <bool host_mapped>
class AllocDeleter {
public:
    AllocDeleter(VkDeviceMemory mem, MemoryAllocator &alloc)
        : mem_(mem),
          alloc_(&alloc)
    {}

    void operator()(VkBuffer buffer) const;
    void operator()(VkImage image) const;

    void clear();

    VkDeviceMemory hdl();

private:
    VkDeviceMemory mem_;

    MemoryAllocator *alloc_;
};

class HostBuffer {
public:
    HostBuffer(const HostBuffer &) = delete;
    HostBuffer(HostBuffer &&o);
    ~HostBuffer();

    HostBuffer & operator=(const HostBuffer &) = delete;
    HostBuffer & operator=(HostBuffer &&);

    void flush(const Device &dev);
    void invalidate(const Device &dev);
    void flush(const Device &dev,
               VkDeviceSize offset,
               VkDeviceSize num_bytes);

    VkDeviceMemory getMemHdl();

    VkBuffer buffer;
    void *ptr;

private:
    HostBuffer(VkBuffer buf,
               void *p,
               VkMappedMemoryRange mem_range,
               AllocDeleter<true> deleter);

    VkMappedMemoryRange mem_range_;

    AllocDeleter<true> deleter_;
    friend class MemoryAllocator;
};

class LocalBuffer {
public:
    LocalBuffer(const LocalBuffer &) = delete;
    LocalBuffer(LocalBuffer &&o);
    ~LocalBuffer();

    LocalBuffer & operator=(const LocalBuffer &) = delete;
    LocalBuffer & operator=(LocalBuffer &&);

    VkBuffer buffer;

private:
    LocalBuffer(VkBuffer buf, AllocDeleter<false> deleter);

    AllocDeleter<false> deleter_;
    friend class MemoryAllocator;
};

struct DedicatedBuffer {
    LocalBuffer buf;
    VkDeviceMemory mem;
};

class LocalImage {
public:
    LocalImage(const LocalImage &) = delete;
    LocalImage(LocalImage &&o);
    ~LocalImage();

    uint32_t width;
    uint32_t height;
    uint32_t mipLevels;
    VkImage image;
    VkMemoryRequirements reqs;

private:
    LocalImage(uint32_t width,
               uint32_t height,
               uint32_t mip_levels,
               VkImage image,
               VkMemoryRequirements reqs,
               AllocDeleter<false> deleter);

    AllocDeleter<false> deleter_;
    friend class MemoryAllocator;
};

struct LocalTexture {
    uint32_t width;
    uint32_t height;
    uint32_t mipLevels;
    VkImage image;
};

struct MemoryChunk {
    VkDeviceMemory hdl;
    uint32_t offset;
    uint32_t chunkID;
};

struct MemoryTypeIndices {
    uint32_t host;
    uint32_t local;
    uint32_t dedicatedBuffer;
};

struct Alignments {
    VkDeviceSize uniformBuffer;
    VkDeviceSize storageBuffer;
};

struct TextureRequirements {
    VkDeviceSize alignment;
    VkDeviceSize size;
};

enum class TextureFormat : uint32_t {
    R8G8B8A8_SRGB,
    R8G8B8A8_UNORM,
    R8G8_UNORM,
    R8_UNORM,
    R32G32B32A32_SFLOAT,
    R32_SFLOAT,
    BC5_UNORM,
    BC7_UNORM,
    BC7_SRGB,
    COUNT,
};

uint32_t getTexelBytes(TextureFormat fmt);

class MemoryAllocator {
public:
    MemoryAllocator(const Device &dev, const Backend &backend);
    MemoryAllocator(const MemoryAllocator &) = delete;
    MemoryAllocator(MemoryAllocator &&) = default;

    HostBuffer makeStagingBuffer(VkDeviceSize num_bytes,
                                 bool dev_addr = false);
    HostBuffer makeParamBuffer(VkDeviceSize num_bytes,
                               bool dev_addr = false);
    HostBuffer makeHostBuffer(VkDeviceSize num_bytes, bool dev_addr = false);

    std::optional<LocalBuffer> makeIndirectBuffer(VkDeviceSize num_bytes);
    std::optional<LocalBuffer> makeLocalBuffer(VkDeviceSize num_bytes,
                                               bool dev_addr = false);

    DedicatedBuffer makeDedicatedBuffer(VkDeviceSize num_bytes,
        bool dev_addr = false, bool support_export = false);

    inline VkFormat getTextureFormat(TextureFormat fmt) const;

    std::pair<LocalTexture, TextureRequirements>
    makeTexture1D(uint32_t width, uint32_t mip_levels, VkFormat fmt);

    std::pair<LocalTexture, TextureRequirements>
    makeTexture2D(uint32_t width, uint32_t height, uint32_t mip_levels,
                VkFormat fmt);

    std::pair<LocalTexture, TextureRequirements>
    makeTexture3D(uint32_t width, uint32_t height, uint32_t depth,
                  uint32_t mip_levels, VkFormat fmt);

    void destroyTexture(LocalTexture &&texture);

    std::optional<VkDeviceMemory> alloc(VkDeviceSize num_bytes);

    VkDeviceSize alignUniformBufferOffset(VkDeviceSize offset) const;
    VkDeviceSize alignStorageBufferOffset(VkDeviceSize offset) const;

    LocalImage makeColorAttachment(
        uint32_t width, uint32_t height, VkFormat format);
    LocalImage makeDepthAttachment(
        uint32_t width, uint32_t height, VkFormat format);

    LocalImage makeConversionImage(
        uint32_t width, uint32_t height, VkFormat fmt);

private:
    HostBuffer makeHostBuffer(VkDeviceSize num_bytes,
                              VkBufferUsageFlags usage,
                              bool dev_addr);

    template <int dims>
    std::pair<LocalTexture, TextureRequirements>
    makeTexture(uint32_t width, uint32_t height, uint32_t depth,
                uint32_t mip_levels, VkFormat fmt);


    std::optional<LocalBuffer> makeLocalBuffer(VkDeviceSize num_bytes,
                                               VkBufferUsageFlags usage,
                                               bool dev_addr);

    LocalImage makeDedicatedImage(uint32_t width,
                                  uint32_t height,
                                  uint32_t mip_levels,
                                  VkFormat format,
                                  VkImageUsageFlags usage,
                                  uint32_t type_idx);

    LocalImage makeDedicatedImage(uint32_t width,
                                  uint32_t height,
                                  uint32_t depth,
                                  uint32_t mip_levels,
                                  VkFormat format,
                                  VkImageUsageFlags usage,
                                  uint32_t type_idx);

    const Device &dev;
    Alignments alignments_;
    VkBufferUsageFlags local_buffer_usage_flags_;
    std::array<VkFormat, size_t(TextureFormat::COUNT)> texture_formats_;
    MemoryTypeIndices type_indices_;

    template <bool>
    friend class AllocDeleter;
};

}
}
}

#include "memory.inl"
