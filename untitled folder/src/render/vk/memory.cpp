#include "memory.hpp"
#include <vulkan/vulkan_core.h>

#include "utils.hpp"
#include "config.hpp"

#include <cstring>
#include <iostream>

using namespace std;

namespace madrona {
namespace render {
namespace vk {

namespace BufferFlags {
static constexpr VkBufferUsageFlags commonUsage =
    VK_BUFFER_USAGE_TRANSFER_DST_BIT;

static constexpr VkBufferUsageFlags stageUsage =
    VK_BUFFER_USAGE_TRANSFER_SRC_BIT;

static constexpr VkBufferUsageFlags geometryUsage =
    VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT;

static constexpr VkBufferUsageFlags shaderUsage =
    VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;

static constexpr VkBufferUsageFlags paramUsage =
    VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT;

static constexpr VkBufferUsageFlags hostRTUsage =
    VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR;

static constexpr VkBufferUsageFlags hostUsage =
    stageUsage | shaderUsage | paramUsage | hostRTUsage | commonUsage;

static constexpr VkBufferUsageFlags indirectUsage =
    VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT;

static constexpr VkBufferUsageFlags rtGeometryUsage =
    VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;

static constexpr VkBufferUsageFlags rtAccelScratchUsage =
    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;

static constexpr VkBufferUsageFlags rtAccelUsage =
    VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR;

static constexpr VkBufferUsageFlags localRTUsage =
    rtGeometryUsage | rtAccelScratchUsage | rtAccelUsage;

static constexpr VkBufferUsageFlags localUsage =
    commonUsage | geometryUsage | shaderUsage | indirectUsage | localRTUsage;

static constexpr VkBufferUsageFlags dedicatedUsage =
    localUsage | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
};

namespace ImageFlags {
static constexpr VkFormatFeatureFlags textureReqs =
    VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT;

static constexpr VkImageUsageFlags textureUsage =
    VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;

static constexpr VkImageUsageFlags colorAttachmentUsage =
    VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_STORAGE_BIT;

static constexpr VkImageUsageFlags depthAttachmentUsage =
    VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT |
    VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
    VK_IMAGE_USAGE_SAMPLED_BIT;

#if 0
static constexpr VkImageUsageFlags rtStorageUsage =
    VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;

static constexpr VkFormatFeatureFlags rtStorageReqs =
    VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT | VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT;
#endif
};

template <bool host_mapped>
void AllocDeleter<host_mapped>::operator()(VkBuffer buffer) const
{
    if (mem_ == VK_NULL_HANDLE) return;

    const Device &dev = alloc_->dev;

    if constexpr (host_mapped) {
        dev.dt.unmapMemory(dev.hdl, mem_);
    }

    dev.dt.destroyBuffer(dev.hdl, buffer, nullptr);
    dev.dt.freeMemory(dev.hdl, mem_, nullptr);
}

template <>
void AllocDeleter<false>::operator()(VkImage image) const
{
    if (mem_ == VK_NULL_HANDLE) return;

    const Device &dev = alloc_->dev;

    dev.dt.destroyImage(dev.hdl, image, nullptr);

    dev.dt.freeMemory(dev.hdl, mem_, nullptr);
}

template <bool host_mapped>
void AllocDeleter<host_mapped>::clear()
{
    mem_ = VK_NULL_HANDLE;
}

template <bool host_mapped>
VkDeviceMemory AllocDeleter<host_mapped>::hdl()
{
    return mem_;
}

HostBuffer::HostBuffer(VkBuffer buf,
                       void *p,
                       VkMappedMemoryRange mem_range,
                       AllocDeleter<true> deleter)
    : buffer(buf),
      ptr(p),
      mem_range_(mem_range),
      deleter_(deleter)
{}

HostBuffer::HostBuffer(HostBuffer &&o)
    : buffer(o.buffer),
      ptr(o.ptr),
      mem_range_(o.mem_range_),
      deleter_(o.deleter_)
{
    o.deleter_.clear();
}

HostBuffer & HostBuffer::operator=(HostBuffer &&o)
{
    buffer = o.buffer;
    ptr = o.ptr;
    mem_range_ = o.mem_range_;
    deleter_ = o.deleter_;

    o.deleter_.clear();

    return *this;
}

HostBuffer::~HostBuffer()
{
    deleter_(buffer);
}

void HostBuffer::flush(const Device &dev)
{
    dev.dt.flushMappedMemoryRanges(dev.hdl, 1, &mem_range_);
}

void HostBuffer::invalidate(const Device &dev)
{
    dev.dt.invalidateMappedMemoryRanges(dev.hdl, 1, &mem_range_);
}

void HostBuffer::flush(const Device &dev,
                       VkDeviceSize offset,
                       VkDeviceSize num_bytes)
{
    VkMappedMemoryRange sub_range = mem_range_;
    sub_range.offset = offset;
    sub_range.size = num_bytes;
    dev.dt.flushMappedMemoryRanges(dev.hdl, 1, &sub_range);
}

VkDeviceMemory HostBuffer::getMemHdl() 
{
    return deleter_.hdl();
}

LocalBuffer::LocalBuffer(VkBuffer buf, AllocDeleter<false> deleter)
    : buffer(buf),
      deleter_(deleter)
{}

LocalBuffer::LocalBuffer(LocalBuffer &&o)
    : buffer(o.buffer),
      deleter_(o.deleter_)
{
    o.deleter_.clear();
}

LocalBuffer & LocalBuffer::operator=(LocalBuffer &&o)
{
    deleter_(buffer);

    buffer = o.buffer;
    deleter_ = o.deleter_;

    o.deleter_.clear();

    return *this;
}

LocalBuffer::~LocalBuffer()
{
    deleter_(buffer);
}

LocalImage::LocalImage(uint32_t w,
                       uint32_t h,
                       uint32_t mip_levels,
                       VkImage img,
                       VkMemoryRequirements reqs,
                       AllocDeleter<false> deleter)
    : width(w),
      height(h),
      mipLevels(mip_levels),
      image(img),
      reqs(reqs),
      deleter_(deleter)
{}

LocalImage::LocalImage(LocalImage &&o)
    : width(o.width),
      height(o.height),
      mipLevels(o.mipLevels),
      image(o.image),
      reqs(o.reqs),
      deleter_(o.deleter_)
{
    o.deleter_.clear();
}

LocalImage::~LocalImage()
{
    deleter_(image);
}

static VkFormatProperties2 getFormatProperties(const Backend &backend,
                                               VkPhysicalDevice phy,
                                               VkFormat fmt)
{
    VkFormatProperties2 props;
    props.sType = VK_STRUCTURE_TYPE_FORMAT_PROPERTIES_2;
    props.pNext = nullptr;

    backend.dt.getPhysicalDeviceFormatProperties2(phy, fmt, &props);
    return props;
}

template <size_t N>
static VkFormat chooseFormat(VkPhysicalDevice phy,
                             const Backend &backend,
                             VkFormatFeatureFlags required_features,
                             const array<VkFormat, N> &desired_formats)
{
    for (auto fmt : desired_formats) {
        VkFormatProperties2 props = getFormatProperties(backend, phy, fmt);
        if ((props.formatProperties.optimalTilingFeatures &
             required_features) == required_features) {
            return fmt;
        }

        cerr << "Warning: preferred format not available" << endl;
    }

    FATAL("Unable to find required features in given formats");
}

static pair<VkBuffer, VkMemoryRequirements> makeUnboundBuffer(
    const Device &dev,
    VkDeviceSize num_bytes,
    VkBufferUsageFlags usage)
{
    VkBufferCreateInfo buffer_info;
    buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buffer_info.pNext = nullptr;
    buffer_info.flags = 0;
    buffer_info.size = num_bytes;
    buffer_info.usage = usage;
    buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    buffer_info.pQueueFamilyIndices = nullptr;
    buffer_info.queueFamilyIndexCount = 0;

    VkBuffer buffer;
    REQ_VK(dev.dt.createBuffer(dev.hdl, &buffer_info, nullptr, &buffer));

    VkMemoryRequirements reqs;
    dev.dt.getBufferMemoryRequirements(dev.hdl, buffer, &reqs);

    return pair(buffer, reqs);
}

template <int dims>
static VkImage makeImage(const Device &dev,
                         uint32_t width,
                         uint32_t height,
                         uint32_t depth,
                         uint32_t mip_levels,
                         VkFormat format,
                         VkImageUsageFlags usage,
                         VkImageCreateFlags img_flags = 0)
{
    VkImageCreateInfo img_info;
    img_info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    img_info.pNext = nullptr;
    img_info.flags = img_flags;

    if constexpr (dims == 1) {
        img_info.imageType = VK_IMAGE_TYPE_1D;
    } else if constexpr (dims == 2) {
        img_info.imageType = VK_IMAGE_TYPE_2D;
    } else if constexpr (dims == 3) {
        img_info.imageType = VK_IMAGE_TYPE_3D;
    }

    img_info.format = format;
    img_info.extent = {width, height, depth};
    img_info.mipLevels = mip_levels;
    img_info.arrayLayers = 1;
    img_info.samples = VK_SAMPLE_COUNT_1_BIT;
    img_info.tiling = VK_IMAGE_TILING_OPTIMAL;
    img_info.usage = usage;
    img_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    img_info.queueFamilyIndexCount = 0;
    img_info.pQueueFamilyIndices = nullptr;
    img_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    VkImage img;
    REQ_VK(dev.dt.createImage(dev.hdl, &img_info, nullptr, &img));

    return img;
}

uint32_t findMemoryTypeIndex(uint32_t allowed_type_bits,
                             VkMemoryPropertyFlags required_props,
                             VkPhysicalDeviceMemoryProperties2 &mem_props)
{
    uint32_t num_mems = mem_props.memoryProperties.memoryTypeCount;

    for (uint32_t idx = 0; idx < num_mems; idx++) {
        uint32_t mem_type_bits = (1 << idx);
        if (!(allowed_type_bits & mem_type_bits)) continue;

        VkMemoryPropertyFlags supported_props =
            mem_props.memoryProperties.memoryTypes[idx].propertyFlags;

        if ((required_props & supported_props) == required_props) {
            return idx;
        }
    }

    FATAL("Failed to find desired memory type");
}

static VkMemoryRequirements getImageMemReqs(const Device &dev,
                                            VkImage img)
{
    VkMemoryRequirements reqs;
    dev.dt.getImageMemoryRequirements(dev.hdl, img, &reqs);

    return reqs;
}

static MemoryTypeIndices findTypeIndices(const Device &dev,
    const Backend &backend,
    const array<VkFormat, size_t(TextureFormat::COUNT)> &tex_formats)
{
    auto get_generic_buffer_reqs = [&](VkBufferUsageFlags usage_flags) {
        auto [test_buffer, reqs] = makeUnboundBuffer(dev, 1, usage_flags);

        dev.dt.destroyBuffer(dev.hdl, test_buffer, nullptr);

        return reqs;
    };

    auto get_generic_image_reqs = [&](VkFormat format,
                                      VkImageUsageFlags usage_flags,
                                      VkImageCreateFlags img_flags = 0) {
        VkImage test_image =
            makeImage<2>(dev, 1, 1, 1, 1, format, usage_flags, img_flags);

        VkMemoryRequirements reqs = getImageMemReqs(dev, test_image);

        dev.dt.destroyImage(dev.hdl, test_image, nullptr);

        return reqs;
    };

    VkPhysicalDeviceMemoryProperties2 dev_mem_props;
    dev_mem_props.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2;
    dev_mem_props.pNext = nullptr;
    backend.dt.getPhysicalDeviceMemoryProperties2(dev.phy, &dev_mem_props);

    VkMemoryRequirements host_generic_reqs =
        get_generic_buffer_reqs(BufferFlags::hostUsage);

    uint32_t host_type_idx =
        findMemoryTypeIndex(host_generic_reqs.memoryTypeBits,
                            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                VK_MEMORY_PROPERTY_HOST_CACHED_BIT,
                            dev_mem_props);

    VkMemoryRequirements buffer_local_reqs =
        get_generic_buffer_reqs(BufferFlags::localUsage);

    // FIXME: This isn't a complete check for the local memory type,
    // specifically color attachment usages are missing, tex_bits doesn't
    // account for it

    uint32_t tex_bits = ~0u;

    for (int i = 0; i < (int)tex_formats.size() - 1; i++) {
        VkMemoryRequirements tex_local_reqs =
            get_generic_image_reqs(tex_formats[i], ImageFlags::textureUsage);

        tex_bits &= tex_local_reqs.memoryTypeBits;
    }

    uint32_t local_type_idx = findMemoryTypeIndex(
        buffer_local_reqs.memoryTypeBits & tex_bits,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, dev_mem_props);

    VkMemoryRequirements dedicated_reqs =
        get_generic_buffer_reqs(BufferFlags::dedicatedUsage);

    uint32_t dedicated_type_idx = findMemoryTypeIndex(
        dedicated_reqs.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        dev_mem_props);

    return MemoryTypeIndices {
        host_type_idx,
        local_type_idx,
        dedicated_type_idx,
    };
}

static Alignments getMemoryAlignments(const Backend &backend,
                                      VkPhysicalDevice phy)
{
    VkPhysicalDeviceProperties2 props {};
    props.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    backend.dt.getPhysicalDeviceProperties2(phy, &props);

    return Alignments {
        props.properties.limits.minUniformBufferOffsetAlignment,
        props.properties.limits.minStorageBufferOffsetAlignment};
}

uint32_t getTexelBytes(TextureFormat fmt)
{
    static array<uint32_t, size_t(TextureFormat::COUNT)> fmt_sizes {
        4,
        4,
        2,
        1,
        16,
        4,
        1,
        1,
        1,
    };

    return fmt_sizes[static_cast<uint32_t>(fmt)];
}

static array<VkFormat, size_t(TextureFormat::COUNT)> chooseTextureFormats(
    const Device &dev, const Backend &backend)
{
    array<VkFormat, size_t(TextureFormat::COUNT)> fmts;

    auto setFormat = [&](TextureFormat fmt,
                         const auto &candidates) {
        fmts[static_cast<uint32_t>(fmt)] =
            chooseFormat(dev.phy, backend, ImageFlags::textureReqs,
                         candidates);
    };

    setFormat(TextureFormat::R8G8B8A8_SRGB, array {
        VK_FORMAT_R8G8B8A8_SRGB,
    });

    setFormat(TextureFormat::R8G8B8A8_UNORM, array {
        VK_FORMAT_R8G8B8A8_UNORM,
    });

    setFormat(TextureFormat::R8G8_UNORM, array {
        VK_FORMAT_R8G8_UNORM,
        VK_FORMAT_R8G8B8A8_UNORM,
    });

    setFormat(TextureFormat::R8_UNORM, array {
        VK_FORMAT_R8_UNORM,
        VK_FORMAT_R8G8B8A8_UNORM,
    });

    setFormat(TextureFormat::R32G32B32A32_SFLOAT, array {
        VK_FORMAT_R32G32B32A32_SFLOAT,
    });

    setFormat(TextureFormat::R32_SFLOAT, array {
        VK_FORMAT_R32_SFLOAT,
    });

    setFormat(TextureFormat::BC5_UNORM, array {
        VK_FORMAT_BC5_UNORM_BLOCK,
    });

    setFormat(TextureFormat::BC7_UNORM, array {
        VK_FORMAT_BC7_UNORM_BLOCK,
    });

    setFormat(TextureFormat::BC7_SRGB, array {
        VK_FORMAT_BC7_SRGB_BLOCK,
    });

    return fmts;
}

MemoryAllocator::MemoryAllocator(const Device &d,
                                 const Backend &backend)
    : dev(d),
      alignments_(getMemoryAlignments(backend, dev.phy)),
      local_buffer_usage_flags_(BufferFlags::localUsage),
      texture_formats_(chooseTextureFormats(dev, backend)),
      type_indices_(findTypeIndices(dev, backend, texture_formats_))
{
}

HostBuffer MemoryAllocator::makeHostBuffer(VkDeviceSize num_bytes,
                                           bool dev_addr)
{
    return makeHostBuffer(num_bytes, BufferFlags::hostUsage, dev_addr);
}

HostBuffer MemoryAllocator::makeHostBuffer(VkDeviceSize num_bytes,
                                           VkBufferUsageFlags usage,
                                           bool dev_addr)
{
    auto [buffer, reqs] = makeUnboundBuffer(dev, num_bytes, usage);

    VkMemoryAllocateInfo alloc;
    alloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc.allocationSize = reqs.size;
    alloc.memoryTypeIndex = type_indices_.host;

    VkMemoryAllocateFlagsInfo flag_info;
    if (dev_addr) {
        flag_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO;
        flag_info.pNext = nullptr;
        flag_info.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
        flag_info.deviceMask = 0;

        alloc.pNext = &flag_info;
    } else {
        alloc.pNext = nullptr;
    }

    VkDeviceMemory memory;
    REQ_VK(dev.dt.allocateMemory(dev.hdl, &alloc, nullptr, &memory));
    REQ_VK(dev.dt.bindBufferMemory(dev.hdl, buffer, memory, 0));

    void *mapped_ptr;
    REQ_VK(dev.dt.mapMemory(dev.hdl, memory, 0, reqs.size, 0, &mapped_ptr));

    VkMappedMemoryRange range;
    range.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
    range.pNext = nullptr;
    range.memory = memory, range.offset = 0;
    range.size = VK_WHOLE_SIZE;

    return HostBuffer(buffer, mapped_ptr, range,
                      AllocDeleter<true>(memory, *this));
}

HostBuffer MemoryAllocator::makeStagingBuffer(VkDeviceSize num_bytes,
                                              bool dev_addr)
{
    return makeHostBuffer(num_bytes, BufferFlags::stageUsage, dev_addr);
}

HostBuffer MemoryAllocator::makeParamBuffer(VkDeviceSize num_bytes,
                                            bool dev_addr)
{
    return makeHostBuffer(num_bytes, BufferFlags::commonUsage |
                                         BufferFlags::shaderUsage |
                                         BufferFlags::paramUsage,
                                         dev_addr);
}

optional<LocalBuffer> MemoryAllocator::makeLocalBuffer(
    VkDeviceSize num_bytes,
    VkBufferUsageFlags usage,
    bool dev_addr)
{
    if (dev_addr) {
        usage |= VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    }

    auto [buffer, reqs] = makeUnboundBuffer(dev, num_bytes, usage);

    VkMemoryAllocateInfo alloc;
    alloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc.pNext = nullptr;
    alloc.allocationSize = reqs.size;
    alloc.memoryTypeIndex = type_indices_.local;

    VkMemoryAllocateFlagsInfo flag_info;
    if (dev_addr) {
        flag_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO;
        flag_info.pNext = nullptr;
        flag_info.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
        flag_info.deviceMask = 0;

        alloc.pNext = &flag_info;
    } else {
        alloc.pNext = nullptr;
    }

    VkDeviceMemory memory;
    REQ_VK(dev.dt.allocateMemory(dev.hdl, &alloc, nullptr, &memory));
    REQ_VK(dev.dt.bindBufferMemory(dev.hdl, buffer, memory, 0));

    return LocalBuffer(buffer, AllocDeleter<false>(memory, *this));
}

optional<LocalBuffer> MemoryAllocator::makeLocalBuffer(VkDeviceSize num_bytes,
                                                       bool dev_addr)
{
    return makeLocalBuffer(num_bytes, local_buffer_usage_flags_, dev_addr);
}

optional<LocalBuffer> MemoryAllocator::makeIndirectBuffer(
    VkDeviceSize num_bytes)
{
    return makeLocalBuffer(num_bytes, BufferFlags::commonUsage |
                                      BufferFlags::shaderUsage |
                                      BufferFlags::indirectUsage);
}

DedicatedBuffer MemoryAllocator::makeDedicatedBuffer(
    VkDeviceSize num_bytes, bool dev_addr, bool support_export)
{
    auto usage = BufferFlags::dedicatedUsage;
    if (dev_addr) {
        usage |= VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    }

    VkExternalMemoryBufferCreateInfo buffer_ext_info;
    VkExternalMemoryBufferCreateInfo *buffer_ext_info_ptr;
    if (support_export) {
        buffer_ext_info.sType =
            VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO;
        buffer_ext_info.pNext = nullptr;
        buffer_ext_info.handleTypes =
            VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
        buffer_ext_info_ptr = &buffer_ext_info;
    } else {
        buffer_ext_info_ptr = nullptr;
    }

    VkBufferCreateInfo buffer_info;
    buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buffer_info.pNext = buffer_ext_info_ptr;
    buffer_info.flags = 0;
    buffer_info.size = num_bytes;
    buffer_info.usage = usage;
    buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    buffer_info.pQueueFamilyIndices = nullptr;
    buffer_info.queueFamilyIndexCount = 0;

    VkBuffer buffer;
    REQ_VK(dev.dt.createBuffer(dev.hdl, &buffer_info, nullptr, &buffer));

    VkMemoryRequirements reqs;
    dev.dt.getBufferMemoryRequirements(dev.hdl, buffer, &reqs);

    VkExportMemoryAllocateInfo export_info;
    VkExportMemoryAllocateInfo *export_info_ptr;
    if (support_export) {
        export_info.sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO;
        export_info.pNext = nullptr;
        export_info.handleTypes =
            VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;

        export_info_ptr = &export_info;
    } else {
        export_info_ptr = nullptr;
    }

    VkMemoryDedicatedAllocateInfo dedicated;
    dedicated.sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO;
    dedicated.image = VK_NULL_HANDLE;
    dedicated.buffer = buffer;

    VkMemoryAllocateInfo alloc;
    alloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc.pNext = &dedicated;
    alloc.allocationSize = reqs.size;
    alloc.memoryTypeIndex = type_indices_.dedicatedBuffer;

    VkMemoryAllocateFlagsInfo flag_info;
    if (dev_addr) {
        flag_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO;
        flag_info.pNext = export_info_ptr;
        flag_info.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
        flag_info.deviceMask = 0;

        dedicated.pNext = &flag_info;
    } else {
        dedicated.pNext = export_info_ptr;
    }

    VkDeviceMemory memory;
    REQ_VK(dev.dt.allocateMemory(dev.hdl, &alloc, nullptr, &memory));
    REQ_VK(dev.dt.bindBufferMemory(dev.hdl, buffer, memory, 0));

    return DedicatedBuffer {
        LocalBuffer(buffer, AllocDeleter<false>(memory, *this)),
        memory,
    };
}

pair<LocalTexture, TextureRequirements> MemoryAllocator::makeTexture1D(
    uint32_t width,
    uint32_t mip_levels,
    VkFormat fmt)
{
    return makeTexture<1>(width, 1, 1, mip_levels, fmt);
}


pair<LocalTexture, TextureRequirements> MemoryAllocator::makeTexture2D(
    uint32_t width,
    uint32_t height,
    uint32_t mip_levels,
    VkFormat fmt)
{
    return makeTexture<2>(width, height, 1, mip_levels, fmt);
}

pair<LocalTexture, TextureRequirements> MemoryAllocator::makeTexture3D(
    uint32_t width,
    uint32_t height,
    uint32_t depth,
    uint32_t mip_levels,
    VkFormat fmt)
{
    return makeTexture<3>(width, height, depth, mip_levels, fmt);
}

template <int dims>
pair<LocalTexture, TextureRequirements> MemoryAllocator::makeTexture(
    uint32_t width,
    uint32_t height,
    uint32_t depth,
    uint32_t mip_levels,
    VkFormat fmt)
{
    VkImage texture_img = makeImage<dims>(dev, width, height, depth,
        mip_levels, fmt, ImageFlags::textureUsage);

    auto reqs = getImageMemReqs(dev, texture_img);

    return {
        LocalTexture {
            width,
            height,
            mip_levels,
            texture_img,
        },
        TextureRequirements {
            reqs.alignment,
            reqs.size,
        },
    };
}

void MemoryAllocator::destroyTexture(LocalTexture &&texture)
{
    dev.dt.destroyImage(dev.hdl, texture.image, nullptr);
}

optional<VkDeviceMemory> MemoryAllocator::alloc(VkDeviceSize num_bytes)
{
    VkMemoryAllocateInfo alloc;
    alloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc.pNext = nullptr;
    alloc.allocationSize = num_bytes;
    alloc.memoryTypeIndex = type_indices_.local;

    VkDeviceMemory mem;
    VkResult res = dev.dt.allocateMemory(dev.hdl, &alloc, nullptr, &mem);

    if (res == VK_ERROR_OUT_OF_DEVICE_MEMORY) {
        return optional<VkDeviceMemory>();
    }

    return mem;
}

LocalImage MemoryAllocator::makeColorAttachment(uint32_t width,
                                                uint32_t height,
                                                VkFormat format)
{
    return makeDedicatedImage(width, height, 1,
                              format,
                              ImageFlags::colorAttachmentUsage,
                              type_indices_.local);
}

LocalImage MemoryAllocator::makeDepthAttachment(uint32_t width,
                                                uint32_t height,
                                                VkFormat format)
{
    return makeDedicatedImage(width, height, 1,
                              format,
                              ImageFlags::depthAttachmentUsage,
                              type_indices_.local);
}

LocalImage MemoryAllocator::makeConversionImage(uint32_t width,
                                                uint32_t height,
                                                VkFormat fmt)
{
    return makeDedicatedImage(width, height, 1,
                              fmt,
                              VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                                  VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                              type_indices_.local);
}

LocalImage MemoryAllocator::makeDedicatedImage(uint32_t width,
                                               uint32_t height,
                                               uint32_t mip_levels,
                                               VkFormat format,
                                               VkImageUsageFlags usage,
                                               uint32_t type_idx)
{
    auto img = makeImage<2>(dev, width, height, 1, mip_levels, format, usage);
    auto reqs = getImageMemReqs(dev, img);

    VkMemoryDedicatedAllocateInfo dedicated;
    dedicated.sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO;
    dedicated.pNext = nullptr;
    dedicated.image = img;
    dedicated.buffer = VK_NULL_HANDLE;
    VkMemoryAllocateInfo alloc;
    alloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc.pNext = &dedicated;
    alloc.allocationSize = reqs.size;
    alloc.memoryTypeIndex = type_idx;

    VkDeviceMemory memory;
    REQ_VK(dev.dt.allocateMemory(dev.hdl, &alloc, nullptr, &memory));
    REQ_VK(dev.dt.bindImageMemory(dev.hdl, img, memory, 0));

    return LocalImage(width, height, mip_levels, img, reqs,
                      AllocDeleter<false>(memory, *this));
}

LocalImage MemoryAllocator::makeDedicatedImage(uint32_t width,
                                               uint32_t height,
                                               uint32_t depth,
                                               uint32_t mip_levels,
                                               VkFormat format,
                                               VkImageUsageFlags usage,
                                               uint32_t type_idx)
{
    auto img = makeImage<3>(dev, width, height, depth, mip_levels, format, usage);
    auto reqs = getImageMemReqs(dev, img);

    VkMemoryDedicatedAllocateInfo dedicated;
    dedicated.sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO;
    dedicated.pNext = nullptr;
    dedicated.image = img;
    dedicated.buffer = VK_NULL_HANDLE;
    VkMemoryAllocateInfo alloc;
    alloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc.pNext = &dedicated;
    alloc.allocationSize = reqs.size;
    alloc.memoryTypeIndex = type_idx;

    VkDeviceMemory memory;
    REQ_VK(dev.dt.allocateMemory(dev.hdl, &alloc, nullptr, &memory));
    REQ_VK(dev.dt.bindImageMemory(dev.hdl, img, memory, 0));

    return LocalImage(width, height, mip_levels, img, reqs,
                      AllocDeleter<false>(memory, *this));
}

VkDeviceSize MemoryAllocator::alignUniformBufferOffset(
    VkDeviceSize offset) const
{
    return alignOffset(offset, alignments_.uniformBuffer);
}

VkDeviceSize MemoryAllocator::alignStorageBufferOffset(
    VkDeviceSize offset) const
{
    return alignOffset(offset, alignments_.storageBuffer);
}

}
}
}
