namespace madrona {
namespace render {
namespace vk {

VkFormat MemoryAllocator::getTextureFormat(TextureFormat fmt) const
{
    return texture_formats_[static_cast<uint32_t>(fmt)];
}

VkFormat MemoryAllocator::getColorAttachmentFormat() const
{
    return color_attach_fmt_;
}

VkFormat MemoryAllocator::getDepthAttachmentFormat() const
{
    return depth_attach_fmt_;
}

}
}
}
