namespace madrona::render::vk {

VkFormat MemoryAllocator::getTextureFormat(TextureFormat fmt) const
{
    return texture_formats_[static_cast<uint32_t>(fmt)];
}

}
