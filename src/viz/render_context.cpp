#include <madrona/render/vk/backend.hpp>
#include <madrona/viz/render_context.hpp>

#include "render_common.hpp"
#include "batch_renderer.hpp"

namespace madrona::render {

using namespace vk;

struct RenderContext::Impl {
    Impl(const RenderContext::Config &cfg);

    ~Impl();

private:
    vk::Backend::LoaderLib loader_lib_;

public:
    vk::Backend backend;
    vk::Device dev;
    vk::MemoryAllocator alloc;

private:
    VkQueue render_queue_;

    // Fixme remove
    render::vk::QueueState present_wrapper_;

    uint32_t br_width_;
    uint32_t br_height_;

    VkPipelineCache pipeline_cache_;
    VkSampler repeat_sampler_;
    VkSampler clamp_sampler_;
    VkRenderPass render_pass_;
    VkRenderPass shadow_pass_;

    Pipeline<1> instance_cull_;
    Pipeline<1> object_draw_;
    Pipeline<1> object_shadow_draw_;
    Pipeline<1> deferred_lighting_;
    Pipeline<1> shadow_gen_;
    Pipeline<1> blur_;
    Pipeline<1> voxel_mesh_gen_;
    Pipeline<1> voxel_draw_;
    Pipeline<1> quad_draw_;

    render::vk::FixedDescriptorPool asset_desc_pool_cull_;
    render::vk::FixedDescriptorPool asset_desc_pool_draw_;
    render::vk::FixedDescriptorPool asset_desc_pool_mat_tx_;

    VkDescriptorSet asset_set_cull_;
    VkDescriptorSet asset_set_draw_;
    VkDescriptorSet asset_set_mat_tex_;

    VkCommandPool load_cmd_pool_;
    VkCommandBuffer load_cmd_;
    VkFence load_fence_;

    EngineInterop engine_interop_;

    HeapArray<shader::DirectionalLight> lights_;

    uint32_t cur_frame_;
    HeapArray<Frame> frames_;
    DynArray<AssetData> loaded_assets_;

    Sky sky_;

    DynArray<MaterialTexture> material_textures_;
    VoxelConfig voxel_config_;

    // This is just a prototype
    int gpu_id_;
    uint32_t num_worlds_;
    std::unique_ptr<BatchRenderer> batch_renderer_;

    VkDescriptorSetLayout asset_layout_;
    VkDescriptorSetLayout asset_tex_layout_;

    // This contains the vertex data buffer and the mesh data buffer
    VkDescriptorSetLayout asset_batch_lighting_layout_;

    VkDescriptorPool asset_pool_;
    VkDescriptorSet asset_set_tex_compute_;
    VkDescriptorSet asset_batch_lighting_set_;

    // This descriptor set contains information about the sky
    VkDescriptorSetLayout sky_data_layout_;
    VkDescriptorSet sky_data_set_;

    bool gpu_input_;

    // This is only used if we are on the CPU backend
    uint32_t *iota_array_;
};

Backend::LoaderLib initLoaderLib(bool make_window)
{
#ifdef MADRONA_MACOS
    Backend::LoaderLib loader_lib = Backend::loadLoaderLib();

    if (make_window) {
        glfwInitVulkanLoader((PFN_vkGetInstanceProcAddr)loader_lib.getEntryFn());
    }
#else
    Backend::LoaderLib loader_lib(nullptr, nullptr);
#endif

    if (make_window) {
        if (!glfwInit()) {
            FATAL("Failed to initialize GLFW");
        }
    }

    return loader_lib;
}

static Backend initBackend()
{
    auto get_inst_addr = glfwGetInstanceProcAddress(
        VK_NULL_HANDLE, "vkGetInstanceProcAddr");

    bool enable_validation;
    char *validate_env = getenv("MADRONA_RENDER_VALIDATE");
    if (!validate_env || validate_env[0] == '0') {
        enable_validation = false;
    } else {
        enable_validation = true;
    }

    return Backend((void (*)())get_inst_addr, enable_validation, false,
                   PresentationState::getInstanceExtensions());
}

RenderContext::Impl::Impl(const RenderContext::Config &cfg)
    : loader_lib_(initLoaderLib(cfg.renderViewer))
{
}

}
