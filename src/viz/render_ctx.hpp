#pragma once

#include <madrona/render/render_mgr.hpp>

#include "batch_renderer.hpp"

namespace madrona::render {

struct RenderContext {
    RenderContext(APIBackend *render_backend,
                  GPUDevice *render_dev,
                  const RenderManager::Config &cfg);
    RenderContext(const RenderContext &) = delete;
    ~RenderContext();

    CountT loadObjects(Span<const imp::SourceObject> src_objs,
                       Span<const imp::SourceMaterial> src_mats,
                       Span<const imp::SourceTexture> textures);

    void configureLighting(Span<const LightConfig> lights);

    void waitForIdle();

    vk::Backend &backend;
    vk::Device &dev;
    vk::MemoryAllocator alloc;

    VkQueue renderQueue;

    uint32_t br_width_;
    uint32_t br_height_;

    VkPipelineCache pipelineCache;
    VkSampler repeatSampler;
    VkSampler clampSampler;

    VkRenderPass renderPass;
    VkRenderPass shadowPass;

    Pipeline<1> instanceCull;
    Pipeline<1> objectDraw;

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

    HeapArray<render::shader::DirectionalLight> lights_;

    DynArray<AssetData> loaded_assets_;

    Sky sky_;

    DynArray<MaterialTexture> material_textures_;
    VoxelConfig voxel_config_;

    uint32_t num_worlds_;
    std::unique_ptr<BatchRenderer> batchRenderer;

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
};

}
