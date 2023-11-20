#pragma once

#include <imgui.h>

#include "batch_renderer.hpp"

namespace madrona::viz {
    
struct ViewerController;
struct ViewerInput;

};

namespace madrona::render {

struct RenderContext::Impl {
    Impl(const RenderContext::Config &cfg);
    ~Impl();

    CountT loadObjects(Span<const imp::SourceObject> src_objs,
                       Span<const imp::SourceMaterial> src_mats,
                       Span<const imp::SourceTexture> textures);

    void configureLighting(Span<const LightConfig> lights);

    void waitUntilFrameReady();

    void startFrame();

    bool renderFlycamFrame(const viz::ViewerInput &input);
    bool renderGridFrame(const viz::ViewerInput &input);

    void render(const viz::ViewerInput &input);

    void renderGUIAndPresent(const viz::ViewerInput &input,
                             bool prepare_screenshot);

    void waitForIdle();

    void selectViewerBatchView(uint32_t batch_view);

private:
    vk::Backend::LoaderLib loader_lib_;

public:
    vk::Backend backend;
    Optional<Window> window;
    vk::Device dev;
    vk::MemoryAllocator alloc;

private:
    VkQueue render_queue_;

    // Fixme remove
    render::vk::QueueState present_wrapper_;

    uint32_t fb_width_;
    uint32_t fb_height_;

    uint32_t br_width_;
    uint32_t br_height_;

    std::array<VkClearValue, 4> fb_clear_;
    std::array<VkClearValue, 2> fb_shadow_clear_;
    std::array<VkClearValue, 2> fb_imgui_clear_;
    Optional<PresentationState> present_;
    VkPipelineCache pipeline_cache_;
    VkSampler repeat_sampler_;
    VkSampler clamp_sampler_;
    VkRenderPass render_pass_;
    VkRenderPass shadow_pass_;
    Optional<ImGuiRenderState> imgui_render_state_;

    Pipeline<1> instance_cull_;
    Pipeline<1> object_draw_;
    Pipeline<1> object_shadow_draw_;
    Pipeline<1> deferred_lighting_;
    Pipeline<1> shadow_gen_;
    Pipeline<1> blur_;
    Pipeline<1> voxel_mesh_gen_;
    Pipeline<1> voxel_draw_;
    Optional<Pipeline<1>> quad_draw_;
    // Draw a grid
    Optional<Pipeline<1>> grid_draw_;

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

    uint32_t cur_frame_;
    HeapArray<Frame> frames_;
    DynArray<AssetData> loaded_assets_;

    Sky sky_;

    DynArray<MaterialTexture> material_textures_;
    VoxelConfig voxel_config_;

    // This is just a prototype
    int gpu_id_;
    uint32_t num_worlds_;
    std::unique_ptr<BatchRenderer> br_proto_;

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
    std::unique_ptr<render::vk::HostBuffer> screenshot_buffer_;

    uint64_t global_frame_no_;

    ImGuiContext *imgui_ctx_;

    friend struct viz::ViewerController;
    friend struct RenderContext;
};

}
