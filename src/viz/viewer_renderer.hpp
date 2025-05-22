#pragma once

#include <madrona/render/api.hpp>

#include <imgui.h>

#include "madrona/render/vk/window.hpp"
#include "present.hpp"
#include "viewer_common.hpp"
#include "render_common.hpp"
#include "render_ctx.hpp"

namespace madrona::viz {

struct Frame {
    render::Framebuffer fb;
    render::Framebuffer imguiFB;

    render::ShadowFramebuffer shadowFB;

    VkCommandPool drawCmdPool;
    VkCommandBuffer drawCmd;
    VkCommandPool presentCmdPool;
    VkCommandBuffer presentCmd;

    VkFence cpuFinished;

    VkSemaphore renderFinished;
    VkSemaphore guiRenderFinished;
    VkSemaphore swapchainReady;

    render::vk::HostBuffer viewStaging;
    render::vk::HostBuffer lightStaging;
    // Don't need a shadow view staging because that will be done on the GPU.
    render::vk::HostBuffer skyStaging;

    // Contains everything
    render::vk::LocalBuffer renderInput;

    render::vk::LocalBuffer voxelVBO;
    render::vk::LocalBuffer voxelIndexBuffer;
    render::vk::LocalBuffer voxelData;

    int64_t renderInputSize;

    uint32_t cameraViewOffset;
    // We now store this in a separate buffer
    // uint32_t simViewOffset;
    uint32_t drawCmdOffset;
    uint32_t drawCountOffset;
    // We now store this in a separate buffer
    // uint32_t instanceOffset;
    uint32_t lightOffset;
    uint32_t shadowOffset;
    uint32_t skyOffset;
    uint32_t maxDraws;

    VkDescriptorSet cullShaderSet;
    VkDescriptorSet drawShaderSet;
    VkDescriptorSet lightingSet;
    VkDescriptorSet shadowGenSet;
    VkDescriptorSet shadowBlurSet;

    VkDescriptorSet voxelGenSet;
    VkDescriptorSet voxelDrawSet;

    // Contains a descriptor set for the sampler state and the final rendered output
    VkDescriptorSet batchOutputQuadSet;

    VkDescriptorSet gridDrawSet;
};

struct ImGuiRenderState {
    VkDescriptorPool descPool;
    VkRenderPass renderPass;
};

struct ViewerRendererState {
    render::RenderContext &rctx;
    render::vk::Device &dev;

    const render::vk::RenderWindow *window;
    render::vk::PresentationState present;
    // Fixme remove
    render::vk::QueueState presentWrapper;

    ImGuiRenderState imguiState;
    std::array<VkClearValue, 2> fbImguiClear;

    uint32_t fbWidth;
    uint32_t fbHeight;

    std::array<VkClearValue, 4> fbClear;
    std::array<VkClearValue, 2> fbShadowClear;

    render::Pipeline<1> objectShadowDraw;
    render::Pipeline<1> deferredLighting;
    render::Pipeline<1> shadowGen;
    render::Pipeline<1> blur;
    render::Pipeline<1> voxelMeshGen;
    render::Pipeline<1> voxelDraw;
    render::Pipeline<1> quadDraw;
    // Draw a grid
    render::Pipeline<1> gridDraw;

    uint32_t curFrame;
    HeapArray<Frame> frames;
    uint64_t globalFrameNum;

    render::vk::HostBuffer screenshotBuffer;

    uint32_t currentSwapchainIndex;

    bool renderFlycamFrame(const ViewerControl &viz_ctrl);
    bool renderGridFrame(const ViewerControl &viz_ctrl);

    // Returns true if nothing wrong happened
    // Returns false if resize happened
    bool renderGUIAndPresent(const ViewerControl &viz_ctrl,
                             bool prepare_screenshot);

    void handleResize();

    void recreateSemaphores();

    void destroy();
};

class ViewerRenderer {
public:
    ViewerRenderer(const render::RenderManager &render_mgr,
                   const Window *window);
    ~ViewerRenderer();

    void waitUntilFrameReady();
    void waitForIdle();

    void startFrame();
    void render(const ViewerControl &viewer_ctrl);

    CountT loadObjects(Span<const imp::SourceObject> objs,
                       Span<const imp::SourceMaterial> mats,
                       Span<const imp::SourceTexture> textures,
                       bool override_materials);

    void configureLighting(Span<const render::LightDesc> lights);
    
    inline GLFWwindow * osWindow() const { return state_.window->hdl; }

    bool needResize() const;

    void handleResize();
 
private:
    ViewerRendererState state_;
};

}
