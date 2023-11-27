#include "../batch_renderer.hpp"
#include "scene.hpp"

#include <madrona/optional.hpp>

#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>

#include "metal/nscpp/NSEvent.hpp"
#include "metal/nscpp/NSWindow.hpp"
#include "metal/nscpp/NSApplication.hpp"
#include "metal/nscpp/NSRunningApplication.hpp"
#include "metal/nscpp/CALayer.hpp"
#include "metal/nscpp/NSScreen.hpp"
#include "metal/nscpp/NSThread.hpp"

#include <thread>

namespace madrona::render {
using namespace metal;

struct BatchRenderer::Impl {
    struct AppDelegate : public NS::ApplicationDelegate {
        BatchRenderer::Impl *renderer;
        NS::Window *window;
        CA::MetalLayer *layer;
        MTL::RenderPassDescriptor *presentPass;
        MTL::RenderPassColorAttachmentDescriptor *presentAttachment;
        MTL::RenderPipelineState *presentPipeline;
        MTL::Buffer *fullscreenTri;

        inline AppDelegate(BatchRenderer::Impl *impl);
        virtual void applicationWillFinishLaunching(
            NS::Notification *notif) override;
        virtual void applicationDidFinishLaunching(
            NS::Notification *notif) override;
        virtual bool applicationShouldTerminateAfterLastWindowClosed(
            NS::Application *sender) override;
    };

    Config cfg;
    NS::AutoreleasePool *appPool;
    MTL::Device *dev;
    MTL::CommandQueue *cmdQueue;
    MTL::RenderPassDescriptor *drawPass;
    MTL::Texture *drawColorTarget;
    MTL::Texture *drawDepthTarget;
    MTL::ComputePipelineState *multiviewSetupPipeline;
    MTL::RenderPipelineState *drawPipeline;
    MTL::IndirectCommandBuffer *drawICB;
    MTL::DepthStencilState *drawDepthTest;
    MTL::Fence *icbSetupFence;
    MTL::Fence *viewSetupFence;
    MTL::Fence *drawFinishedFence;
    MTL::Buffer *engineInteropBuffer;
    MTL::Buffer *renderDataBuffer;
    InstanceData *instanceDataBase;
    HeapArray<PerspectiveCameraData *> viewCamPointers;
    uint32_t *numViewsBase;
    int32_t renderDataArgBufferOffset;
    int32_t renderDataDrawCountOffset;
    MTL::Heap *outHeap;
    MTL::Buffer *rgbOutBuffer;
    MTL::Buffer *depthOutBuffer;
    AssetManager assetMgr;
    DynArray<Assets> assets;
    CacheAlignedU32 numInstancesCounter;
    Optional<AppDelegate> appDelegate;

    static inline Impl * make(const Config &cfg);
    inline ~Impl();

    inline void render();

    inline CountT loadObjects(Span<const imp::SourceObject> objs);
};

static inline NS::String * nsStrUTF8(const char *str)
{
    return NS::String::string(str, NS::StringEncoding::UTF8StringEncoding);
}

static inline void stopEventLoop(NS::Application *app)
{
    NS::Event *event = NS::Event::otherEventWithType(
        NS::EventTypeApplicationDefined,
        CGPoint {0, 0}, 0, 0, 0, nullptr, 0, 0, 0);
    app->postEvent(event, true);
    app->stop(nullptr);
}

struct ObjCMemberFunction {
    const char *name;
    const char *signature;
    IMP callback;
};

static Class makeObjCClass(
    Class parent, const char *name, 
    Span<const ObjCMemberFunction> mem_funcs,
    Span<SEL> out_selectors)
{
    Class objc_type = objc_allocateClassPair(parent, name, 0);

    for (CountT i = 0; i < mem_funcs.size(); i++) {
        const auto &mem_func = mem_funcs[i];

        SEL sel = sel_registerName(mem_func.name);
        class_addMethod(objc_type, sel, mem_func.callback, mem_func.signature);
        out_selectors[i] = sel;
    }

    return objc_type;
}

struct ObjCAllocHelper : NS::Object {
    static inline NS::Object * make(Class type)
    {
        auto ptr = NS::Object::alloc<ObjCAllocHelper>(type);
        return ptr->init<NS::Object>();
    }
};

void setupCocoaMultiThreading()
{
    void (*noop_cb)(NS::Object *, SEL, void *) = 
        [](NS::Object *obj, SEL, void *) { obj->release(); };

    SEL helper_noop;
    Class nsthread_init_helper = makeObjCClass(
        (Class)NS::Private::Class::s_kNSObject, "MadronaNSThreadInitHelper",
        {{ "noop", "v@:@", (IMP)noop_cb }}, { &helper_noop, 1 });

    NS::Object *init_helper = ObjCAllocHelper::make(nsthread_init_helper);
    NS::Thread::detachNewThreadSelector(helper_noop, init_helper, nullptr);
}

BatchRenderer::Impl::AppDelegate::AppDelegate(Impl *impl)
    : renderer(impl),
      window(nullptr)
{
}

#if 0
static inline NS::Menu * makeMenuBar()
{
    NS::Menu *main_menu = NS::Menu::alloc()->init();
    NS::MenuItem *app_menu_item = NS::MenuItem::alloc()->init();
    NS::Menu *app_menu = NS::Menu::alloc()->init(
        nsStrUTF8("Madrona Batch Renderer"));

    printf("Will Launch 2\n");

    NS::String *app_name =
        NS::RunningApplication::currentApplication()->localizedName();
    NS::String *quit_item_name =
        nsStrUTF8("Quit")->stringByAppendingString(app_name);
    SEL quit_cb = NS::MenuItem::registerActionCallback("appQuit",
        [](void *, SEL, const NS::Object *sender) {
            auto *app = NS::Application::sharedApplication();
            app->terminate(sender);
        });

    NS::MenuItem* quit_item =
        app_menu->addItem(quit_item_name, quit_cb, nsStrUTF8("q"));
    quit_item->setKeyEquivalentModifierMask(NS::EventModifierFlagCommand);
    quit_item->setSubmenu(app_menu);

    NS::MenuItem *window_menu_item  = NS::MenuItem::alloc()->init();
    NS::Menu *window_menu = NS::Menu::alloc()->init(nsStrUTF8("Window"));

    SEL close_window_cb = NS::MenuItem::registerActionCallback("windowClose",
        [](void *, SEL, const NS::Object *){
            auto *app = NS::Application::sharedApplication();
            app->windows()->object< NS::Window >(0)->close();
        });

    NS::MenuItem* close_window = window_menu->addItem(
        nsStrUTF8("Close Window"), close_window_cb, nsStrUTF8("w"));
    close_window->setKeyEquivalentModifierMask(NS::EventModifierFlagCommand);

    window_menu_item->setSubmenu(window_menu);

    main_menu->addItem(app_menu_item);
    main_menu->addItem(window_menu_item);

    app_menu_item->release();
    window_menu_item->release();
    app_menu->release();
    window_menu->release();

    return main_menu->autorelease();
}
#endif

void BatchRenderer::Impl::AppDelegate::applicationWillFinishLaunching(
    NS::Notification *)
{}

void BatchRenderer::Impl::AppDelegate::applicationDidFinishLaunching(
    NS::Notification *notif)
{
    auto main_screen = NS::Screen::main();
    float screen_scale = main_screen->backingScaleFactor();

    // Scale this window to be backed by renderWidth / renderHeight pixels not
    // logical coordinates. Note that this doesn't correspond to screen pixels,
    // just backing pixels.
    CGRect window_frame = {
        { 0, 0 },
        {
            (CGFloat)renderer->cfg.renderWidth / screen_scale,
            (CGFloat)renderer->cfg.renderHeight / screen_scale, 
        },
    };

    auto tmp_pool = NS::AutoreleasePool::alloc()->init();

    window = NS::Window::alloc()->init(
        window_frame,
        NS::WindowStyleMaskTitled,
        NS::BackingStoreBuffered,
        false);

    window->setTitle(nsStrUTF8("Madrona Batch Renderer"));

    layer = CA::MetalLayer::layer();
    layer->setDevice(renderer->dev);
    layer->setPixelFormat(MTL::PixelFormatBGRA8Unorm_sRGB);
    NS::View *view = window->contentView();
    view->setLayer(layer);

    {
        // CAMetalLayer from metal-cpp doesn't define the CALayer
        // accessors so have to manually cast to the parent class
        // FIXME: is this the right way to do this?
        CA::Layer *layer_parent = (CA::Layer *)layer;
        layer_parent->setContentsScale(window->backingScaleFactor());
    }

    window->makeKeyAndOrderFront(nullptr);

    presentPass = MTL::RenderPassDescriptor::renderPassDescriptor()->retain();
    presentAttachment = presentPass->colorAttachments()->object(0);
    presentAttachment->setLoadAction(MTL::LoadActionDontCare);
    presentAttachment->setStoreAction(MTL::StoreActionStore);

    // Need to be rounded to 4 bytes on macos
    uint16_t fullscreen_tri_stage_indices[4] = {0, 1, 2, 0xFFFF};

    fullscreenTri = renderer->dev->newBuffer(
        sizeof(uint16_t) * 4,
        MTL::ResourceStorageModePrivate |
        MTL::ResourceHazardTrackingModeUntracked);

    auto fullscreen_tri_staging = renderer->dev->newBuffer(
        fullscreen_tri_stage_indices,
        sizeof(uint16_t) * 4, MTL::ResourceStorageModeShared);

    auto copy_cmd =
        renderer->cmdQueue->commandBufferWithUnretainedReferences();
    auto copy_enc = copy_cmd->blitCommandEncoder();
    copy_enc->copyFromBuffer(fullscreen_tri_staging, 0, fullscreenTri, 0,
                             sizeof(uint16_t) * 4);
    copy_enc->endEncoding();
    copy_cmd->commit();
    copy_cmd->waitUntilCompleted();
    fullscreen_tri_staging->release();

    NS::Error *mtl_err = nullptr;

    auto present_shader_url = NS::URL::fileURLWithPath(
        nsStrUTF8(MADRONA_BATCHRENDERER_MTL_SHADER_DIR "/present.metallib"));

    MTL::Library *present_lib = renderer->dev->newLibrary(
        present_shader_url, &mtl_err)->autorelease();

    if (!present_lib) {
        FATAL("%s\n", mtl_err->localizedDescription()->utf8String());
    }

    MTL::Function *vert_fn =
        present_lib->newFunction(nsStrUTF8("vertMain"))->autorelease();
    MTL::Function *frag_fn =
        present_lib->newFunction(nsStrUTF8("fragMain"))->autorelease();

    auto *pipeline_desc =
        MTL::RenderPipelineDescriptor::alloc()->init()->autorelease();
    pipeline_desc->setVertexFunction(vert_fn);
    pipeline_desc->setFragmentFunction(frag_fn);
    pipeline_desc->colorAttachments()->object(0)->setPixelFormat(
        MTL::PixelFormatBGRA8Unorm_sRGB);

    presentPipeline = renderer->dev->newRenderPipelineState(pipeline_desc, &mtl_err);
    if (!presentPipeline) {
        FATAL("%s\n", mtl_err->localizedDescription()->utf8String());
    }

    tmp_pool->release();

    auto app = (NS::Application *)notif->object();
    app->activateIgnoringOtherApps(true);

    stopEventLoop(app);
}

bool BatchRenderer::Impl::AppDelegate::
    applicationShouldTerminateAfterLastWindowClosed(NS::Application *)
{
    return false;
}

static bool debugPresent()
{
    char *debug_present = getenv("MADRONA_RENDER_DEBUG_PRESENT");
    return debug_present && debug_present[0] == '1';
}

BatchRenderer::Impl * BatchRenderer::Impl::make(
    const Config &cfg)
{
    const bool need_present = debugPresent();
    const int64_t num_worlds = cfg.numWorlds;
    const int64_t max_views_per_world = cfg.maxViewsPerWorld;
    const int64_t max_insts_per_world = cfg.maxInstancesPerWorld;

    const int64_t max_views = num_worlds * max_views_per_world;
    const int64_t max_instances = num_worlds * max_insts_per_world;
    const int64_t max_draws = max_instances * max_views_per_world;

    auto pool = NS::AutoreleasePool::alloc()->init();

    setupCocoaMultiThreading();

    MTL::Device *dev = MTL::CreateSystemDefaultDevice()->autorelease();
    MTL::CommandQueue *cmd_queue = dev->newCommandQueue()->autorelease();

    auto *color_target_desc =
        MTL::TextureDescriptor::alloc()->init();
    color_target_desc->setTextureType(MTL::TextureType2DArray);
    color_target_desc->setArrayLength(max_views);
    color_target_desc->setWidth((NS::UInteger)cfg.renderWidth);
    color_target_desc->setHeight((NS::UInteger)cfg.renderHeight);
    color_target_desc->setPixelFormat(MTL::PixelFormatBGRA8Unorm_sRGB);
    color_target_desc->setStorageMode(MTL::StorageModePrivate);
    color_target_desc->setHazardTrackingMode(
        MTL::HazardTrackingModeUntracked);

    MTL::TextureUsage color_target_usage = MTL::TextureUsageRenderTarget;
    if (need_present) {
        color_target_usage |= MTL::TextureUsageShaderRead;
    }
    color_target_desc->setUsage(color_target_usage);

    MTL::Texture *color_target =
        dev->newTexture(color_target_desc)->autorelease();
    color_target_desc->release();

    auto *depth_target_desc =
        MTL::TextureDescriptor::alloc()->init();
    depth_target_desc->setTextureType(MTL::TextureType2DArray);
    depth_target_desc->setArrayLength(max_views);
    depth_target_desc->setWidth((NS::UInteger)cfg.renderWidth);
    depth_target_desc->setHeight((NS::UInteger)cfg.renderHeight);
    depth_target_desc->setPixelFormat(MTL::PixelFormatDepth32Float);
    depth_target_desc->setStorageMode(MTL::StorageModePrivate);
    depth_target_desc->setHazardTrackingMode(
        MTL::HazardTrackingModeUntracked);
    depth_target_desc->setUsage(MTL::TextureUsageRenderTarget);

    MTL::Texture *depth_target =
        dev->newTexture(depth_target_desc)->autorelease();
    depth_target_desc->release();

    auto *render_pass = MTL::RenderPassDescriptor::renderPassDescriptor();
    render_pass->setRenderTargetWidth(cfg.renderWidth);
    render_pass->setRenderTargetHeight(cfg.renderHeight);
    render_pass->setRenderTargetArrayLength(max_views);
    MTL::RenderPassColorAttachmentDescriptor *color_attachment =
        render_pass->colorAttachments()->object(0);
    color_attachment->setClearColor(MTL::ClearColor { 0, 0, 0, 1 });
    color_attachment->setLoadAction(MTL::LoadActionClear);
    color_attachment->setStoreAction(MTL::StoreActionStore);
    color_attachment->setTexture(color_target);

    MTL::RenderPassDepthAttachmentDescriptor *depth_attachment =
        render_pass->depthAttachment();
    depth_attachment->setClearDepth(0.0);
    depth_attachment->setLoadAction(MTL::LoadActionClear);
    depth_attachment->setStoreAction(MTL::StoreActionStore);
    depth_attachment->setTexture(depth_target);

    NS::AutoreleasePool *init_pool = NS::AutoreleasePool::alloc()->init();
    NS::Error *pipeline_err = nullptr;

    auto draw_shader_url = NS::URL::fileURLWithPath(
        nsStrUTF8(MADRONA_BATCHRENDERER_MTL_SHADER_DIR "/draw.metallib"));

    MTL::Library *draw_lib = dev->newLibrary(
        draw_shader_url, &pipeline_err)->autorelease();

    if (!draw_lib) {
        FATAL("%s\n", pipeline_err->localizedDescription()->utf8String());
    }

    MTL::Function *multiview_setup_fn =
        draw_lib->newFunction(nsStrUTF8("setupMultiview"))->autorelease();

    auto *multiview_setup_pipeline_desc =
        MTL::ComputePipelineDescriptor::alloc()->init()->autorelease();
    multiview_setup_pipeline_desc->setComputeFunction(multiview_setup_fn);
    multiview_setup_pipeline_desc->setMaxTotalThreadsPerThreadgroup(
        shader::consts::threadsPerInstance);
    multiview_setup_pipeline_desc->buffers()->object(0)->setMutability(
        MTL::MutabilityImmutable);
    multiview_setup_pipeline_desc->buffers()->object(1)->setMutability(
        MTL::MutabilityImmutable);
    multiview_setup_pipeline_desc->buffers()->object(2)->setMutability(
        MTL::MutabilityImmutable);

    MTL::ComputePipelineState *multiview_setup_pipeline =
        dev->newComputePipelineState(multiview_setup_pipeline_desc,
                                     MTL::PipelineOptionNone,
                                     nullptr, &pipeline_err);

    if (!multiview_setup_pipeline) {
        FATAL("%s\n", pipeline_err->localizedDescription()->utf8String());
    }

    MTL::Function *vert_fn =
        draw_lib->newFunction(nsStrUTF8("vertMain"))->autorelease();
    MTL::Function *frag_fn =
        draw_lib->newFunction(nsStrUTF8("fragMain"))->autorelease();

    auto *draw_pipeline_desc =
        MTL::RenderPipelineDescriptor::alloc()->init()->autorelease();
    draw_pipeline_desc->setSupportIndirectCommandBuffers(true);
    draw_pipeline_desc->setVertexFunction(vert_fn);
    draw_pipeline_desc->setFragmentFunction(frag_fn);
    draw_pipeline_desc->setInputPrimitiveTopology(
        MTL::PrimitiveTopologyClassTriangle);
    draw_pipeline_desc->colorAttachments()->object(0)->setPixelFormat(
        MTL::PixelFormatBGRA8Unorm_sRGB);
    draw_pipeline_desc->setDepthAttachmentPixelFormat(
        MTL::PixelFormat::PixelFormatDepth32Float);
    draw_pipeline_desc->vertexBuffers()->object(0)->setMutability(
        MTL::MutabilityImmutable);
    draw_pipeline_desc->vertexBuffers()->object(1)->setMutability(
        MTL::MutabilityImmutable);
    draw_pipeline_desc->fragmentBuffers()->object(0)->setMutability(
        MTL::MutabilityImmutable);
    draw_pipeline_desc->fragmentBuffers()->object(1)->setMutability(
        MTL::MutabilityImmutable);

    MTL::RenderPipelineState *draw_pipeline = dev->newRenderPipelineState(
        draw_pipeline_desc, &pipeline_err);
    if (!draw_pipeline) {
        FATAL("%s\n", pipeline_err->localizedDescription()->utf8String());
    }

    auto *draw_icb_desc =
        MTL::IndirectCommandBufferDescriptor::alloc()->init()->autorelease();
    draw_icb_desc->setCommandTypes(MTL::IndirectCommandTypeDrawIndexed);
    draw_icb_desc->setInheritBuffers(true);
    draw_icb_desc->setMaxVertexBufferBindCount(2);
    draw_icb_desc->setMaxFragmentBufferBindCount(2);
    draw_icb_desc->setInheritPipelineState(true);

    MTL::IndirectCommandBuffer *draw_icb = dev->newIndirectCommandBuffer(
        draw_icb_desc, max_draws, MTL::ResourceStorageModePrivate |
        MTL::ResourceHazardTrackingModeUntracked);
    assert(draw_icb);

    auto *draw_depth_test_desc =
        MTL::DepthStencilDescriptor::alloc()->init()->autorelease();
    draw_depth_test_desc->setDepthWriteEnabled(true);
    draw_depth_test_desc->setDepthCompareFunction(
        MTL::CompareFunctionGreaterEqual);

    MTL::DepthStencilState *draw_depth_test =
        dev->newDepthStencilState(draw_depth_test_desc);

    int64_t engine_interop_offsets[2];
    int64_t num_engine_interop_bytes = utils::computeBufferOffsets({
            (int64_t)sizeof(InstanceData) * max_instances,
            (int64_t)sizeof(PerspectiveCameraData) * max_views,
            (int64_t)sizeof(uint32_t) * num_worlds,
        }, engine_interop_offsets, consts::mtlBufferAlignment);
     
    MTL::Buffer *engine_interop_buf = dev->newBuffer(
        num_engine_interop_bytes, MTL::ResourceStorageModeShared |
        MTL::ResourceHazardTrackingModeUntracked);

    int64_t render_data_offsets[3];
    int64_t num_render_data_bytes = utils::computeBufferOffsets({
            sizeof(DrawICBArgBuffer),
            sizeof(RenderArgBuffer),
            (int64_t)sizeof(DrawInstanceData) * max_draws,
            (int64_t)sizeof(MTL::IndirectCommandBufferExecutionRange),
        }, render_data_offsets, consts::mtlBufferAlignment);

    MTL::Buffer *render_data_buf = dev->newBuffer(
        num_render_data_bytes, MTL::ResourceStorageModePrivate |
        MTL::ResourceHazardTrackingModeUntracked);

    {
        uint64_t render_argbuffer_gpu_addr = render_data_buf->gpuAddress();
        uint64_t engine_interop_gpu_addr = engine_interop_buf->gpuAddress();

        uint64_t num_staging_bytes =
            render_data_offsets[0] + sizeof(RenderArgBuffer);

        MTL::Buffer *render_argbuffers_staging = dev->newBuffer(
            num_staging_bytes,
            MTL::ResourceStorageModeShared |
            MTL::ResourceHazardTrackingModeUntracked |
            MTL::ResourceCPUCacheModeWriteCombined)->autorelease();

        auto *staging_argbuffers_base = 
            (char *)render_argbuffers_staging->contents();

        auto *icb_argbuffer_staging = (DrawICBArgBuffer *)(
            staging_argbuffers_base);

        icb_argbuffer_staging->hdl = draw_icb->gpuResourceID();

        auto *render_argbuffer_staging = (RenderArgBuffer *)(
                staging_argbuffers_base + render_data_offsets[0]);

        render_argbuffer_staging->drawInstances =
            render_argbuffer_gpu_addr + render_data_offsets[1];
        render_argbuffer_staging->numDraws =
            render_argbuffer_gpu_addr + render_data_offsets[2] + 4;

        render_argbuffer_staging->engineInstances = engine_interop_gpu_addr;
        render_argbuffer_staging->views =
            engine_interop_gpu_addr + engine_interop_offsets[0];
        render_argbuffer_staging->numViews =
            engine_interop_gpu_addr + engine_interop_offsets[1];

        render_argbuffer_staging->numMaxViewsPerWorld = cfg.maxViewsPerWorld;

        MTL::CommandBuffer *render_args_setup_cmd =
            cmd_queue->commandBufferWithUnretainedReferences();
        MTL::BlitCommandEncoder *blit_enc =
            render_args_setup_cmd->blitCommandEncoder();
        // Setup argbuffer
        blit_enc->copyFromBuffer(render_argbuffers_staging, 0,
            render_data_buf, 0, num_staging_bytes);
        blit_enc->endEncoding();
        render_args_setup_cmd->commit();
        render_args_setup_cmd->waitUntilCompleted();
    }

    uint64_t num_pixels = uint64_t(max_views) * uint64_t(cfg.renderWidth) *
        uint64_t(cfg.renderHeight);
    MTL::Buffer *rgb_out_buffer = dev->newBuffer(NS::UInteger(
         num_pixels * sizeof(uint8_t) * 4),
        MTL::ResourceStorageModeShared |
            MTL::ResourceHazardTrackingModeUntracked);

    MTL::Buffer *depth_out_buffer = dev->newBuffer(NS::UInteger(
            num_pixels * sizeof(float)),
        MTL::ResourceStorageModeShared |
            MTL::ResourceHazardTrackingModeUntracked);

    init_pool->release();

    // Autorelease in the outer pool
    multiview_setup_pipeline->autorelease();
    draw_pipeline->autorelease();
    draw_icb->autorelease();
    draw_depth_test->autorelease();
    engine_interop_buf->autorelease();
    render_data_buf->autorelease();
    rgb_out_buffer->autorelease();
    depth_out_buffer->autorelease();

    char *engine_interop_base_ptr = (char *)engine_interop_buf->contents();
    HeapArray<PerspectiveCameraData *> view_cam_ptrs(cfg.numWorlds);
    PerspectiveCameraData *base_cam_ptr = (PerspectiveCameraData *)(
        engine_interop_base_ptr + engine_interop_offsets[0]);
    for (int32_t i = 0; i < (int32_t)cfg.numWorlds; i++) {
        view_cam_ptrs[i] = base_cam_ptr + i * cfg.maxViewsPerWorld;
    }

    AssetManager asset_mgr(dev);
    asset_mgr.transferQueue->autorelease();

    auto *impl = new Impl {
        .cfg = cfg,
        .appPool = pool,
        .dev = dev,
        .cmdQueue = cmd_queue,
        .drawPass = render_pass,
        .drawColorTarget = color_target,
        .drawDepthTarget = depth_target,
        .multiviewSetupPipeline = multiview_setup_pipeline,
        .drawPipeline = draw_pipeline,
        .drawICB = draw_icb,
        .drawDepthTest = draw_depth_test,
        .icbSetupFence = dev->newFence()->autorelease(),
        .viewSetupFence = dev->newFence()->autorelease(),
        .drawFinishedFence = dev->newFence()->autorelease(),
        .engineInteropBuffer = engine_interop_buf,
        .renderDataBuffer = render_data_buf,
        .instanceDataBase = (InstanceData *)(engine_interop_base_ptr),
        .viewCamPointers = std::move(view_cam_ptrs),
        .numViewsBase = (uint32_t *)(
            engine_interop_base_ptr + engine_interop_offsets[1]),
        .renderDataArgBufferOffset = int32_t(render_data_offsets[0]),
        .renderDataDrawCountOffset = int32_t(render_data_offsets[2]),
        .rgbOutBuffer = rgb_out_buffer,
        .depthOutBuffer = depth_out_buffer,
        .assetMgr = asset_mgr,
        .assets = DynArray<Assets>(1),
        .numInstancesCounter = { 0 },
        .appDelegate = Optional<AppDelegate>::none(),
    };

    if (need_present) {
        impl->appDelegate.emplace(impl);

        auto *app = NS::Application::sharedApplication();
        app->setDelegate(&*impl->appDelegate);

        if (!NS::RunningApplication::currentApplication()->
                isFinishedLaunching()) {
            app->run();
        }

        // Need to autorelease these here so they the top level pool releases
        // them, not the event handler pool in app->run()
        impl->appDelegate->fullscreenTri->autorelease();
        impl->appDelegate->presentPipeline->autorelease();
        impl->appDelegate->presentPass->autorelease();
    }
    
    return impl;
}

static inline void pumpCocoaEvents(NS::Application *app)
{
    while (true) {
        auto event_pool = NS::AutoreleasePool::alloc()->init();

        NS::Event *event = app->nextEventMatchingMask(NS::EventMaskAny,
                                                      NS::DateOverride::distantPast(),
                                                      NS::DefaultRunLoopMode(),
                                                      true);
        if (event == nullptr) {
            break;
        }

        app->sendEvent(event);

        event_pool->release();
    }
}

BatchRenderer::Impl::~Impl()
{
    for (Assets &asset : assets) {
        assetMgr.free(asset);
    }

    if (appDelegate.has_value()) {
        appDelegate->window->close();
        pumpCocoaEvents(NS::app());
    }

    appPool->release();
}

void BatchRenderer::Impl::render()
{
    if (appDelegate.has_value()) {
        pumpCocoaEvents(NS::app());
    }

    auto frame_render_pool = NS::AutoreleasePool::alloc()->init();

    MTL::CommandBuffer *cmd =
        cmdQueue->commandBufferWithUnretainedReferences();

    uint32_t num_instances = numInstancesCounter.v;
    numInstancesCounter.v = 0;

    auto icb_setup_enc = cmd->blitCommandEncoder();
    icb_setup_enc->fillBuffer(renderDataBuffer,
        {NS::UInteger(renderDataDrawCountOffset) + 4, 4}, 0);
    icb_setup_enc->updateFence(icbSetupFence);
    icb_setup_enc->endEncoding();

    auto multiview_setup_enc = cmd->computeCommandEncoder();
    multiview_setup_enc->waitForFence(icbSetupFence);
    multiview_setup_enc->setComputePipelineState(multiviewSetupPipeline);
    multiview_setup_enc->setBuffer(renderDataBuffer, 0, 0);
    multiview_setup_enc->setBuffer(renderDataBuffer,
                                   renderDataArgBufferOffset, 1);
    multiview_setup_enc->setBuffer(assets[0].buffer, 0, 2);
    multiview_setup_enc->setThreadgroupMemoryLength(16, 0);
    multiview_setup_enc->useHeap(assets[0].heap);
    multiview_setup_enc->useResource(engineInteropBuffer,
                                     MTL::ResourceUsageRead);
    multiview_setup_enc->useResource(drawICB,
                                     MTL::ResourceUsageWrite);

    multiview_setup_enc->dispatchThreadgroups(
        {num_instances, 1, 1},
        {shader::consts::threadsPerInstance, 1, 1});

    multiview_setup_enc->updateFence(viewSetupFence);
    multiview_setup_enc->endEncoding();

    auto draw_enc = cmd->renderCommandEncoder(drawPass);
    draw_enc->waitForFence(viewSetupFence, MTL::RenderStageVertex);
    draw_enc->setDepthStencilState(drawDepthTest);
    draw_enc->setRenderPipelineState(drawPipeline);
    draw_enc->setCullMode(MTL::CullModeFront);
    draw_enc->setVertexBuffer(renderDataBuffer,
                              renderDataArgBufferOffset, 0);
    draw_enc->setVertexBuffer(assets[0].buffer, 0, 1);
    draw_enc->useHeap(assets[0].heap, MTL::RenderStageVertex);

    draw_enc->executeCommandsInBuffer(drawICB, renderDataBuffer,
                                      renderDataDrawCountOffset);
    draw_enc->updateFence(
        drawFinishedFence, MTL::RenderStageFragment);

    draw_enc->endEncoding();

    auto copy_out_enc = cmd->blitCommandEncoder();
    copy_out_enc->waitForFence(drawFinishedFence);
    for (CountT world_idx = 0; world_idx < cfg.numWorlds; world_idx++) {
        CountT num_world_views = numViewsBase[world_idx];
        for (CountT world_view_idx = 0; world_view_idx < num_world_views;
             world_view_idx++) {
            CountT view_idx = world_idx * cfg.maxViewsPerWorld + world_view_idx;

            copy_out_enc->copyFromTexture(drawColorTarget, view_idx, 0,
                MTL::Origin(0, 0, 0),
                MTL::Size(cfg.renderWidth, cfg.renderHeight, 1),
                rgbOutBuffer, view_idx * cfg.renderWidth *
                    cfg.renderHeight * sizeof(uint8_t) * 4,
                sizeof(uint8_t) * 4 * cfg.renderWidth, 0);
            copy_out_enc->copyFromTexture(drawDepthTarget, view_idx, 0,
                MTL::Origin(0, 0, 0),
                MTL::Size(cfg.renderWidth, cfg.renderHeight, 1),
                depthOutBuffer, view_idx * cfg.renderWidth *
                    cfg.renderHeight * sizeof(float),
                sizeof(float) * cfg.renderWidth, 0);
        }
    }
    copy_out_enc->endEncoding();

    cmd->commit();

    if (appDelegate.has_value()) {
        MTL::CommandBuffer *present_cmd = cmdQueue->commandBuffer();
        auto *drawable = appDelegate->layer->nextDrawable();

        appDelegate->presentAttachment->setTexture(drawable->texture());
        auto present_enc =
            present_cmd->renderCommandEncoder(appDelegate->presentPass);
        present_enc->setRenderPipelineState(appDelegate->presentPipeline);
        present_enc->setFragmentTexture(drawColorTarget, 0);
        present_enc->waitForFence(
            drawFinishedFence, MTL::RenderStageFragment);
        present_enc->drawIndexedPrimitives(
            MTL::PrimitiveTypeTriangle, 3, MTL::IndexTypeUInt16,
            appDelegate->fullscreenTri, 0);
        present_enc->endEncoding();
        present_cmd->presentDrawable(drawable);
        present_cmd->commit();
    }

    cmd->waitUntilCompleted();
    frame_render_pool->release();
}

CountT BatchRenderer::Impl::loadObjects(Span<const imp::SourceObject> objs)
{

    auto metadata = *assetMgr.prepareSourceAssets(objs);
    StagedAssets staged = assetMgr.stageSourceAssets(dev, metadata, objs);

    Assets loaded_assets = assetMgr.load(dev, metadata, staged);
    assets.emplace_back(std::move(loaded_assets));

    return 0;
}

BatchRenderer::BatchRenderer(const Config &cfg)
    : impl_(Impl::make(cfg))
{}

BatchRenderer::BatchRenderer(BatchRenderer &&o) = default;
BatchRenderer::~BatchRenderer() = default;

CountT BatchRenderer::loadObjects(Span<const imp::SourceObject> objs)
{
    return impl_->loadObjects(objs);
}

RendererInterface BatchRenderer::getInterface() const
{
    return {
        impl_->viewCamPointers.data(),
        impl_->numViewsBase,
        impl_->instanceDataBase,
        &impl_->numInstancesCounter.v,
        int32_t(impl_->cfg.renderWidth),
        int32_t(impl_->cfg.renderHeight),
    };
}

uint8_t * BatchRenderer::rgbPtr() const
{
    return (uint8_t *)impl_->rgbOutBuffer->contents();
}

float * BatchRenderer::depthPtr() const
{
    return (float *)impl_->depthOutBuffer->contents();
}

void BatchRenderer::render()
{
    impl_->render();
}

}
