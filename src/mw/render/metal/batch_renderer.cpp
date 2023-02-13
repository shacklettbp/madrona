#include "../batch_renderer.hpp"

#include <madrona/optional.hpp>

#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>

#include "metal/nscpp/NSEvent.hpp"
#include "metal/nscpp/NSWindow.hpp"
#include "metal/nscpp/NSApplication.hpp"
#include "metal/nscpp/NSRunningApplication.hpp"

#include <fstream>
#include <thread>

namespace madrona::render {

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
    MTL::RenderPassDescriptor *renderPass;
    MTL::Texture *renderTarget;
    Optional<AppDelegate> appDelegate;

    static inline Impl * make(const Config &cfg);
    inline ~Impl();

    inline void render();
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
    CGRect window_frame = {
        { 0, 0 },
        {
            (CGFloat)renderer->cfg.renderWidth,
            (CGFloat)renderer->cfg.renderHeight, 
        },
    };

    auto tmp_pool = NS::AutoreleasePool::alloc()->init();

    window = NS::Window::alloc()->init(
        window_frame,
        NS::WindowStyleMaskResizable |
            NS::WindowStyleMaskClosable |
            NS::WindowStyleMaskTitled,
        NS::BackingStoreBuffered,
        false);

    window->setTitle(nsStrUTF8("Madrona Batch Renderer"));

    layer = CA::MetalLayer::layer();
    layer->setDevice(renderer->dev);
    layer->setPixelFormat(MTL::PixelFormatBGRA8Unorm_sRGB);

    NS::View *view = window->contentView();
    view->setWantsLayer(true);
    view->setLayer(layer);

    window->makeKeyAndOrderFront(nullptr);

    presentPass = MTL::RenderPassDescriptor::renderPassDescriptor()->retain();
    presentAttachment = presentPass->colorAttachments()->object(0);
    presentAttachment->setLoadAction(MTL::LoadActionDontCare);
    presentAttachment->setStoreAction(MTL::StoreActionStore);

    uint16_t fullscreen_tri_stage_indices[3] = {0, 1, 2};
    auto fullscreen_tri_staging = renderer->dev->newBuffer(
        fullscreen_tri_stage_indices,
        sizeof(uint16_t) * 3, MTL::ResourceStorageModeShared);

    fullscreenTri = renderer->dev->newBuffer(sizeof(uint16_t) * 3,
        MTL::ResourceStorageModePrivate |
        MTL::ResourceHazardTrackingModeUntracked);

    auto copy_cmd =
        renderer->cmdQueue->commandBufferWithUnretainedReferences();
    auto copy_enc = copy_cmd->blitCommandEncoder();
    copy_enc->copyFromBuffer(fullscreen_tri_staging, 0, fullscreenTri, 0,
                             sizeof(uint16_t) * 3);
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
    bool need_present = debugPresent();

    auto pool = NS::AutoreleasePool::alloc()->init();
    MTL::Device *dev = MTL::CreateSystemDefaultDevice()->autorelease();
    MTL::CommandQueue *cmd_queue = dev->newCommandQueue()->autorelease();

    auto *target_desc =
        MTL::TextureDescriptor::alloc()->init();
    target_desc->setTextureType(MTL::TextureType2D);
    target_desc->setWidth((NS::UInteger)cfg.renderWidth);
    target_desc->setHeight((NS::UInteger)cfg.renderHeight);
    target_desc->setPixelFormat(MTL::PixelFormatBGRA8Unorm_sRGB);
    target_desc->setStorageMode(MTL::StorageModePrivate);

    MTL::TextureUsage target_usage = MTL::TextureUsageRenderTarget;
    if (need_present) {
        target_usage |= MTL::TextureUsageShaderRead;
    }
    target_desc->setUsage(target_usage);

    MTL::Texture *render_target = dev->newTexture(target_desc)->autorelease();

    target_desc->release();

    auto *render_pass = MTL::RenderPassDescriptor::renderPassDescriptor();
    MTL::RenderPassColorAttachmentDescriptor *attachment =
        render_pass->colorAttachments()->object(0);
    attachment->setClearColor(MTL::ClearColor { 1, 0, 0, 1 });
    attachment->setLoadAction(MTL::LoadActionClear);
    attachment->setStoreAction(MTL::StoreActionStore);
    attachment->setTexture(render_target);

    auto *impl = new Impl {
        .cfg = cfg,
        .appPool = pool,
        .dev = dev,
        .cmdQueue = cmd_queue,
        .renderPass = render_pass,
        .renderTarget = render_target,
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
    if (appDelegate.has_value()) {
        appDelegate->presentPipeline->release();
        appDelegate->presentPass->release();
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

    MTL::CommandBuffer *cmd = cmdQueue->commandBufferWithUnretainedReferences();
    auto enc = cmd->renderCommandEncoder(renderPass);
    enc->endEncoding();
    cmd->commit();

    if (appDelegate.has_value()) {
        auto drawable_pool = NS::AutoreleasePool::alloc()->init();

        MTL::CommandBuffer *present_cmd = cmdQueue->commandBuffer();
        auto *drawable = appDelegate->layer->nextDrawable();

        appDelegate->presentAttachment->setTexture(drawable->texture());
        auto present_enc =
            present_cmd->renderCommandEncoder(appDelegate->presentPass);
        present_enc->setRenderPipelineState(appDelegate->presentPipeline);
        // FIXME: replace with argument buffer?
        present_enc->setFragmentTexture(renderTarget, 0);
        present_enc->drawIndexedPrimitives(
            MTL::PrimitiveTypeTriangle, 3, MTL::IndexTypeUInt16,
            appDelegate->fullscreenTri, 0);
        present_enc->endEncoding();
        present_cmd->presentDrawable(drawable);
        present_cmd->commit();

        drawable_pool->release();
    }
}

BatchRenderer::BatchRenderer(const Config &cfg)
    : impl_(Impl::make(cfg))
{}

BatchRenderer::BatchRenderer(BatchRenderer &&o) = default;
BatchRenderer::~BatchRenderer() = default;

CountT BatchRenderer::loadObjects(Span<const imp::SourceObject> objs)
{
    (void)objs;

    return 0;
}

RendererInterface BatchRenderer::getInterface() const
{
    return {
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
    };
}

uint8_t * BatchRenderer::rgbPtr() const
{
    return nullptr;
}

float * BatchRenderer::depthPtr() const
{
    return nullptr;
}

void BatchRenderer::render()
{
    impl_->render();
}

}
