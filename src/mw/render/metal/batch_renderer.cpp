#include "../batch_renderer.hpp"

#include <madrona/optional.hpp>

#include <Metal/Metal.hpp>

#include "metal/nscpp/NSEvent.hpp"
#include "metal/nscpp/NSWindow.hpp"
#include "metal/nscpp/NSApplication.hpp"
#include "metal/nscpp/NSRunningApplication.hpp"

#include <thread>

namespace madrona::render {

struct BatchRenderer::Impl {
    struct AppDelegate : public NS::ApplicationDelegate {
        BatchRenderer::Impl *renderer;
        NS::Window *window;

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

    window = NS::Window::alloc()->init(
        window_frame,
        NS::WindowStyleMaskResizable |
            NS::WindowStyleMaskClosable |
            NS::WindowStyleMaskTitled,
        NS::BackingStoreBuffered,
        false);

    window->setTitle(nsStrUTF8("Madrona Batch Renderer"));
    window->makeKeyAndOrderFront(nullptr);

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
    auto pool = NS::AutoreleasePool::alloc()->init();
    MTL::Device *dev = MTL::CreateSystemDefaultDevice();

    auto *impl = new Impl {
        .cfg = cfg,
        .appPool = pool,
        .dev = dev,
        .appDelegate = Optional<AppDelegate>::none(),
    };

    bool need_present = debugPresent();

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
        appDelegate->window->close();
        pumpCocoaEvents(NS::app());
    }

    dev->release();
    appPool->release();
}

void BatchRenderer::Impl::render()
{
    if (appDelegate.has_value()) {
        pumpCocoaEvents(NS::app());
    }

    auto frame_pool = NS::AutoreleasePool::alloc()->init();

    frame_pool->release();
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
