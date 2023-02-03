#include "../batch_renderer.hpp"

#include <madrona/optional.hpp>

#include <Metal/Metal.hpp>
#include <AppKit/AppKit.hpp>

namespace madrona {
namespace render {

struct PresentationState {
    NS::Window *window;
};

struct BatchRenderer::Impl {
    MTL::Device *dev;
    Optional<PresentationState> present;

    static inline Impl * make(const Config &cfg);
    inline ~Impl();
};

static PresentationState makePresentationState(
    const BatchRenderer::Config &cfg)
{
    CGRect window_frame = {
        { 0, 0 },
        { (CGFloat)cfg.renderWidth, (CGFloat)cfg.renderHeight, },
    };

    NS::Window *window = NS::Window::alloc()->init(
        window_frame,
        NS::WindowStyleMaskBorderless,
        NS::BackingStoreBuffered,
        false);

    return PresentationState {
        .window = window,
    };
}

static bool debugPresent()
{
    char *debug_present = getenv("MADRONA_RENDER_DEBUG_PRESENT");
    return debug_present && debug_present[0] == '1';
}

BatchRenderer::Impl * BatchRenderer::Impl::make(
    const Config &cfg)
{
    MTL::Device *dev = MTL::CreateSystemDefaultDevice();

    bool need_present = debugPresent();

    auto present_state = Optional<PresentationState>::none();
    if (need_present) {
        present_state = makePresentationState(cfg);
    }

    return new Impl {
        .dev = dev,
        .present = std::move(present_state),
    };
}

BatchRenderer::Impl::~Impl()
{
    if (present.has_value()) {
        present->window->release();
    }

    dev->release();
}

BatchRenderer::BatchRenderer(const Config &cfg)
    : impl_(Impl::make(cfg))
{
}

BatchRenderer::BatchRenderer(BatchRenderer &&o) = default;
BatchRenderer::~BatchRenderer() = default;

CountT BatchRenderer::loadObjects(Span<const imp::SourceObject> objs)
{
    (void)objs;
    FATAL("Unimplemented");
}

RendererInterface BatchRenderer::getInterface() const
{
    FATAL("Unimplemented");
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
}

}
}
