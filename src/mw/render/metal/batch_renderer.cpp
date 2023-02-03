#include "../batch_renderer.hpp"

#include <Metal/Metal.hpp>

namespace madrona {
namespace render {

struct BatchRenderer::Impl {
    MTL::Device *dev;

    static inline Impl * make(const Config &cfg);
    inline ~Impl();
};

BatchRenderer::Impl * BatchRenderer::Impl::make(
    const Config &cfg)
{
    MTL::Device *dev = MTL::CreateSystemDefaultDevice();

    return new Impl {
        .dev = dev,
    };
}

BatchRenderer::Impl::~Impl()
{
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
