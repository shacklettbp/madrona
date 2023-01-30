#include "../batch_renderer.hpp"

namespace madrona {
namespace render {

struct BatchRenderer::Impl {
};

BatchRenderer::BatchRenderer(const Config &cfg)
    : impl_(nullptr)
{
    (void)cfg;
    FATAL("Unimplemented");
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
