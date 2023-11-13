#pragma once

#include <memory>
#include <madrona/importer.hpp>
#include <madrona/viz/common.hpp>

namespace madrona::render {

struct RenderContext;
    
}

namespace madrona::viz {

struct ViewerAppCfg;
    
// The viewer app simply provides UI overlay over the rendering output
// of the render context and presents the whole rendering output
// to the screen.
struct ViewerApp {
    struct Impl;
    std::unique_ptr<Impl> impl;

    // Viewer app can also load objects (this would be used if the
    // batch renderer isn't used).
    CountT loadObjects(Span<const imp::SourceObject> objs,
                       Span<const imp::SourceMaterial> mats,
                       Span<const imp::SourceTexture> textures);

    void configureLighting(Span<const render::LightConfig> lights);

private:
    ViewerApp(ViewerAppCfg &cfg);

    friend struct render::RenderContext;
};

}
