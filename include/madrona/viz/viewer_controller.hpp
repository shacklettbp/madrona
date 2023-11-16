#pragma once

#include <memory>
#include <madrona/importer.hpp>
#include <madrona/viz/common.hpp>

namespace madrona::render {

struct RenderContext;
    
}

namespace madrona::viz {

struct ViewerControllerCfg;
    
// The viewer app simply provides UI overlay over the rendering output
// of the render context and presents the whole rendering output
// to the screen.
struct ViewerController {
    enum class KeyboardKey : uint32_t {
        W, A, S, D, Q, E, R, X, Z, C, G, L, T, F, M,
        K1, K2, K3, K4, K5, K6, K7, K8, K9, K0,
        Shift, Space, NumKeys,
    };

    class UserInput {
    public:
        inline UserInput(bool *keys_state, bool *press_state);

        inline bool keyPressed(KeyboardKey key) const;
        inline bool keyHit(KeyboardKey key) const;

    private:
        bool *keys_state_;
        bool *press_state_;
    };

    struct Impl;
    std::unique_ptr<Impl> impl;

    // Viewer app can also load objects (this would be used if the
    // batch renderer isn't used).
    CountT loadObjects(Span<const imp::SourceObject> objs,
                       Span<const imp::SourceMaterial> mats,
                       Span<const imp::SourceTexture> textures);

    void configureLighting(Span<const render::LightConfig> lights);



    // Run the viewer
    template <typename InputFn, typename StepFn, typename UIFn>
    void loop(InputFn &&input_fn, StepFn &&step_fn, UIFn &&ui_fn);

    void stopLoop();

    ViewerController(ViewerController &&);
    ~ViewerController();

    void setTickRate(uint32_t tick_rate);

private:
    ViewerController(ViewerControllerCfg &cfg);

    void loop(void (*input_fn)(void *, CountT, CountT, const UserInput &),
              void *input_data, void (*step_fn)(void *), void *step_data,
              void (*ui_fn)(void *), void *ui_data);

    friend struct render::RenderContext;
};

}

#include "viewer_controller.inl"
