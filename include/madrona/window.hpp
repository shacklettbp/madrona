#pragma once

#include <madrona/render/api.hpp>

namespace madrona {

struct Window {
    uint32_t width;
    uint32_t height;
};

class WindowManager {
public:
    struct Config {
        bool enableRenderAPIValidation = false;
        render::APIBackendSelect renderBackendSelect =
            render::APIBackendSelect::Auto;
    };

    WindowManager(const Config &cfg = defaultConfig());
    WindowManager(WindowManager &&);
    ~WindowManager();

    Window * makeWindow(const char *title,
                        uint32_t width,
                        uint32_t height);

    void destroyWindow(Window *window);

    render::APIManager & gpuAPIManager();

    render::GPUHandle initGPU(
        CountT gpu_idx, Span<const Window * const> windows);

private:
    static inline Config defaultConfig();

    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}

#include "window.inl"
