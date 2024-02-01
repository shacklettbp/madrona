#pragma once

#include <madrona/render/api.hpp>
#include <memory>

namespace madrona {

struct Window {
    uint32_t width;
    uint32_t height;
    bool needResize;
};

class WindowManager;

class WindowHandle {
public:
    inline WindowHandle(Window *win, WindowManager *wm);

    inline WindowHandle(WindowHandle &&o);
    inline ~WindowHandle();

    inline Window * get();

private:
    Window *win_;
    WindowManager *wm_;
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

    WindowHandle makeWindow(const char *title,
                            uint32_t width,
                            uint32_t height);

    render::APIManager & gpuAPIManager();

    render::GPUHandle initGPU(
        CountT gpu_idx, Span<const Window * const> windows);

private:
    static inline Config defaultConfig();

    void destroyWindow(Window *window);

    struct Impl;
    std::unique_ptr<Impl> impl_;

    friend class WindowHandle;
};

}

#include "window.inl"
