#pragma once

#include <madrona/render/api.hpp>

namespace madrona {

struct Window {
    uint32_t width;
    uint32_t height;
};

class WindowManager;

class WindowHandle {
public:
    inline Window * get();
    inline ~WindowHandle();

private:
    inline WindowHandle(Window *win, WindowManager &wm);

    Window *win_;
    WindowManager &wm_;

    friend class WindowManager;
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
