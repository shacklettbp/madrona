#include <madrona/window.hpp>
#include <madrona/crash.hpp>
#include <madrona/heap_array.hpp>
#include <madrona/render/vk/backend.hpp>
#include <madrona/render/vk/window.hpp>

#include <cassert>

#include "../render/vk/utils.hpp"

namespace madrona {

using namespace render;

struct WindowManager::Impl {
    APILibHandle apiLib;
    APIManager gpuAPIManager;

    static inline Impl * init(const Config &cfg);
};

WindowManager::Impl * WindowManager::Impl::init(const Config &cfg)
{
    assert(cfg.renderBackendSelect == APIBackendSelect::Vulkan ||
           cfg.renderBackendSelect == APIBackendSelect::Auto);

    using namespace vk;

    LoaderLib *loader_lib = nullptr;
#ifdef MADRONA_MACOS
    loader_lib = LoaderLib::load();
    glfwInitVulkanLoader((PFN_vkGetInstanceProcAddr)loader_lib->getEntryFn());
#endif

    if (!glfwInit()) {
        FATAL("Failed to initialize GLFW");
    }

    if (loader_lib == nullptr) {
        loader_lib = LoaderLib::external(
            (void (*)())glfwGetInstanceProcAddress(
                VK_NULL_HANDLE, "vkGetInstanceProcAddr"));
    }

    uint32_t count;
    const char **names = glfwGetRequiredInstanceExtensions(&count);

    HeapArray<const char *> instance_exts(count);
    memcpy(instance_exts.data(), names, count * sizeof(const char *));

    APILibHandle lib_hdl(APIBackendSelect::Vulkan, loader_lib);

    APIManager api_mgr(lib_hdl.lib(), {
        .enableValidation = cfg.enableRenderAPIValidation,
        .enablePresent = true,
        .apiExtensions = instance_exts,
    }, APIBackendSelect::Vulkan);

    return new Impl {
        std::move(lib_hdl),
        std::move(api_mgr),
    };
}

static VkSurfaceKHR getWindowSurface(const vk::Backend &backend,
                                     GLFWwindow *window)
{
    VkSurfaceKHR surface;
    REQ_VK(glfwCreateWindowSurface(backend.hdl, window, nullptr, &surface));

    return surface;
}

static GLFWwindow *makeGLFWwindow(const char *title,
                                  uint32_t width,
                                  uint32_t height)
{
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_DECORATED, GLFW_TRUE);

#ifdef MADRONA_MACOS
    width = utils::divideRoundUp(width, 2_u32);
    height = utils::divideRoundUp(height, 2_u32);
#endif

    return glfwCreateWindow(width, height, title, nullptr, nullptr);
}

// This gets called before the vulkan acquire/present failure
static void framebufferSizeCallback(GLFWwindow *window, int width, int height)
{
    vk::RenderWindow *render_win = 
        (vk::RenderWindow *)glfwGetWindowUserPointer(window);

    render_win->width = width;
    render_win->height = height;

    render_win->needResize = true;
}

WindowManager::WindowManager(const Config &cfg)
    : impl_(Impl::init(cfg))
{}

WindowManager::WindowManager(WindowManager &&) = default;
WindowManager::~WindowManager() = default;

WindowHandle::WindowHandle(Window *win, WindowManager *wm)
    : win_(win),
      wm_(wm)
{}

WindowHandle WindowManager::makeWindow(const char *title,
                                       uint32_t width,
                                       uint32_t height)
{
    GLFWwindow *glfw_window =
        makeGLFWwindow(title, width, height);

    vk::Backend &backend = *static_cast<vk::Backend *>(
        impl_->gpuAPIManager.backend());

    VkSurfaceKHR surface = getWindowSurface(backend, glfw_window);

    glfwSetFramebufferSizeCallback(glfw_window, &framebufferSizeCallback);

    vk::RenderWindow *render_window = new vk::RenderWindow {};
    render_window->width = width;
    render_window->height = height;
    render_window->hdl = glfw_window;
    render_window->surface = surface;
    render_window->needResize = false;

    glfwSetWindowUserPointer(glfw_window, render_window);

    return WindowHandle(render_window, this);
}

void WindowManager::destroyWindow(Window *window)
{
    vk::RenderWindow *render_window = static_cast<vk::RenderWindow *>(window);

    vk::Backend &backend = *static_cast<vk::Backend *>(
        impl_->gpuAPIManager.backend());
    backend.dt.destroySurfaceKHR(
        backend.hdl, render_window->surface, nullptr);

    glfwDestroyWindow(render_window->hdl);

    delete window;
}

APIManager & WindowManager::gpuAPIManager()
{
    return impl_->gpuAPIManager;
}

GPUHandle WindowManager::initGPU(
    CountT gpu_idx,
    Span<const Window * const> windows)
{
    vk::Backend &backend =
        *static_cast<vk::Backend *>(impl_->gpuAPIManager.backend());

    HeapArray<VkSurfaceKHR> vk_surfaces(windows.size());
    for (CountT i = 0; i < windows.size(); i++) {
        auto render_window = static_cast<const vk::RenderWindow *>(
            windows[i]);
        vk_surfaces[i] = render_window->surface;
    }

    return GPUHandle(APIBackendSelect::Vulkan,
                     backend.makeDevice(gpu_idx, vk_surfaces));
}

}
