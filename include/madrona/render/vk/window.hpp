#pragma once

#include <GLFW/glfw3.h>
#include <vulkan/vulkan.h>

#include <madrona/window.hpp>

namespace madrona::render::vk {

struct RenderWindow : Window {
    GLFWwindow *hdl;
    VkSurfaceKHR surface;
};

}
