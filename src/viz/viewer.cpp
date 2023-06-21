#include <madrona/viz/viewer.hpp>

#include "viewer_renderer.hpp"

#include <imgui.h>

#include <cstdio>
#include <cstdlib>
#include <thread>

using namespace std;

namespace madrona::viz {

namespace InternalConfig {
inline constexpr float cameraMoveSpeed = 5.f;
inline constexpr float mouseSpeed = 2e-4f;

inline constexpr auto nsPerFrame = chrono::nanoseconds(8333333);
inline constexpr auto nsPerFrameLongWait =
    chrono::nanoseconds(7000000);
inline constexpr float secondsPerFrame =
    chrono::duration<float>(nsPerFrame).count();
}

struct SceneProperties {
    math::AABB aabb;
    uint32_t totalTriangles; // Post transform
};

struct Viewer::Impl {
    ViewerCam cam;
    Renderer::FrameConfig frameCfg;
    Renderer renderer;
    uint32_t maxNumAgents;

    Impl(const Viewer::Config &cfg);

    void startFrame();
    void render(float frame_duration);
};

CountT Viewer::loadObjects(Span<const imp::SourceObject> objs)
{
    return impl_->renderer.loadObjects(objs);
}

const render::RendererBridge * Viewer::rendererBridge() const
{
    return &impl_->renderer.getBridgeRef();
}

static void handleCamera(GLFWwindow *window, ViewerCam &cam)
{
    auto keyPressed = [&](uint32_t key) {
        return glfwGetKey(window, key) == GLFW_PRESS;
    };

    math::Vector3 translate = math::Vector3::zero();

    auto cursorPosition = [window]() {
        double mouse_x, mouse_y;
        glfwGetCursorPos(window, &mouse_x, &mouse_y);

        return math::Vector2 { float(mouse_x), float(-mouse_y) };
    };


    if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_2) == GLFW_PRESS) {
        glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

        math::Vector2 mouse_cur = cursorPosition();
        math::Vector2 mouse_delta = mouse_cur - cam.mousePrev;

        auto around_right = math::Quat::angleAxis(
            mouse_delta.y * InternalConfig::mouseSpeed, cam.right);

        auto around_up = math::Quat::angleAxis(
            -mouse_delta.x * InternalConfig::mouseSpeed, math::up);

        auto rotation = (around_up * around_right).normalize();

        cam.up = rotation.rotateVec(cam.up);
        cam.view = rotation.rotateVec(cam.view);
        cam.right = rotation.rotateVec(cam.right);

        if (keyPressed(GLFW_KEY_W)) {
            translate += cam.view;
        }

        if (keyPressed(GLFW_KEY_A)) {
            translate -= cam.right;
        }

        if (keyPressed(GLFW_KEY_S)) {
            translate -= cam.view;
        }

        if (keyPressed(GLFW_KEY_D)) {
            translate += cam.right;
        }

        cam.mousePrev = mouse_cur;
    } else {
        if (keyPressed(GLFW_KEY_W)) {
            translate += cam.up;
        }

        if (keyPressed(GLFW_KEY_A)) {
            translate -= cam.right;
        }

        if (keyPressed(GLFW_KEY_S)) {
            translate -= cam.up;
        }

        if (keyPressed(GLFW_KEY_D)) {
            translate += cam.right;
        }

        cam.mousePrev = cursorPosition();
    }

    if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_2) == GLFW_RELEASE) {
        glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
    }

    cam.position += translate * InternalConfig::cameraMoveSpeed *
        InternalConfig::secondsPerFrame;
}

static float throttleFPS(chrono::time_point<chrono::steady_clock> start) {
    using namespace chrono;
    using namespace chrono_literals;
    
    auto end = steady_clock::now();
    while (end - start <
           InternalConfig::nsPerFrameLongWait) {
        this_thread::sleep_for(1ms);
    
        end = steady_clock::now();
    }
    
    while (end - start < InternalConfig::nsPerFrame) {
        this_thread::yield();
    
        end = steady_clock::now();
    }

    return duration<float>(end - start).count();
}

void Viewer::Impl::startFrame()
{
    renderer.waitUntilFrameReady();

    glfwPollEvents();

    renderer.startFrame();
    ImGui::NewFrame();
}

static void renderCFGUI(Renderer::FrameConfig &cfg,
                        ViewerCam &cam,
                        uint32_t num_agents)
{
    (void)cfg;

    ImGui::Begin("Controls");

    {
        const char *agent_id_opts[] = {
            "None",
            "0",
        };

        int selected_id_opt = cfg.viewIDX;
        if (ImGui::Combo("Control Agent:", &selected_id_opt, agent_id_opts, 2)) {
            cfg.viewIDX = selected_id_opt;
        }
    }

    ImGui::TextUnformatted("Free Camera");
    ImGui::Separator();

    auto side_size = ImGui::CalcTextSize(" Bottom " );
    side_size.y *= 1.4f;
    ImGui::PushStyleVar(ImGuiStyleVar_ButtonTextAlign,
                        ImVec2(0.5f, 0.f));

    if (ImGui::Button("Top", side_size)) {
        cam.position = 10.f * math::up;
        cam.view = -math::up;
        cam.up = -math::fwd;
        cam.right = cross(cam.view, cam.up);
    }

    ImGui::SameLine();

    if (ImGui::Button("Left", side_size)) {
        cam.position = -10.f * math::right;
        cam.view = math::right;
        cam.up = math::up;
        cam.right = cross(cam.view, cam.up);
    }

    ImGui::SameLine();

    if (ImGui::Button("Right", side_size)) {
        cam.position = 10.f * math::right;
        cam.view = -math::right;
        cam.up = math::up;
        cam.right = cross(cam.view, cam.up);
    }

    ImGui::SameLine();

    if (ImGui::Button("Bottom", side_size)) {
        cam.position = -10.f * math::up;
        cam.view = math::up;
        cam.up = math::fwd;
        cam.right = cross(cam.view, cam.up);
    }

    ImGui::PopStyleVar();

#if 0
    auto ortho_size = ImGui::CalcTextSize(" Orthographic ");
    ImGui::PushStyleVar(ImGuiStyleVar_SelectableTextAlign,
                        ImVec2(0.5f, 0.f));
    if (ImGui::Selectable("Perspective", cam.perspective, 0,
                          ortho_size)) {
        cam.perspective = true;
    }
    ImGui::SameLine();

    if (ImGui::Selectable("Orthographic", !cam.perspective, 0,
                          ortho_size)) {
        cam.perspective = false;
    }

    ImGui::SameLine();

    ImGui::PopStyleVar();

    ImGui::TextUnformatted("Projection");
#endif

    float digit_width = ImGui::CalcTextSize("0").x;
    ImGui::SetNextItemWidth(digit_width * 6);
    if (cam.perspective) {
        ImGui::DragFloat("FOV", &cam.fov, 1.f, 1.f, 179.f, "%.0f");
    } else {
        ImGui::DragFloat("View Size", &cam.orthoHeight,
                          0.5f, 0.f, 100.f, "%0.1f");
    }

    ImGui::End();
}

static void fpsCounterUI(float frame_duration)
{
    auto viewport = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(ImVec2(viewport->WorkSize.x, 0.f),
                            0, ImVec2(1.f, 0.f));
    ImGui::PushStyleVar(ImGuiStyleVar_Alpha, 0.5f);
    ImGui::Begin("FPS Counter", nullptr,
                 ImGuiWindowFlags_NoMove |
                 ImGuiWindowFlags_NoInputs |
                 ImGuiWindowFlags_NoTitleBar |
                 ImGuiWindowFlags_AlwaysAutoResize);
    ImGui::PopStyleVar();
    ImGui::Text("%.3f ms per frame (%.1f FPS)",
                1000.f * frame_duration, 1.f / frame_duration);

    ImGui::End();
}

static ViewerCam initDefaultCam()
{
    ViewerCam default_cam;
    default_cam.position = math::Vector3 { 0, 0, 10 };
    default_cam.view = math::Vector3 { 0, 0, -1 };
    default_cam.up = math::Vector3 { 0, 1, 0 };
    default_cam.right = cross(default_cam.view, default_cam.up);

    return default_cam;
}

Viewer::Impl::Impl(const Config &cfg)
    : cam(initDefaultCam()),
      frameCfg {
          .worldIDX = 0,
          .viewIDX = 0,
      },
      renderer(cfg.gpuID,
               cfg.renderWidth,
               cfg.renderHeight,
               cfg.numWorlds,
               cfg.maxViewsPerWorld,
               cfg.maxInstancesPerWorld),
      maxNumAgents(cfg.maxViewsPerWorld)
{}

void Viewer::Impl::render(float frame_duration)
{
    // FIXME: pass actual active agents, not max
    renderCFGUI(frameCfg, cam, maxNumAgents);

    fpsCounterUI(frame_duration);

    ImGui::Render();

    renderer.render(cam, frameCfg);
}

Viewer::Viewer(const Config &cfg)
    : impl_(new Impl(cfg))
{}

Viewer::Viewer(Viewer &&o) = default;
Viewer::~Viewer() = default;

void Viewer::loop(void (*step_fn)(void *), void *data)
{
    auto window = impl_->renderer.window.platformWindow;

    float frame_duration = InternalConfig::secondsPerFrame;
    while (!glfwWindowShouldClose(window)) {
        auto start_time = chrono::steady_clock::now();

        step_fn(data);

        impl_->startFrame();

        handleCamera(window, impl_->cam);
        impl_->render(frame_duration);

        frame_duration = throttleFPS(start_time);
    }

    impl_->renderer.waitForIdle();
}

}
