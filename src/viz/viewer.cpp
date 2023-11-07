#include <madrona/viz/viewer.hpp>
#include <madrona/stack_alloc.hpp>
#include <madrona/utils.hpp>

#include "viewer_renderer.hpp"

#include <imgui.h>

#include <cstdio>
#include <cstdlib>
#include <thread>

using namespace std;

namespace madrona::viz {

namespace InternalConfig {
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
    uint32_t numWorlds;
    uint32_t maxNumAgents;
    int32_t simTickRate;
    float cameraMoveSpeed;
    bool shouldExit;
    VoxelConfig voxelConfig;

    Impl(const Viewer::Config &cfg);

    inline void startFrame();
    inline void render(float frame_duration);

    inline void loop(
        void (*input_fn)(void *, CountT, CountT, const UserInput &),
        void *input_data, void (*step_fn)(void *), void *step_data,
        void (*ui_fn)(void *), void *ui_data);
};

CountT Viewer::loadObjects(Span<const imp::SourceObject> objs,
                           Span<const imp::SourceMaterial> mats,
                           Span<const imp::SourceTexture> textures)
{
    return impl_->renderer.loadObjects(objs, mats, textures);
}

void Viewer::configureLighting(Span<const LightConfig> lights)
{
    impl_->renderer.configureLighting(lights);
}

const VizECSBridge * Viewer::rendererBridge() const
{
    return impl_->renderer.getBridgePtr();
}

static void handleCamera(GLFWwindow *window,
                         ViewerCam &cam,
                         float cam_move_speed)
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


    if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_2) == GLFW_PRESS ||
            glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS) {

        glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

        math::Vector2 mouse_cur = cursorPosition();
        math::Vector2 mouse_delta = mouse_cur - cam.mousePrev;

        auto around_right = math::Quat::angleAxis(
            mouse_delta.y * InternalConfig::mouseSpeed, cam.right);

        auto around_up = math::Quat::angleAxis(
            -mouse_delta.x * InternalConfig::mouseSpeed, math::up);

        auto rotation = (around_up * around_right).normalize();

        cam.up = rotation.rotateVec(cam.up);
        cam.fwd = rotation.rotateVec(cam.fwd);
        cam.right = rotation.rotateVec(cam.right);

        if (keyPressed(GLFW_KEY_W)) {
            translate += cam.fwd;
        }

        if (keyPressed(GLFW_KEY_A)) {
            translate -= cam.right;
        }

        if (keyPressed(GLFW_KEY_S)) {
            translate -= cam.fwd;
        }

        if (keyPressed(GLFW_KEY_D)) {
            translate += cam.right;
        }

        cam.mousePrev = mouse_cur;
    } else {
        glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);

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

    cam.position += translate * cam_move_speed *
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

// https://lemire.me/blog/2021/06/03/computing-the-number-of-digits-of-an-integer-even-faster/
static int32_t numDigits(uint32_t x)
{
  static uint64_t table[] = {
      4294967296,  8589934582,  8589934582,  8589934582,  12884901788,
      12884901788, 12884901788, 17179868184, 17179868184, 17179868184,
      21474826480, 21474826480, 21474826480, 21474826480, 25769703776,
      25769703776, 25769703776, 30063771072, 30063771072, 30063771072,
      34349738368, 34349738368, 34349738368, 34349738368, 38554705664,
      38554705664, 38554705664, 41949672960, 41949672960, 41949672960,
      42949672960, 42949672960};

  uint32_t idx = 31 - __builtin_clz(x | 1);
  return (x + table[idx]) >> 32;
}

static void cfgUI(Renderer::FrameConfig &cfg,
                  ViewerCam &cam,
                  CountT num_agents,
                  CountT num_worlds,
                  int32_t *tick_rate)
{
    ImGui::Begin("Controls");

    ImGui::TextUnformatted("Utilities");
    bool requested_screenshot = ImGui::Button("Take Screenshot");
    static char output_file_path[256] = "screenshot.bmp";
    ImGui::InputText("Screenshot Output File (.bmp)", output_file_path, 256);

    cfg.requestedScreenshot = requested_screenshot;
    cfg.screenshotFilePath = output_file_path;

    ImGui::TextUnformatted("Simulation Settings");
    ImGui::Separator();

    {
        int world_idx = cfg.worldIDX;
        float worldbox_width =
            ImGui::CalcTextSize(" ").x * (max(numDigits(num_worlds) + 2, 7_i32));
        if (num_worlds == 1) {
            ImGui::BeginDisabled();
        }
        ImGui::PushItemWidth(worldbox_width);
        ImGui::DragInt("Current World ID", &world_idx, 1.f, 0, num_worlds - 1,
                       "%d", ImGuiSliderFlags_AlwaysClamp);
        ImGui::PopItemWidth();

        if (num_worlds == 1) {
            ImGui::EndDisabled();
        }

        cfg.worldIDX = world_idx;
    }

    ImGui::PushItemWidth(ImGui::CalcTextSize(" ").x * 7);
    ImGui::DragInt("Tick Rate (Hz)", (int *)tick_rate, 5.f, 1, 1000);
    if (*tick_rate < 1) {
        *tick_rate = 1;
    }
    ImGui::PopItemWidth();

    ImGui::TextUnformatted("View Settings");
    ImGui::Separator();
    {
        StackAlloc str_alloc;
        const char **cam_opts = str_alloc.allocN<const char *>(num_agents + 1);
        cam_opts[0] = "Free Camera";

        ImVec2 combo_size = ImGui::CalcTextSize(" Free Camera ");

        for (CountT i = 0; i < num_agents; i++) {
            const char *agent_prefix = "Agent ";

            CountT num_bytes = strlen(agent_prefix) + numDigits(i) + 1;
            cam_opts[i + 1] = str_alloc.allocN<char>(num_bytes);
            snprintf((char *)cam_opts[i + 1], num_bytes, "%s%u",
                     agent_prefix, (uint32_t)i);
        }

        CountT cam_idx = cfg.viewIDX;
        ImGui::PushItemWidth(combo_size.x * 1.25f);
        if (ImGui::BeginCombo("Current View", cam_opts[cam_idx])) {
            for (CountT i = 0; i < num_agents + 1; i++) {
                const bool is_selected = (cam_idx == i);
                if (ImGui::Selectable(cam_opts[i], is_selected)) {
                    cam_idx = i;
                }

                if (is_selected) {
                    ImGui::SetItemDefaultFocus();
                }
            }

            ImGui::EndCombo();
        }
        ImGui::PopItemWidth();

        cfg.viewIDX = cam_idx;
    }

    ImGui::Spacing();

    ImGui::TextUnformatted("Free Camera Config");
    ImGui::Separator();

    if (cfg.viewIDX != 0) {
        ImGui::BeginDisabled();
    }
    auto side_size = ImGui::CalcTextSize(" Bottom " );
    side_size.y *= 1.4f;
    ImGui::PushStyleVar(ImGuiStyleVar_ButtonTextAlign,
                        ImVec2(0.5f, 0.f));

    if (ImGui::Button("Top", side_size)) {
        cam.position = 10.f * math::up;
        cam.fwd = -math::up;
        cam.up = -math::fwd;
        cam.right = cross(cam.fwd, cam.up);
    }

    ImGui::SameLine();

    if (ImGui::Button("Left", side_size)) {
        cam.position = -10.f * math::right;
        cam.fwd = math::right;
        cam.up = math::up;
        cam.right = cross(cam.fwd, cam.up);
    }

    ImGui::SameLine();

    if (ImGui::Button("Right", side_size)) {
        cam.position = 10.f * math::right;
        cam.fwd = -math::right;
        cam.up = math::up;
        cam.right = cross(cam.fwd, cam.up);
    }

    ImGui::SameLine();

    if (ImGui::Button("Bottom", side_size)) {
        cam.position = -10.f * math::up;
        cam.fwd = math::up;
        cam.up = math::fwd;
        cam.right = cross(cam.fwd, cam.up);
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

    if (cfg.viewIDX != 0) {
        ImGui::EndDisabled();
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

static ViewerCam initCam(math::Vector3 pos, math::Quat rot)
{
    ViewerCam cam;
    cam.position = pos;
    cam.fwd = normalize(rot.rotateVec(math::fwd));
    cam.up = normalize(rot.rotateVec(math::up));
    cam.right = normalize(cross(cam.fwd, cam.up));

    return cam;
}

Viewer::Impl::Impl(const Config &cfg)
    : cam(initCam(cfg.cameraPosition, cfg.cameraRotation)),
      frameCfg{
         .worldIDX = 0,
         .viewIDX = 0,
      },
      renderer(cfg.gpuID,
               cfg.renderWidth,
               cfg.renderHeight,
               cfg.numWorlds,
               cfg.maxViewsPerWorld,
               cfg.maxInstancesPerWorld,
               cfg.execMode == ExecMode::CUDA,
               {cfg.xLength, cfg.yLength, cfg.zLength}),
      numWorlds(cfg.numWorlds),
      maxNumAgents(cfg.maxViewsPerWorld),
      simTickRate(cfg.defaultSimTickRate),
      cameraMoveSpeed(cfg.cameraMoveSpeed),
      shouldExit(false),
      voxelConfig {
         .xLength = cfg.xLength,
         .yLength = cfg.yLength,
         .zLength = cfg.zLength,
      }
{}

void Viewer::Impl::render(float frame_duration)
{
    // FIXME: pass actual active agents, not max
    cfgUI(frameCfg, cam, maxNumAgents, numWorlds, &simTickRate);

    fpsCounterUI(frame_duration);

    ImGui::Render();

    renderer.render(cam, frameCfg);
}

Viewer::UserInput::UserInput(bool *keys_state, bool *press_state)
    : keys_state_(keys_state), press_state_(press_state)
{}

void Viewer::Impl::loop(
    void (*input_fn)(void *, CountT, CountT, const UserInput &),
    void *input_data, void (*step_fn)(void *), void *step_data,
    void (*ui_fn)(void *), void *ui_data)
{
    GLFWwindow *window = renderer.window.platformWindow;

    std::array<bool, (uint32_t)KeyboardKey::NumKeys> key_state;
    std::array<bool, (uint32_t)KeyboardKey::NumKeys> press_state;
    std::array<bool, (uint32_t)KeyboardKey::NumKeys> prev_key_state;
    utils::zeroN<bool>(key_state.data(), key_state.size());
    utils::zeroN<bool>(prev_key_state.data(), prev_key_state.size());
    utils::zeroN<bool>(press_state.data(), press_state.size());

    float frame_duration = InternalConfig::secondsPerFrame;
    auto last_sim_tick_time = chrono::steady_clock::now();
    while (!glfwWindowShouldClose(window) && !shouldExit) {
        if (frameCfg.viewIDX != 0) {
            key_state[(uint32_t)KeyboardKey::W] |=
                    (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS);
            key_state[(uint32_t)KeyboardKey::A] |=
                    (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS);
            key_state[(uint32_t)KeyboardKey::S] |=
                    (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS);
            key_state[(uint32_t)KeyboardKey::D] |=
                    (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS);
            key_state[(uint32_t)KeyboardKey::Q] |=
                    (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS);
            key_state[(uint32_t)KeyboardKey::E] |=
                    (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS);
            key_state[(uint32_t)KeyboardKey::R] |=
                    (glfwGetKey(window, GLFW_KEY_R) == GLFW_PRESS);
            press_state[(uint32_t)KeyboardKey::R] |=
                    (!prev_key_state[(uint32_t)KeyboardKey::R]) &&
                    (glfwGetKey(window, GLFW_KEY_R) == GLFW_PRESS);
            key_state[(uint32_t)KeyboardKey::X] |=
                    (glfwGetKey(window, GLFW_KEY_X) == GLFW_PRESS);
            press_state[(uint32_t)KeyboardKey::X] |=
                    (!prev_key_state[(uint32_t)KeyboardKey::X]) &&
                    (glfwGetKey(window, GLFW_KEY_X) == GLFW_PRESS);
            key_state[(uint32_t)KeyboardKey::M] |=
                    (glfwGetKey(window, GLFW_KEY_M) == GLFW_PRESS);
            press_state[(uint32_t)KeyboardKey::M] |=
                    (!prev_key_state[(uint32_t)KeyboardKey::M]) &&
                    (glfwGetKey(window, GLFW_KEY_M) == GLFW_PRESS);
            key_state[(uint32_t)KeyboardKey::Z] |=
                    (glfwGetKey(window, GLFW_KEY_Z) == GLFW_PRESS);
            press_state[(uint32_t)KeyboardKey::Z] |=
                    (!prev_key_state[(uint32_t)KeyboardKey::Z]) &&
                    (glfwGetKey(window, GLFW_KEY_Z) == GLFW_PRESS);
            key_state[(uint32_t)KeyboardKey::C] |=
                    (glfwGetKey(window, GLFW_KEY_C) == GLFW_PRESS);
            key_state[(uint32_t)KeyboardKey::G] |=
                    (glfwGetKey(window, GLFW_KEY_G) == GLFW_PRESS);
            key_state[(uint32_t)KeyboardKey::L] |=
                    (glfwGetKey(window, GLFW_KEY_L) == GLFW_PRESS);
            press_state[(uint32_t)KeyboardKey::L] |=
                    (!prev_key_state[(uint32_t)KeyboardKey::L]) &&
                    (glfwGetKey(window, GLFW_KEY_L) == GLFW_PRESS);
            key_state[(uint32_t)KeyboardKey::Space] |=
                    (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS);
            key_state[(uint32_t)KeyboardKey::T] |=
                    (glfwGetKey(window, GLFW_KEY_T) == GLFW_PRESS);
            key_state[(uint32_t)KeyboardKey::F] |=
                    (glfwGetKey(window, GLFW_KEY_F) == GLFW_PRESS);
            key_state[(uint32_t)KeyboardKey::K1] |=
                    (glfwGetKey(window, GLFW_KEY_1) == GLFW_PRESS);
            key_state[(uint32_t)KeyboardKey::K2] |=
                    (glfwGetKey(window, GLFW_KEY_2) == GLFW_PRESS);
            key_state[(uint32_t)KeyboardKey::K3] |=
                    (glfwGetKey(window, GLFW_KEY_3) == GLFW_PRESS);
            key_state[(uint32_t)KeyboardKey::K4] |=
                    (glfwGetKey(window, GLFW_KEY_4) == GLFW_PRESS);
            key_state[(uint32_t)KeyboardKey::K5] |=
                    (glfwGetKey(window, GLFW_KEY_5) == GLFW_PRESS);
            key_state[(uint32_t)KeyboardKey::K6] |=
                    (glfwGetKey(window, GLFW_KEY_6) == GLFW_PRESS);
            key_state[(uint32_t)KeyboardKey::K7] |=
                    (glfwGetKey(window, GLFW_KEY_7) == GLFW_PRESS);
            key_state[(uint32_t)KeyboardKey::K8] |=
                    (glfwGetKey(window, GLFW_KEY_8) == GLFW_PRESS);
            key_state[(uint32_t)KeyboardKey::K9] |=
                    (glfwGetKey(window, GLFW_KEY_9) == GLFW_PRESS);
            key_state[(uint32_t)KeyboardKey::Shift] |=
                    (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS);
        } else {
            handleCamera(window, cam, cameraMoveSpeed);
        }

        auto cur_frame_start_time = chrono::steady_clock::now();

        auto sim_delta_t = chrono::duration<float>(1.f / (float)simTickRate);

        if (cur_frame_start_time - last_sim_tick_time >= sim_delta_t) {
            if (frameCfg.viewIDX != 0) {
                prev_key_state = key_state;
                UserInput user_input(key_state.data(),press_state.data());
                input_fn(input_data, frameCfg.worldIDX,
                         frameCfg.viewIDX - 1, user_input);
                utils::zeroN<bool>(key_state.data(), key_state.size());
                utils::zeroN<bool>(press_state.data(), press_state.size());
            }

            step_fn(step_data);

            last_sim_tick_time = cur_frame_start_time;
        }

        startFrame();

        ui_fn(ui_data);
        render(frame_duration);

        frame_duration = throttleFPS(cur_frame_start_time);
    }

    renderer.waitForIdle();
}

Viewer::Viewer(const Config &cfg)
    : impl_(new Impl(cfg))
{}

Viewer::Viewer(Viewer &&o) = default;
Viewer::~Viewer() = default;

void Viewer::loop(void (*input_fn)(void *, CountT, CountT, const UserInput &),
                  void *input_data, void (*step_fn)(void *), void *step_data,
                  void (*ui_fn)(void *), void *ui_data)
{
    impl_->loop(input_fn, input_data, step_fn, step_data, ui_fn, ui_data);
}

CountT Viewer::getRenderedWorldID() {
    return (CountT)impl_->frameCfg.worldIDX;
}

CountT Viewer::getRenderedViewID() {
    return (CountT)impl_->frameCfg.viewIDX;
}

void Viewer::stopLoop()
{
    impl_->shouldExit = true;
}

}
