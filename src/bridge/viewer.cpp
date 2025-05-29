#include <madrona/viz/viewer.hpp>
#include <madrona/render/render_mgr.hpp>
#include <madrona/window.hpp>
#include <madrona/py/bindings.hpp>

#include "sim.hpp"
#include "mgr.hpp"
#include "types.hpp"

#include <filesystem>
#include <fstream>
#include <imgui.h>

using namespace madrona;
using namespace madrona::viz;

namespace nb = nanobind;

namespace madMJX {

struct VisualizerGPUState {
    WindowManager wm;
    WindowHandle window;
    render::GPUHandle renderGPU;

    inline VisualizerGPUState(int64_t window_width,
                              int64_t window_height,
                              int gpu_id)
        : wm(),
          window(wm.makeWindow("MadMJX", window_width, window_height)),
          renderGPU(wm.initGPU(gpu_id, { window.get() }))
    {}

    inline VisualizerGPUHandles getGPUHandles()
    {
        return VisualizerGPUHandles {
            .renderAPI = wm.gpuAPIManager().backend(),
            .renderDev = renderGPU.device(),
        };
    }
};

struct Visualizer {
    Viewer viewer;
    uint32_t numCams;
    uint32_t batchViewWidth;
    uint32_t batchViewHeight;

    inline Visualizer(VisualizerGPUState &gpu_state, Manager &mgr)
        : viewer(mgr.getRenderManager(), gpu_state.window.get(), {
            .numWorlds = mgr.numWorlds(),
            .simTickRate = 30,
            .cameraMoveSpeed = 5.f,
            .cameraPosition = { 0, -3, 0 },
            .cameraRotation = { 1, 0, 0, 0 },
        }), numCams(mgr.numCams()),
            batchViewWidth(mgr.batchViewWidth()),
            batchViewHeight(mgr.batchViewHeight())
    {}

    template <typename Fn>
    inline void loop(Manager &mgr, Fn &&sim_cb)
    {
        // Main loop for the viewer
        viewer.loop(
        [&mgr](CountT world_idx, const Viewer::UserInput &input)
        {
            (void)mgr;
            (void)world_idx;
            (void)input;
        },
        [&mgr](CountT world_idx, CountT,
               const Viewer::UserInput &input)
        {
            (void)mgr;
            (void)world_idx;
            (void)input;
        }, [&]() {
            sim_cb();
        }, [&]() {
#ifdef MADRONA_CUDA_SUPPORT
            uint32_t raycast_output_width = batchViewWidth;
            uint32_t raycast_output_height = batchViewHeight;
            uint32_t image_idx = viewer.getCurrentWorldID() * 
                numCams + std::max(viewer.getCurrentViewID(), (CountT)0);
            int64_t num_bytes = 4 * raycast_output_width * 
                raycast_output_height;

            // Extract depth from buffer
            unsigned char* depth_print_ptr;
            depth_print_ptr = (unsigned char*)cu::allocReadback(num_bytes);

            char *raycast_tensor_depth = (char *)(mgr.depthTensor().devicePtr());

            uint32_t bytes_per_image = 4 * raycast_output_width * 
                raycast_output_height;

            raycast_tensor_depth += image_idx * bytes_per_image;

            cudaMemcpy(depth_print_ptr, raycast_tensor_depth,
                bytes_per_image,
                cudaMemcpyDeviceToHost);
            raycast_tensor_depth = (char *)depth_print_ptr;

            // Extract rgb from buffer
            unsigned char* rgb_print_ptr;
            rgb_print_ptr = (unsigned char*)cu::allocReadback(num_bytes);
            char *raycast_tensor_rgb = (char *)(mgr.rgbTensor().devicePtr());

            raycast_tensor_rgb += image_idx * bytes_per_image;
            cudaMemcpy(rgb_print_ptr, raycast_tensor_rgb,
                bytes_per_image,
                cudaMemcpyDeviceToHost);
            raycast_tensor_rgb = (char *)rgb_print_ptr;

            ImGui::Begin("RGB and Depth Tensor Debug");

            auto draw2 = ImGui::GetWindowDrawList();
            ImVec2 windowPos = ImGui::GetWindowPos();
            char *raycasters = raycast_tensor_depth;

            int vertOff = 40;

            float pixScale = 5;
            float pixSpace = 5;

            for (int i = 0; i < (int)raycast_output_height; i++) {
                for (int j = 0; j < (int)raycast_output_width; j++) {
                    uint32_t linear_idx = 4 * (j + i * raycast_output_width);

                    float *depth = (float *)(raycasters + linear_idx);

                    // float depth_convert = std::max(0.0f, std::min(255.0f * (1.0f / (*depth)), 255.0f));
                    float depth_convert = 255.0f * (*depth) / 4.f;
                    depth_convert = std::min(255.f, std::max(0.f, depth_convert));

                    auto realColor = IM_COL32(
                        (uint8_t)depth_convert,
                        (uint8_t)depth_convert,
                        (uint8_t)depth_convert, 
                        255);

                    draw2->AddRectFilled(
                        { (j * pixSpace) + windowPos.x, 
                          (i * pixSpace) + windowPos.y +vertOff }, 
                        { (j * pixSpace + pixScale) + windowPos.x,   
                          (i * pixSpace + pixScale)+ +windowPos.y+vertOff },
                        realColor, 0, 0);
                }
            }

            int horOff = 340;
            raycasters = raycast_tensor_rgb;
            for (int i = 0; i < (int)raycast_output_height; i++) {
                for (int j = 0; j < (int)raycast_output_width; j++) {
                    uint32_t linear_idx = 4 * (j + i * raycast_output_width);

                    auto realColor = IM_COL32(
                        (uint8_t)raycasters[linear_idx + 0],
                        (uint8_t)raycasters[linear_idx + 1],
                        (uint8_t)raycasters[linear_idx + 2], 
                        255);

                    draw2->AddRectFilled(
                        { (j * pixSpace) + windowPos.x + horOff, 
                          (i * pixSpace) + windowPos.y + vertOff }, 
                        { (j * pixSpace + pixScale) + windowPos.x + horOff,   
                          (i * pixSpace + pixScale)+ +windowPos.y+vertOff },
                        realColor, 0, 0);
                }
            }
            ImGui::End();
#endif
           });
    }
};

NB_MODULE(_madrona_gs_visualizer, m) {
    nb::class_<VisualizerGPUState>(m, "VisualizerGPUState")
        .def("__init__", [](VisualizerGPUState *self,
                            int64_t window_width,
                            int64_t window_height,
                            int gpu_id) {
            new (self) VisualizerGPUState(window_width, window_height, gpu_id);
        }, nb::arg("window_width"),
           nb::arg("window_height"),
           nb::arg("gpu_id") = 0)
        .def("get_gpu_handles", &VisualizerGPUState::getGPUHandles, 
             nb::keep_alive<0, 1>())
    ;

    nb::class_<Visualizer>(m, "Visualizer")
        .def("__init__", [](Visualizer *self,
                            VisualizerGPUState *viz_gpu_state,
                            Manager *mgr) {
            new (self) Visualizer(*viz_gpu_state, *mgr);
        }, nb::arg("visualizer_gpu_state"),
           nb::arg("manager"),
           nb::keep_alive<1, 2>(),
           nb::keep_alive<1, 3>())
        .def("loop",
            [](Visualizer *self, Manager *mgr, nb::callable cb,
               nb::object carry)
        {
            self->loop(*mgr, [&]() {
                carry = cb(carry);
            });

            return carry;
        })
    ;
}

}
