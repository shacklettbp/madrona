#pragma once

#include <madrona/types.hpp>
#include <madrona/render/batch_renderer_system.hpp>
#include <madrona/importer.hpp>
#include <madrona/exec_mode.hpp>
#include <memory>

#include <madrona/viz/system.hpp>

namespace madrona::viz {

struct LightConfig {
    bool isDirectional;

    // Used for direction or position depending on value of isDirectional
    math::Vector3 dir;
    math::Vector3 color;
};

struct VoxelConfig {
    uint32_t xLength;
    uint32_t yLength;
    uint32_t zLength;
};

class Viewer {
public:
    struct Config {
        int gpuID;

        // This is the rendering width/height of the actual viewer itself
        uint32_t renderWidth;
        uint32_t renderHeight;
        // This is the rendering width/height of the batch renderer output
        uint32_t batchWidth;
        uint32_t batchHeight;

        uint32_t numWorlds;
        uint32_t maxViewsPerWorld;
        uint32_t maxInstancesPerWorld;
        uint32_t defaultSimTickRate;
        float cameraMoveSpeed;
        math::Vector3 cameraPosition;
        math::Quat cameraRotation;
        ExecMode execMode;
        uint32_t xLength = 0;
        uint32_t yLength = 0;
        uint32_t zLength = 0;
    };

    enum class KeyboardKey : uint32_t {
        W,
        A,
        S,
        D,
        Q,
        E,
        R,
        X,
        Z,
        C,
        G,
        L,
        T,
        F,
        K1,
        K2,
        K3,
        K4,
        K5,
        K6,
        K7,
        K8,
        K9,
        K0,
        Shift,
        Space,
        NumKeys,
    };

    class UserInput {
    public:
        inline UserInput(bool *keys_state);

        inline bool keyPressed(KeyboardKey key) const;

    private:
        bool *keys_state_;
    };

    Viewer(const Config &cfg);
    Viewer(Viewer &&o);

    ~Viewer();

    CountT loadObjects(Span<const imp::SourceObject> objs,
                       Span<const imp::SourceMaterial> mats,
                       Span<const imp::SourceTexture> textures);

    void configureLighting(Span<const LightConfig> lights);

    const VizECSBridge * rendererBridge() const;

    template <typename InputFn, typename StepFn, typename UIFn>
    void loop(InputFn &&input_fn, StepFn &&step_fn, UIFn &&ui_fn);

    CountT getRenderedWorldID();
    CountT getRenderedViewID();

    void stopLoop();

private:
    void loop(void (*input_fn)(void *, CountT, CountT, const UserInput &),
              void *input_data, void (*step_fn)(void *), void *step_data, 
              void (*ui_fn)(void *), void *ui_data);

    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}

#include "viewer.inl"
