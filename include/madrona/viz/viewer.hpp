#pragma once

#include <madrona/types.hpp>
#include <madrona/render/mw.hpp>
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

class Viewer {
public:
    struct Config {
        int gpuID;
        uint32_t renderWidth;
        uint32_t renderHeight;
        uint32_t numWorlds;
        uint32_t maxViewsPerWorld;
        uint32_t maxInstancesPerWorld;
        uint32_t defaultSimTickRate;
        float cameraMoveSpeed;
        ExecMode execMode;
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

    CountT loadObjects(Span<const imp::SourceObject> objs, Span<const imp::SourceMaterial> mats,
                       Span<const imp::SourceTexture> textures);

    void configureLighting(Span<const LightConfig> lights);

    const VizECSBridge * rendererBridge() const;

    template <typename InputFn, typename StepFn, typename UIFn>
    void loop(InputFn &&input_fn, StepFn &&step_fn, UIFn &&ui_fn);

    int32_t getRenderedWorldID();

    int32_t getRenderedViewID();

    void stopLoop();

private:
    void loop(void (*input_fn)(void *, CountT, CountT, const UserInput &),
              void *input_data, void (*step_fn)(void *), void *step_data, 
              void (*ui_fn)(void*), void* ui_data);

    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}

#include "viewer.inl"
