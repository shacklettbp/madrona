#pragma once

#include <madrona/span.hpp>
#include <madrona/render/common.hpp>

namespace madrona::render {

enum class APIBackendSelect : uint32_t {
    Auto,
    Vulkan,
    Metal,
    Cuda,
};

class GPUHandle {
public:
    GPUHandle(APIBackendSelect backend, GPUDevice *dev);
    ~GPUHandle();

    inline GPUHandle(GPUHandle &&o);
    inline GPUHandle & operator=(GPUHandle &&o);

    inline GPUDevice * device() const;

private:
    APIBackendSelect backend_;
    GPUDevice *dev_;
};

class APILibHandle {
public:
    APILibHandle(APIBackendSelect backend, APILib *lib);
    ~APILibHandle();

    inline APILibHandle(APILibHandle &&o);
    inline APILibHandle & operator=(APILibHandle &&o);

    inline APILib * lib() const;

private:
    APIBackendSelect backend_;
    APILib *lib_;
};

class APIManager {
public:
    struct Config {
        bool enableValidation = false;
        bool enablePresent = false;
        Span<const char *const> apiExtensions = {};
    };

    static APILibHandle loadDefaultLib(
        APIBackendSelect backend_select = APIBackendSelect::Auto);

    APIManager(APILib *api_lib,
               const Config &cfg = defaultConfig(),
               APIBackendSelect backend_select = APIBackendSelect::Auto);
    APIManager(APIManager &&);
    ~APIManager();

    APIManager & operator=(APIManager &&);

    inline APIBackend * backend() const;
    GPUHandle initGPU(CountT gpu_idx);

private:
    static inline Config defaultConfig();

    APIBackendSelect chosen_backend_;
    APIBackend *backend_;
};

}

#include "api.inl"
