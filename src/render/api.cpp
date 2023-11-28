#include <madrona/render/api.hpp>
#include <madrona/render/vk/backend.hpp>

#include <cassert>

namespace madrona::render {

static APIBackendSelect chooseBackend(APIBackendSelect desired)
{
    assert(desired == APIBackendSelect::Vulkan ||
           desired == APIBackendSelect::Auto);

    return APIBackendSelect::Vulkan;
}

static bool checkVkValidationOverride(bool enable_validation)
{
    char *validate_env = getenv("MADRONA_VK_VALIDATE");
    if (validate_env != nullptr && validate_env[0] == '1') {
        enable_validation = true;
    }

    return enable_validation;
}

static APIBackend * initVkBackend(APILib *lib, const APIManager::Config &cfg)
{
    auto vk_lib = static_cast<vk::LoaderLib *>(lib);
    return new vk::Backend(
        vk_lib->getEntryFn(),
        checkVkValidationOverride(cfg.enableValidation),
        cfg.enablePresent,
        cfg.apiExtensions);
}

static APIBackend * initBackend(APIBackendSelect chosen_backend,
                                APILib *lib,
                                const APIManager::Config &cfg)
{
    switch (chosen_backend) {
    case APIBackendSelect::Vulkan: {
        return initVkBackend(lib, cfg);
    } break;
    default: {
        MADRONA_UNREACHABLE();
    } break;
    }
}

GPUHandle::GPUHandle(APIBackendSelect backend, GPUDevice *dev)
    : backend_(backend),
      dev_(dev)
{}

GPUHandle::~GPUHandle()
{
    if (!dev_) {
        return;
    }

    switch (backend_) {
    case APIBackendSelect::Vulkan: {
        auto api_dev = static_cast<vk::Device *>(dev_);
        delete api_dev;
    } break;
    default: {
        MADRONA_UNREACHABLE();
    } break;
    }
}

APILibHandle::APILibHandle(APIBackendSelect backend, APILib *lib)
    : backend_(backend),
      lib_(lib)
{}

APILibHandle::~APILibHandle()
{
    if (lib_ == nullptr) {
        return;
    }

    switch (backend_) {
    case APIBackendSelect::Vulkan: {
        auto lib = static_cast<vk::LoaderLib *>(lib_);
        delete lib;
    } break;
    default: {
        MADRONA_UNREACHABLE();
    } break;
    }
}

APILibHandle APIManager::loadDefaultLib(APIBackendSelect backend_select)
{
    backend_select = chooseBackend(backend_select);

    APILib *lib;
    switch (backend_select) {
    case APIBackendSelect::Vulkan: {
        lib = vk::LoaderLib::load();
    } break;
    default: {
        MADRONA_UNREACHABLE();
    } break;
    }

    return APILibHandle(backend_select, lib);
}

APIManager::APIManager(APILib *api_lib,
                       const Config &cfg,
                       APIBackendSelect backend_select)
    : chosen_backend_(chooseBackend(backend_select)),
      backend_(initBackend(chosen_backend_, api_lib, cfg))
{}

APIManager::APIManager(APIManager &&o)
    : chosen_backend_(o.chosen_backend_),
      backend_(o.backend_)
{
    o.backend_ = nullptr;
}

APIManager::~APIManager()
{
    if (backend_ == nullptr) {
        return;
    }

    switch (chosen_backend_) {
    case APIBackendSelect::Vulkan: {
        auto backend = static_cast<vk::Backend *>(backend_);
        delete backend;
    } break;
    default: {
        MADRONA_UNREACHABLE();
    } break;
    }
}

APIManager & APIManager::operator=(APIManager &&o) 
{
    chosen_backend_ = o.chosen_backend_;
    backend_ = o.backend_;

    o.backend_ = nullptr;

    return *this;
}

GPUHandle APIManager::initGPU(CountT gpu_idx)
{
    switch (chosen_backend_) {
    case APIBackendSelect::Vulkan: {
        auto backend = static_cast<vk::Backend *>(backend_);
        return GPUHandle(
            APIBackendSelect::Vulkan, backend->makeDevice(gpu_idx));
    } break;
    default: {
        MADRONA_UNREACHABLE();
    } break;
    }
}

}
