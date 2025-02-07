#pragma once

#include <madrona/importer.hpp>
#include <madrona/cvphysics.hpp>
#include <madrona/physics_assets.hpp>

namespace madrona::imp {

struct URDFLoader {
    struct Impl;

    URDFLoader();
    ~URDFLoader();

    std::unique_ptr<Impl> impl;

    struct BuiltinPrimitives {
        uint32_t cubeRenderIdx;
        uint32_t cubePhysicsIdx;
        uint32_t planeRenderIdx;
        uint32_t planePhysicsIdx;
        uint32_t sphereRenderIdx;
        uint32_t spherePhysicsIdx;
    };

    uint32_t load(
          const char *path, 
          BuiltinPrimitives primitives,
          std::vector<std::string> &render_asset_paths,
          std::vector<std::string> &physics_asset_paths,
          bool visualize_colliders = false);

    phys::cv::ModelData getModelData();
    phys::cv::ModelConfig * getModelConfigs(uint32_t &num_cfgs);
};
    
}
