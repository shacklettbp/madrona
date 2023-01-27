#pragma once

#include <madrona/types.hpp>
#include <madrona/mw_render.hpp>
#include <madrona/importer.hpp>
#include <memory>

namespace madrona {
namespace render {

class WorldGrid {
public:
    inline WorldGrid(int64_t num_worlds, float cell_dim)
    {
        num_cells_x_ =
            CountT(roundf(powf(float(num_worlds), 1.f / 3.f)));
        num_cells_y_ = num_cells_x_;
        num_cells_z_ = num_worlds / (num_cells_x_ * num_cells_y_);

        while (num_cells_x_ * num_cells_y_ * num_cells_z_ < num_worlds) {
            num_cells_z_++;
        }

        float grid_width = cell_dim * num_cells_x_;
        float grid_depth = cell_dim * num_cells_y_;
        float grid_height = cell_dim * num_cells_z_;

        float min_x = -grid_width / 2.f + cell_dim / 2.f;
        float min_y = -grid_depth / 2.f + cell_dim / 2.f;
        float min_z = -grid_height / 2.f + cell_dim / 2.f;

        cell_dim_ = cell_dim;
        min_corner_ = {
            min_x,
            min_y,
            min_z,
        };
    }

    inline math::Vector3 getOffset(int64_t world_idx)
    {
        int64_t x_offset = world_idx % num_cells_x_;
        int64_t y_offset = (world_idx / num_cells_x_) % num_cells_y_;
        int64_t z_offset = world_idx / (num_cells_x_ * num_cells_y_);

        return min_corner_ + math::Vector3 {
            cell_dim_ * float(x_offset),
            cell_dim_ * float(y_offset),
            cell_dim_ * float(z_offset),
        };
    }

private:
    int64_t num_cells_x_;
    int64_t num_cells_y_;
    int64_t num_cells_z_;
    float cell_dim_;
    math::Vector3 min_corner_;
};

class BatchRenderer {
public:
    enum class CameraMode : uint32_t {
        Perspective,
        Lidar,
    };

    enum class InputMode : uint32_t {
        CPU,
        CUDA,
    };

    struct Config {
        int gpuID;
        uint32_t renderWidth;
        uint32_t renderHeight;
        uint32_t numWorlds;
        uint32_t maxViewsPerWorld;
        uint32_t maxInstancesPerWorld;
        uint32_t maxObjects;
        CameraMode cameraMode;
        InputMode inputMode;
    };

    BatchRenderer(const Config &cfg);
    BatchRenderer(BatchRenderer &&o);

    ~BatchRenderer();

    CountT loadObjects(Span<const imp::SourceObject> objs);

    RendererInterface getInterface() const;

    uint8_t * rgbPtr() const;
    float * depthPtr() const;

    void render();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}
}
