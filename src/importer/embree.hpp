#pragma once

#include <madrona/mesh_bvh.hpp>
#include <madrona/importer.hpp>

namespace madrona::imp {

struct EmbreeLoader {
    struct Impl;

    EmbreeLoader(Span<char> err_buf);
    EmbreeLoader(EmbreeLoader &&) = default;
    ~EmbreeLoader();

    std::unique_ptr<Impl> impl_;

    Optional<render::MeshBVH> load(const SourceObject &obj);
};
    
}
