#pragma once

#include <madrona/importer.hpp>

namespace madrona::imp {

struct OBJLoader {
    struct Impl;

    OBJLoader(Span<char> err_buf);
    OBJLoader(OBJLoader &&) = default;
    ~OBJLoader();

    std::unique_ptr<Impl> impl_;

    bool load(const char *path, ImportedAssets &imported_assets);
};

}
