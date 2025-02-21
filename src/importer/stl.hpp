#pragma once

#include <madrona/importer.hpp>

namespace madrona::imp {
    
struct STLLoader {
    struct Impl;

    STLLoader(Span<char> err_buf);
    STLLoader(STLLoader &&) = default;
    ~STLLoader();

    std::unique_ptr<Impl> impl_;

    bool load(const char *path, ImportedAssets &imported_assets);
};

}
