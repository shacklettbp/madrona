#pragma once

#include <madrona/importer.hpp>

namespace madrona::imp {

struct URDFLoader {
    struct Impl;

    URDFLoader(Span<char> err_buf);
    URDFLoader(URDFLoader &&) = default;
    ~URDFLoader();

    std::unique_ptr<Impl> impl_;

    bool load(const char *path, ImportedAssets &imported_assets);
};
    
}
