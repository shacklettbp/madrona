#pragma once

#include <madrona/importer.hpp>

#include <memory>

namespace madrona::imp {

struct GLTFLoader {
    struct Impl;

    GLTFLoader(ImageImporter &img_importer, Span<char> err_buf);
    GLTFLoader(GLTFLoader &&) = default;
    ~GLTFLoader();

    std::unique_ptr<Impl> impl_;

    bool load(const char *path,
              ImportedAssets &imported_assets,
              bool merge_and_flatten,
              ImageImporter &img_importer);
};

}
