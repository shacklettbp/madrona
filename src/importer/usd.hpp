#pragma once

#include <madrona/importer.hpp>
#include <memory>

namespace madrona::imp {

class USDLoader {
public:
    USDLoader(ImageImporter &img_importer, Span<char> err_buf);
    USDLoader(USDLoader &&) = default;
    ~USDLoader();

    bool load(const char *path,
              ImportedAssets &imported_assets,
              bool merge_and_flatten,
              ImageImporter &img_importer);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}
