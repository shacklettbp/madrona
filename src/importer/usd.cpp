#undef min
#undef max
#include <tinyusdz.hh>

#include "usd.hpp"

#include <string>

namespace madrona::imp {

struct USDLoader::Impl {
    Span<char> errBuf;

    static inline Impl * init(Span<char> err_buf);
};

USDLoader::Impl * USDLoader::Impl::init(Span<char> err_buf)
{
    return new Impl {
        .errBuf = err_buf,
    };
}

USDLoader::USDLoader(ImageImporter &, Span<char> err_buf)
    : impl_(Impl::init(err_buf))
{}

USDLoader::~USDLoader() = default;

bool USDLoader::load(const char *path,
                     ImportedAssets &imported_assets,
                     bool merge_and_flatten,
                     ImageImporter &)
{
    tinyusdz::Stage stage;
    std::string warn, err;

    bool ret = tinyusdz::LoadUSDFromFile(path, &stage, &warn, &err, {
        .load_assets = false,
        .do_composition = true,
        .load_sublayers = true,
        .load_references = true,
        .load_payloads = true,
    });

    if (warn.size()) {
        printf("USD Loader Warning: %s\n", warn.c_str());
    }

    if (!ret) {
        if (!err.empty()) {
            printf("USD Loader Error: %s\n", err.c_str());
        }

        return false;
    }

    printf("USD File: %s\n", stage.ExportToString().c_str());

    return false;
}

}
