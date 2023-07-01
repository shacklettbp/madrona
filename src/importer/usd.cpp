#include "usd.hpp"

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

USDLoader::USDLoader(Span<char> err_buf)
    : impl_(Impl::init(err_buf))
{}

USDLoader::~USDLoader() = default;

bool USDLoader::load(const char *path, ImportedAssets &imported_assets,
                     bool merge_and_flatten)
{
    (void)path;
    (void)imported_assets;
    (void)merge_and_flatten;

    return false;
}

}
