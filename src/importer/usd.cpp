#include "usd.hpp"

namespace madrona::imp {

struct USDLoader::Impl {
    Span<char> errBuf;

    static inline Impl * init(Span<char> err_buf);
};

USDLoader::Impl * USDLoader::Impl::init(Span<char> err_buf)
{
    return nullptr;
}

USDLoader::USDLoader(Span<char> err_buf)
    : impl_(Impl::init(err_buf))
{}

USDLoader::~USDLoader() = default;

}
