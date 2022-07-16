#include <madrona/memory.hpp>
#include <madrona/utils.hpp>

#include <algorithm>
#include <cstdlib>

#if defined(__linux__) or defined(__APPLE__)
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace madrona {

namespace InternalConfig {
    inline constexpr size_t BumpAllocBlockSize = 2 * 1024 * 1024;
};

AllocScope::AllocScope(const PolyAlloc &alloc, AllocScope *parent,
                       AllocContext *ctx)
    : cur_alloc_(alloc), parent_(parent), ctx_(ctx)
{
    ctx_->cur_alloc_ = alloc;
    ctx_->cur_scope_ = this;
}

AllocScope::~AllocScope()
{
    if (parent_ != nullptr) [[likely]] {
        ctx_->cur_alloc_ = parent_->cur_alloc_;
    }
    ctx_->cur_scope_ = parent_;
}

AllocContext::AllocContext()
    : cur_alloc_(nullptr, nullptr, nullptr),
      cur_scope_(nullptr)
{}

}
