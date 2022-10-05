/*
 * Copyright 2021-2022 Brennan Shacklett and contributors
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 */
#include <madrona/memory.hpp>
#include <madrona/utils.hpp>

#include <algorithm>
#include <cstdlib>

#if defined(__linux__) or defined(__APPLE__)
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace madrona {

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
