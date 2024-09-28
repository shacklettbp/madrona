/*
 * Copyright 2021-2022 Brennan Shacklett and contributors
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 */
#pragma once

#include <madrona/types.hpp>
#include <madrona/macros.hpp>

#include <array>
#include <cstdint>
#include <initializer_list>

namespace madrona {

template <typename T>
class Span {
public:
    Span() : ptr_(nullptr), n_(0) {}

    Span(T *ptr, CountT num_elems)
        : ptr_(ptr), n_(num_elems)
    {}

    template <CountT N>
    Span(T (&arr)[N])
        : ptr_(arr),
          n_(N)
    {}

    template <typename U, CountT N>
    Span(const std::array<U, N> &arr)
            requires(std::is_same_v<std::remove_cv_t<T>,
                                    std::remove_cv_t<U>>)
        : ptr_(arr.data()),
          n_(N)
    {}

    // GCC correctly warns that the below constructor is dangerous, but it's
    // convenient as long as the Span doesn't outlive the current expression
#if MADRONA_GCC
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Winit-list-lifetime"
#endif
    Span(std::initializer_list<std::remove_cv_t<T>> const && init MADRONA_LFBOUND)
            requires(std::is_const_v<T>)
        : ptr_(init.begin()), n_(init.size())
    {}
#ifdef MADRONA_GCC
#pragma GCC diagnostic pop
#endif

    template <typename U>
    Span(const U &u)
        : ptr_(u.data()), n_(u.size())
    {}

    constexpr T * data() const { return ptr_; }

    constexpr CountT size() const { return n_; }

    T & operator[](CountT idx) const { return ptr_[idx]; }

    T * begin() const { return ptr_; }
    T * end() const { return ptr_ + n_; }

private:
    T *ptr_;
    CountT n_;
};

}
