/*
 * Copyright 2021-2022 Brennan Shacklett and contributors
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 */
#pragma once

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <type_traits>
#include <utility>

#include <madrona/memory.hpp>
#include <madrona/utils.hpp>
#include <madrona/span.hpp>

namespace madrona {

template <typename T, typename A = DefaultAlloc>
class HeapArray {
public:
    using RefT = std::add_lvalue_reference_t<T>;

    inline explicit HeapArray(CountT n, A alloc = A())
        : alloc_(std::move(alloc)),
          ptr_((T *)alloc_.alloc(n * sizeof(T))),
          n_(n)
    {}

    template <typename U = T,
              typename std::enable_if_t<std::is_copy_constructible_v<U> &&
                                        std::is_same_v<U, T>, bool> = false>
    HeapArray(std::initializer_list<T> init, A alloc = DefaultAlloc())
        : HeapArray(init.size(), std::move(alloc))
    {
        memcpy(ptr_, std::data(init), init.size() * sizeof(T));
    }

    HeapArray(const HeapArray &) = delete;

    HeapArray(HeapArray &&o)
        : alloc_(std::move(o.alloc_)),
          ptr_(o.ptr_),
          n_(o.n_)
    {
        o.ptr_ = nullptr;
    }

    HeapArray<T, A> & operator=(const HeapArray &) = delete;
    HeapArray<T, A> & operator=(HeapArray &&o)
    {
        if (ptr_ != nullptr) {
            clear();
            alloc_.dealloc(ptr_);
        }

        ptr_ = o.ptr_;
        n_ = o.n_;

        o.ptr_ = nullptr;
        o.n_ = 0;

        return *this;
    }

    void clear()
    {
        if constexpr (!std::is_trivially_destructible_v<T>) {
            for (CountT i = 0; i < n_; i++) {
                ptr_[i].~T();
            }
        }
    }

    ~HeapArray()
    {
        if (ptr_ == nullptr) return;

        clear();
        alloc_.dealloc(ptr_);
    }

    Span<T> release()
    {
        Span<T> span(ptr_, n_);

        ptr_ = nullptr;
        n_ = 0;

        return span;
    }

    RefT insert(CountT i, T &&v)
    {
        new (&ptr_[i]) T(std::move(v));

        return ptr_[i];
    }

    template <typename... Args>
    RefT emplace(CountT i, Args &&...args)
    {
        new (&ptr_[i]) T(std::forward<Args>(args)...);

        return ptr_[i];
    }

    void destruct(CountT i)
    {
        ptr_[i].~T();
    }

    RefT operator[](CountT idx) { return ptr_[idx]; }
    const RefT operator[](CountT idx) const { return ptr_[idx]; }

    T *data() { return ptr_; }
    const T *data() const { return ptr_; }

    T *begin() { return ptr_; }
    T *end() { return ptr_ + n_; }
    const T *begin() const { return ptr_; }
    const T *end() const { return ptr_ + n_; }

    CountT size() const { return n_; }

private:
#if __cplusplus >= 202002L
    [[no_unique_address]] 
#endif
        A alloc_;
    T *ptr_;
    CountT n_;
};

}
