#pragma once

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iterator>
#include <type_traits>
#include <utility>

#include "memory.hpp"
#include "utils.hpp"

namespace madrona {

template <typename T, typename A = DefaultAlloc>
class HeapArray {
public:
    using RefT = std::add_lvalue_reference_t<T>;

    explicit HeapArray(size_t n, A alloc = DefaultAlloc())
        : alloc_(std::move(alloc)),
          ptr_((T *)alloc_.alloc(n * sizeof(T))),
          n_(n)
    {}

    HeapArray(const HeapArray &) = delete;

    HeapArray(HeapArray &&o)
        : alloc_(std::move(o.alloc_)),
          ptr_(o.ptr_),
          n_(o.n_)
    {
        o.ptr_ = nullptr;
    }

    RefT operator=(const HeapArray &) = delete;
    RefT operator=(HeapArray &&o)
    {
        clear();
        alloc_.dealloc(ptr_);

        ptr_ = o.ptr_;
        n_ = o.n_;

        o.ptr_ = nullptr;
        o.n_ = 0;
    }

    void clear()
    {
        if constexpr (!std::is_trivially_destructible_v<T>) {
            for (size_t i = 0; i < n_; i++) {
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

    RefT insert(size_t i, T v)
    {
        new (&ptr_[i]) T(std::move(v));

        return ptr_[i];
    }

    RefT insert(size_t i, T &&v)
    {
        new (&ptr_[i]) T(std::move(v));

        return ptr_[i];
    }

    template <typename... Args>
    RefT emplace(size_t i, Args &&...args)
    {
        new (&ptr_[i]) T(std::forward<Args>(args)...);

        return ptr_[i];
    }

    void destruct(size_t i)
    {
        ptr_[i].~T();
    }

    RefT operator[](size_t idx) { return ptr_[idx]; }
    const RefT operator[](size_t idx) const { return ptr_[idx]; }

    T *data() { return ptr_; }
    const T *data() const { return ptr_; }

    T *begin() { return ptr_; }
    T *end() { return ptr_ + n_; }
    const T *begin() const { return ptr_; }
    const T *end() const { return ptr_ + n_; }

    size_t size() const { return n_; }

private:
    [[no_unique_address]] A alloc_;
    T *ptr_;
    const size_t n_;
};

}
