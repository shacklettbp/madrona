/*
 * Copyright 2021-2022 Brennan Shacklett and contributors
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 */
#pragma once

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iterator>
#include <type_traits>
#include <utility>

#include <madrona/memory.hpp>
#include <madrona/utils.hpp>

namespace madrona {

// Not a replacement for std::vector!
// This class memcpys items on expansion
template <typename T, typename A = DefaultAlloc>
class DynArray {
public:
    using RefT = std::add_lvalue_reference_t<T>;

    explicit DynArray(CountT init_capacity, A alloc = DefaultAlloc())
        : alloc_(std::move(alloc)),
          ptr_(init_capacity > 0 ?
                   (T *)alloc_.alloc(init_capacity * sizeof(T)) :
                   nullptr),
          n_(0),
          capacity_(init_capacity)
    {}

    template <typename U = T,
              typename std::enable_if_t<std::is_copy_constructible_v<U> &&
                                        std::is_same_v<U, T>, bool> = false>
    DynArray(std::initializer_list<T> init, A alloc = DefaultAlloc())
        : DynArray(init.size(), std::move(alloc))
    {
        for (const T &t : init) {
            push_back(t);
        }
    }

    DynArray(const DynArray &) = delete;
    DynArray(DynArray &&o)
        : alloc_(std::move(o.alloc_)),
          ptr_(o.ptr_),
          n_(o.n_),
          capacity_(o.capacity_)
    {
        o.ptr_ = nullptr;
        o.n_ = 0;
        o.capacity_ = 0;
    }

    ~DynArray()
    {
        if (ptr_ == nullptr) return;

        release();
    }

    DynArray & operator=(const DynArray &) = delete;
    DynArray & operator=(DynArray &&o)
    {
        release();

        ptr_ = o.ptr_;
        n_ = o.n_;
        capacity_ = o.capacity_;

        o.ptr_ = nullptr;
        o.n_ = 0;
        o.capacity_ = 0;

        return *this;
    }

    void clear()
    {
        if constexpr (!std::is_trivially_destructible_v<T>) {
            for (CountT i = 0; i < n_; i++) {
                ptr_[i].~T();
            }
        }
        n_ = 0;
    }

    void release()
    {
        clear();
        alloc_.dealloc(ptr_);

        ptr_ = nullptr;
    }

    T * retrieve_ptr()
    {
        T *ptr = ptr_;
        ptr_ = nullptr;
        return ptr;
    }

    void set_min_capacity(CountT capacity)
    {
        if (capacity >= capacity_) {
            return;
        }

        void *old = realloc(capacity);
        alloc_.dealloc(old);
    }

    template <typename Fn>
    void resize(CountT new_size, Fn &&fn)
    {
        if (new_size > capacity_) {
            void *old = expand(new_size);
            alloc_.dealloc(old);
        }

        if (new_size > n_) {
            for (CountT i = n_; i < new_size; i++) {
                fn(&ptr_[i]);
            }
        } else {
            if constexpr (!std::is_trivially_destructible_v<T>) {
                for (CountT i = new_size; i < n_; i++) {
                    ptr_[i].~T();
                }
            }
        }

        n_ = new_size;
    }

    void reserve(CountT new_capacity)
    {
        if (new_capacity > capacity_) {
            void *old_ptr = expand(new_capacity);
            alloc_.dealloc(old_ptr);
        }
    }

    RefT insert(CountT i, T v)
    {
        new (&ptr_[i]) T(std::move(v));

        return ptr_[i];
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

    template <typename... Args>
    RefT emplace_back(Args &&...args)
    {
        if (n_ == capacity_) [[unlikely]] {
            T *old_ptr = expand(n_ + 1);
            new (&ptr_[n_]) T(std::forward<Args>(args)...);
            alloc_.dealloc(old_ptr);
        } else {
            new (&ptr_[n_]) T(std::forward<Args>(args)...);
        }

        return ptr_[n_++];
    }

    RefT push_back(const T &v)
    {
        if (n_ == capacity_) [[unlikely]] {
            T *old_ptr = expand(n_ + 1);
            new (&ptr_[n_]) T(v);
            alloc_.dealloc(old_ptr);
        } else {
            new (&ptr_[n_]) T(v);
        }

        return ptr_[n_++];
    }

    RefT push_back(T &&v)
    {
        if (n_ == capacity_) [[unlikely]] {
            T *old_ptr = expand(n_ + 1);
            new (&ptr_[n_]) T(std::move(v));
            alloc_.dealloc(old_ptr);
        } else {
            new (&ptr_[n_]) T(std::move(v));
        }

        return ptr_[n_++];
    }

    CountT uninit_back()
    {
        if (n_ == capacity_) [[unlikely]] {
            T *old_ptr = expand(n_ + 1);
            alloc_.dealloc(old_ptr);
        }

        return n_++;
    }

    void pop_back()
    {
        ptr_[--n_].~T();
    }

    RefT operator[](CountT idx) { return ptr_[idx]; }
    const RefT operator[](CountT idx) const { return ptr_[idx]; }

    T *data() { return ptr_; }
    const T *data() const { return ptr_; }

    T *begin() { return ptr_; }
    T *end() { return ptr_ + n_; }
    const T *begin() const { return ptr_; }
    const T *end() const { return ptr_ + n_; }

    RefT front()
    {
        return ptr_[0];
    }

    RefT front() const
    {
        return ptr_[0];
    }

    RefT back()
    {
        return ptr_[n_ - 1];
    }

    RefT back() const
    {
        return ptr_[n_ - 1];
    }

    CountT size() const { return n_; }

private:
    [[nodiscard]] T * expand(CountT new_size)
    {
        CountT new_capacity = capacity_ * expansion_factor_;
        new_capacity = std::max(new_capacity, new_size);

        return realloc(new_capacity);
    }

    [[nodiscard]] T * realloc(CountT new_capacity)
    {
        auto new_ptr = (T *)alloc_.alloc(new_capacity * sizeof(T));

        if (ptr_) {
#ifdef MADRONA_GCC
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wclass-memaccess"
#endif
            memcpy(new_ptr, ptr_, sizeof(T) * n_);
#if MADRONA_GCC
#pragma GCC diagnostic pop
#endif
        }

        T *old_ptr = ptr_;

        ptr_ = new_ptr;
        capacity_ = new_capacity;

        return old_ptr;
    }

    [[no_unique_address]] A alloc_;
    T *ptr_;
    CountT n_;
    CountT capacity_;

    static constexpr CountT expansion_factor_ = 2;
};

}
