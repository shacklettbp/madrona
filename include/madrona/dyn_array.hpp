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

    DynArray(size_t init_capacity, A alloc = DefaultAlloc())
        : alloc_(std::move(alloc)),
          ptr_((T *)alloc_.alloc(init_capacity * sizeof(T))),
          n_(0),
          capacity_(init_capacity)
    {}

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

        clear();
        alloc_.dealloc(ptr_);
    }

    DynArray & operator=(const DynArray &) = delete;
    DynArray & operator=(DynArray &&o)
    {
        clear();
        alloc_.dealloc(ptr_);

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
            for (size_t i = 0; i < n_; i++) {
                ptr_[i].~T();
            }
        }
        n_ = 0;
    }

    template <typename Fn>
    void resize(size_t new_size, Fn &&fn)
    {
        if (new_size > capacity_) {
            expand(new_size);
            
            for (size_t i = n_; i < new_size; i++) {
                fn(&ptr_[i]);
            }
        } else {
            if constexpr (!std::is_trivially_destructible_v<T>) {
                for (size_t i = new_size; i < n_; i++) {
                    ptr_[i].~T();
                }
            }
        }

        n_ = new_size;
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

    template <typename... Args>
    RefT emplace_back(Args &&...args)
    {
        if (n_ == capacity_) [[unlikely]] {
            expand(n_ + 1);
        }

        new (&ptr_[n_]) T(std::forward<Args>(args)...);

        return ptr_[n_++];
    }

    RefT push_back(const T &v)
    {
        if (n_ == capacity_) [[unlikely]] {
            expand(n_ + 1);
        }

        new (&ptr_[n_]) T(v);

        return ptr_[n_++];
    }

    RefT push_back(T &&v)
    {
        if (n_ == capacity_) [[unlikely]] {
            expand(n_ + 1);
        }

        new (&ptr_[n_]) T(std::move(v));

        return ptr_[n_++];
    }

    void pop_back()
    {
        ptr_[--n_].~T();
    }

    RefT operator[](size_t idx) { return ptr_[idx]; }
    const RefT operator[](size_t idx) const { return ptr_[idx]; }

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

    size_t size() const { return n_; }

private:
    void expand(size_t new_size)
    {
        size_t new_capacity = capacity_ * expansion_factor_;
        new_capacity = std::max(new_capacity, new_size);

        auto new_ptr = (T *)alloc_.alloc(new_capacity * sizeof(T));

#ifdef MADRONA_GCC
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wclass-memaccess"
#endif
        memcpy(new_ptr, ptr_, sizeof(T) * n_);
#if MADRONA_GCC
#pragma GCC diagnostic pop
#endif

        alloc_.dealloc(ptr_);

        ptr_ = new_ptr;
        capacity_ = new_capacity;
    }

    [[no_unique_address]] A alloc_;
    T *ptr_;
    size_t n_;
    size_t capacity_;

    static constexpr size_t expansion_factor_ = 2;
};

}
