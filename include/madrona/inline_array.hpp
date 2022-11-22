/*
 * Copyright 2021-2022 Brennan Shacklett and contributors
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 */
#pragma once

#include <cstddef>
#include <type_traits>
#include <utility>

namespace madrona {

template <typename T, size_t N>
class InlineArray {
public:
    InlineArray()
        : b_(),
          size_(0)
    {}

    ~InlineArray()
    {
        if constexpr (!std::is_trivially_destructible_v<T>) {
            for (size_t i = 0; i < size_; i++) {
                arr_[i].~T();
            }
        }
    }

    T & push_back(T &&v)
    {
        return *new (&arr_[size_++]) T(std::forward<T>(v));
    }

    T & push_back(const T &v)
    {
        return *new (&arr_[size_++]) T(v);
    }

    template <typename ...Args>
    T & emplace_back(Args && ...args)
    {
        return *new (&arr_[size_++]) T(std::forward<Args>(args)...);
    }

    void pop_back()
    {
        arr_[size_--].~T();
    }

    T & operator[](size_t idx) { return arr_[idx]; }
    const T & operator[](size_t idx) const { return arr_[idx]; }

    T * begin() { return data(); }
    T * end() { return data() + size_; }

    const T * begin() const { return data(); }
    const T * end() const { return data() + size_; }

    T * data() { return arr_; }
    const T * data() const { return arr_; }

    size_t size() const { return size_; }

    constexpr size_t capacity() const { return N; }

    void clear()
    {
        if constexpr (!std::is_trivially_destructible_v<T>) {
            for (size_t i = 0; i < size_; i++) {
                arr_[i].~T();
            }
        }

        size_ = 0;
    }

private:
    union {
        bool b_;
        T arr_[N];
    };
    size_t size_;
};

template <typename T, size_t N>
class FixedInlineArray {
public:
    FixedInlineArray()
        : b_(),
          size_(0)
    {}

    ~FixedInlineArray()
    {
        if constexpr (!std::is_trivially_destructible_v<T>) {
            for (size_t i = 0; i < N; i++) {
                arr_[i].~T();
            }
        }
    }

    template <typename ...Args>
    T & emplace(size_t i, Args && ...args)
    {
        return *new (&arr_[i]) T(std::forward<Args>(args)...);
    }

    T & operator[](size_t idx) { return arr_[idx]; }
    const T & operator[](size_t idx) const { return arr_[idx]; }

    T * begin() { return data(); }
    T * end() { return data() + N; }

    const T * begin() const { return data(); }
    const T * end() const { return data() + N; }

    T * data() { return arr_; }
    const T * data() const { return arr_; }

    constexpr size_t size() const { return N; }

private:
    union {
        bool b_;
        T arr_[N];
    };
    size_t size_;
};

}
