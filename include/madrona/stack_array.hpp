#pragma once

#include <stddef.h>
#include <type_traits>

namespace madrona {

template <typename T, size_t N>
class StackArray {
public:
    StackArray()
        : b_(),
          size_(0)
    {}

    ~StackArray()
    {
        for (size_t i = 0; i < size_; i++) {
            arr_[i].~T();
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

private:
    union {
        bool b_;
        T arr_[N];
    };
    size_t size_;
};

}
