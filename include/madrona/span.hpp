#pragma once

#include <array>
#include <cstdint>
#include <initializer_list>

namespace madrona {

template <typename T>
class Span {
public:
    Span(T *ptr, uint32_t num_elems)
        : ptr_(ptr), n_(num_elems)
    {}

    Span(std::initializer_list<T> init)
        : ptr_(init.begin()), n_(init.size())
    {}

    constexpr T * data() const { return ptr_; }

    constexpr uint32_t size() const { return n_; }

    T & operator[](uint32_t idx) const { return ptr_[idx]; }

    T * begin() const { return ptr_; }
    T * end() const { return ptr_ + n_; }

private:
    T *ptr_;
    uint32_t n_;
};

}
