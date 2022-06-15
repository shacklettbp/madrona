#pragma once

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
        : ptr_(init.data()), n_(init.size())
    {}

    T * data() { return ptr_; }
    const T * data() const { return ptr_; }

    uint32_t size() { return n_; }

    T & operator[](uint32_t idx) { return ptr_[idx]; }
    const T & operator[](uint32_t idx) const { return ptr_[idx]; }

private:
    T *ptr_;
    uint32_t n_;
};

}
