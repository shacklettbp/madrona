namespace madrona {

template <typename T>
FixedSizeQueue<T>::FixedSizeQueue(T *data, uint32_t max_size)
    : data_(data),
      max_size_(max_size),
      head_(0),
      tail_(0)
{}

template <typename T>
void FixedSizeQueue<T>::add(T t)
{
    data_[tail_] = t;
    tail_ = increment(tail_);
}

template <typename T>
T FixedSizeQueue<T>::remove()
{
    T t = data_[head_];
    head_ = increment(head_);
    return t;
}

template <typename T>
uint32_t FixedSizeQueue<T>::capacity() const
{
    return max_size_;
}

template <typename T>
bool FixedSizeQueue<T>::isEmpty() const
{
    return head_ == tail_;
}

template <typename T>
void FixedSizeQueue<T>::clear()
{
    head_ = 0;
    tail_ = 0;
}

template <typename T>
uint32_t FixedSizeQueue<T>::increment(uint32_t i)
{
    if (i == max_size_ - 1) {
        return 0;
    }

    return i + 1;
}

namespace utils {

inline uint32_t u32mulhi(uint32_t a, uint32_t b)
{
#ifdef MADRONA_GPU_MODE
    return __umulhi(a, b);
#else
    uint64_t m = uint64_t(a) * uint64_t(b);
    return uint32_t(m >> 32);
#endif
}

}

}
