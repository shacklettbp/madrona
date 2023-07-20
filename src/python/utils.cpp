#include <madrona/py/utils.hpp>

#ifdef MADRONA_CUDA_SUPPORT
#include <madrona/cuda_utils.hpp>
#endif

#include <cassert>
#include <cstring>
#include <cstdio>

namespace madrona::py {

#ifdef MADRONA_CUDA_SUPPORT
CudaSync::CudaSync(cudaExternalSemaphore_t sema)
    : sema_(sema)
{}

void CudaSync::wait(uint64_t strm)
{
    // Get the current CUDA stream from pytorch and force it to wait
    // on an external semaphore to finish
    cudaStream_t cuda_strm = (cudaStream_t)strm;
    cudaExternalSemaphoreWaitParams params {};
    REQ_CUDA(cudaWaitExternalSemaphoresAsync(&sema_, &params, 1, cuda_strm));
}

#ifdef MADRONA_LINUX
void CudaSync::key_() {}
#endif
#endif

#ifdef MADRONA_LINUX
void PyExecMode::key_() {}
#endif

Tensor::Printer::Printer(Printer &&o)
    : dev_ptr_(o.dev_ptr_),
      print_ptr_(o.print_ptr_)
{
    o.print_ptr_ = nullptr;
}

Tensor::Printer::~Printer()
{
    if (print_ptr_ == nullptr) {
        return;
    }

#ifdef MADRONA_CUDA_SUPPORT
    cu::deallocCPU(print_ptr_);
#endif
}

void Tensor::Printer::print() const
{
    void *print_ptr;
    if (print_ptr_ == nullptr) {
        print_ptr = dev_ptr_;
    } else {
#ifdef MADRONA_CUDA_SUPPORT
        cudaMemcpy(print_ptr_, dev_ptr_,
                   num_items_ * num_bytes_per_item_,
                   cudaMemcpyDeviceToHost);
#endif
        print_ptr = print_ptr_;
    }

    for (int64_t i = 0; i < num_items_; i++) {
        switch (type_) {
        case ElementType::Int32: {
            printf("%d ", ((int32_t *)print_ptr)[i]);
        } break;
        case ElementType::Float32: {
            printf("%.3f ", ((float *)print_ptr)[i]);
        } break;
        default: break;
        }
    }

    printf("\n");
}

Tensor::Printer::Printer(void *dev_ptr,
                         void *print_ptr,
                         ElementType type,
                         int64_t num_items,
                         int64_t num_bytes_per_item)
    : dev_ptr_(dev_ptr),
      print_ptr_(print_ptr),
      type_(type),
      num_items_(num_items),
      num_bytes_per_item_(num_bytes_per_item)
{}

Tensor::Tensor(void *dev_ptr, ElementType type,
                              Span<const int64_t> dimensions,
                              Optional<int> gpu_id)
    : dev_ptr_(dev_ptr),
      type_(type),
      gpu_id_(gpu_id.has_value() ? *gpu_id : -1),
      num_dimensions_(dimensions.size()),
      dimensions_()
{
    assert(num_dimensions_ <= maxDimensions);
    memcpy(dimensions_.data(), dimensions.data(),
           num_dimensions_ * sizeof(int64_t));
}

Tensor::Printer Tensor::makePrinter() const
{
    int64_t num_items = dimensions_[num_dimensions_ - 1];
    for (int64_t i = num_dimensions_ - 2; i >= 0; i--) {
        num_items *= dimensions_[i];
    }
    int64_t bytes_per_item = numBytesPerItem();
    int64_t num_bytes = bytes_per_item * num_items;

    void *print_ptr;
    if (!isOnGPU()) {
        print_ptr = nullptr;
    } else {
#ifdef MADRONA_CUDA_SUPPORT
        print_ptr = cu::allocReadback(num_bytes);
#endif
    }

    return Printer {
        dev_ptr_,
        print_ptr,
        type_,
        num_items,
        bytes_per_item,
    };
}

int64_t Tensor::numBytesPerItem() const
{
    switch (type_) {
        case ElementType::UInt8: return 1;
        case ElementType::Int8: return 1;
        case ElementType::Int16: return 2;
        case ElementType::Int32: return 4;
        case ElementType::Int64: return 8;
        case ElementType::Float16: return 2;
        case ElementType::Float32: return 4;
        default: return 0;
    }
}

#ifdef MADRONA_LINUX
void Tensor::key_() {}
#endif


}
