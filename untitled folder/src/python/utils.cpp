#include <madrona/py/utils.hpp>
#include <madrona/heap_array.hpp>

#ifdef MADRONA_CUDA_SUPPORT
#include <madrona/cuda_utils.hpp>
#endif

#include <cassert>
#include <cstring>
#include <cstdio>
#include <string>

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
#else
        (void)num_bytes_per_item_;
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

Tensor::Tensor(const Tensor &o)
    : dev_ptr_(o.dev_ptr_),
      type_(o.type_),
      gpu_id_(o.gpu_id_),
      num_dimensions_(o.num_dimensions_),
      dimensions_(o.dimensions_)
{}

Tensor::Printer Tensor::makePrinter() const
{
    int64_t num_items = dimensions_[num_dimensions_ - 1];
    for (int64_t i = num_dimensions_ - 2; i >= 0; i--) {
        num_items *= dimensions_[i];
    }
    int64_t bytes_per_item = numBytesPerItem();

    void *print_ptr;
    if (!isOnGPU()) {
        print_ptr = nullptr;
    } else {
#ifdef MADRONA_CUDA_SUPPORT
        int64_t num_bytes = bytes_per_item * num_items;
        print_ptr = cu::allocReadback(num_bytes);
#else
        print_ptr = nullptr;
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

struct TrainInterface::Impl {
    HeapArray<NamedTensor> obs;
    Tensor actions;
    Tensor rewards;
    Tensor dones;
    Tensor resets;
    Optional<Tensor> policyAssignments;

    HeapArray<NamedTensor> stats;
    HeapArray<std::string> nameStrings;
};

TrainInterface::TrainInterface(
        std::initializer_list<NamedTensor> obs,
        Tensor actions,
        Tensor rewards,
        Tensor dones,
        Tensor resets,
        Optional<Tensor> policy_assignments,
        std::initializer_list<NamedTensor> stats)
    : impl_(new Impl {
        .obs = HeapArray<NamedTensor>(obs.size()),
        .actions = actions,
        .rewards = rewards,
        .dones = dones,
        .resets = resets,
        .policyAssignments = policy_assignments,
        .stats = HeapArray<NamedTensor>(stats.size()),
        .nameStrings = HeapArray<std::string>(obs.size() + stats.size()),
    })
{
    const NamedTensor *src_obs = std::data(obs);
    const NamedTensor *src_stats = std::data(stats);

    CountT cur_str_idx = 0;
    for (CountT i = 0; i < (CountT)obs.size(); i++) {
        impl_->nameStrings.emplace(cur_str_idx, src_obs[i].name);
        impl_->obs.emplace(i, NamedTensor {
            impl_->nameStrings[cur_str_idx].c_str(),
            src_obs[i].hdl,
        });

        cur_str_idx += 1;
    }

    for (CountT i = 0; i < (CountT)stats.size(); i++) {
        impl_->nameStrings.emplace(cur_str_idx, src_stats[i].name);
        impl_->stats.emplace(i, NamedTensor {
            impl_->nameStrings[cur_str_idx].c_str(),
            src_stats[i].hdl,
        });

        cur_str_idx += 1;
    }
}

TrainInterface::TrainInterface(TrainInterface &&) = default;
TrainInterface::~TrainInterface() = default;

Span<const TrainInterface::NamedTensor> TrainInterface::observations() const
{
    return Span<const NamedTensor>(
        impl_->obs.data(), impl_->obs.size());
}

Tensor TrainInterface::actions() const
{
    return impl_->actions;
}

Tensor TrainInterface::rewards() const
{
    return impl_->rewards;
}

Tensor TrainInterface::dones() const
{
    return impl_->dones;
}

Tensor TrainInterface::resets() const
{
    return impl_->resets;
}

Optional<Tensor> TrainInterface::policyAssignments() const
{
    return impl_->policyAssignments;
}

Span<const TrainInterface::NamedTensor> TrainInterface::stats() const
{
    return Span<const NamedTensor>(
        impl_->stats.data(), impl_->stats.size());
}

#ifdef MADRONA_LINUX
void PyExecMode::key_() {}
void Tensor::key_() {}
void TrainInterface::key_() {}
#endif

}
