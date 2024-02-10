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

struct TrainInterface::Impl {
    HeapArray<char> nameBuffer;
    HeapArray<int64_t> dimsBuffer;
    HeapArray<NamedTensorInterface> namedInterfaces;

    TrainStepInputInterface inputs;
    TrainStepOutputInterface outputs;

    static inline Impl * init(TrainStepInputInterface inputs,
                              TrainStepOutputInterface outputs);
};

TrainInterface::Impl * TrainInterface::Impl::init(
    TrainStepInputInterface inputs,
    TrainStepOutputInterface outputs)
{
    CountT num_total_name_chars = 0;
    CountT num_total_dims = 0;
    CountT num_total_interfaces = 0;

    auto sumStorageRequirements = [
        &num_total_dims, &num_total_name_chars, &num_total_interfaces
    ](Span<const NamedTensorInterface> tensors)
    {
        for (NamedTensorInterface named_iface : tensors) {
            num_total_name_chars += strlen(named_iface.name) + 1;
            num_total_dims += named_iface.interface.dimensions.size();
        }

        num_total_interfaces += tensors.size();
    };

    num_total_dims += inputs.actions.dimensions.size();
    num_total_dims += inputs.resets.dimensions.size();
    sumStorageRequirements(inputs.pbt);

    sumStorageRequirements(outputs.observations);
    num_total_dims += outputs.rewards.dimensions.size();
    num_total_dims += outputs.dones.dimensions.size();
    sumStorageRequirements(outputs.stats);
    sumStorageRequirements(outputs.pbt);

    HeapArray<char> name_buffer(num_total_name_chars);
    HeapArray<int64_t> dims_buffer(num_total_dims);
    HeapArray<NamedTensorInterface> named_interfaces(num_total_interfaces);

    char *cur_name_ptr = name_buffer.data();
    int64_t *cur_dims_ptr = dims_buffer.data();
    NamedTensorInterface *cur_iface_ptr = named_interfaces.data();

    auto makeOwnedName = [
        &cur_name_ptr
    ](const char *in_name)
    {
        char *out_name = cur_name_ptr;

        size_t name_len = strlen(in_name) + 1;
        memcpy(out_name, in_name, name_len);
        cur_name_ptr += name_len;

        return out_name;
    };

    auto makeOwnedDims = [
        &cur_dims_ptr
    ](Span<const int64_t> in_dims)
    {
        Span out_dims(cur_dims_ptr, in_dims.size());

        utils::copyN<int64_t>(cur_dims_ptr, in_dims.data(), in_dims.size());
        cur_dims_ptr += in_dims.size();

        return out_dims;
    };

    auto makeOwnedInterface = [
        &makeOwnedDims
    ](TensorInterface in_iface)
    {
        return TensorInterface {
            .type = in_iface.type,
            .dimensions = makeOwnedDims(in_iface.dimensions),
        };
    };

    auto makeOwnedNamedInterfaces = [
        &makeOwnedName, &makeOwnedInterface, &cur_iface_ptr
    ](Span<const NamedTensorInterface> inputs)
    {
        NamedTensorInterface *out_start = cur_iface_ptr;

        for (NamedTensorInterface named_in : inputs) {
            *(cur_iface_ptr++) = {
                .name = makeOwnedName(named_in.name),
                .interface = makeOwnedInterface(named_in.interface),
            };
        }

        return Span<const NamedTensorInterface>(out_start, inputs.size());
    };

    TrainStepInputInterface owned_inputs {
        .actions = makeOwnedInterface(inputs.actions),
        .resets = makeOwnedInterface(inputs.resets),
        .pbt = makeOwnedNamedInterfaces(inputs.pbt),
    };

    TrainStepOutputInterface owned_outputs {
        .observations = makeOwnedNamedInterfaces(outputs.observations),
        .rewards = makeOwnedInterface(outputs.rewards),
        .dones = makeOwnedInterface(outputs.dones),
        .stats = makeOwnedNamedInterfaces(outputs.stats),
        .pbt = makeOwnedNamedInterfaces(outputs.pbt),
    };

    assert(cur_name_ptr == name_buffer.data() + name_buffer.size());
    assert(cur_dims_ptr == dims_buffer.data() + dims_buffer.size());
    assert(cur_iface_ptr == named_interfaces.data() + named_interfaces.size());

    return new Impl {
        .nameBuffer = std::move(name_buffer),
        .dimsBuffer = std::move(dims_buffer),
        .namedInterfaces = std::move(named_interfaces),
        .inputs = owned_inputs,
        .outputs = owned_outputs,
    };
}

TrainInterface::TrainInterface(
        TrainStepInputInterface step_inputs,
        TrainStepOutputInterface step_outputs)
    : impl_(Impl::init(step_inputs, step_outputs))
{}
    
TrainInterface::TrainInterface(TrainInterface &&) = default;
TrainInterface::~TrainInterface() = default;

TrainStepInputInterface TrainInterface::stepInputs() const
{
    return impl_->inputs;
}

TrainStepOutputInterface TrainInterface::stepOutputs() const
{
    return impl_->outputs;
}

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
        case TensorElementType::Int32: {
            printf("%d ", ((int32_t *)print_ptr)[i]);
        } break;
        case TensorElementType::Float32: {
            printf("%.3f ", ((float *)print_ptr)[i]);
        } break;
        default: break;
        }
    }

    printf("\n");
}

Tensor::Printer::Printer(void *dev_ptr,
                         void *print_ptr,
                         TensorElementType type,
                         int64_t num_items,
                         int64_t num_bytes_per_item)
    : dev_ptr_(dev_ptr),
      print_ptr_(print_ptr),
      type_(type),
      num_items_(num_items),
      num_bytes_per_item_(num_bytes_per_item)
{}

Tensor::Tensor(void *dev_ptr, TensorElementType type,
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

Tensor & Tensor::operator=(const Tensor &o)
{
    dev_ptr_ = o.dev_ptr_;
    type_ = o.type_;
    gpu_id_ = o.gpu_id_;
    num_dimensions_ = o.num_dimensions_;
    dimensions_ = o.dimensions_;

    return *this;
}

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
        case TensorElementType::UInt8: return 1;
        case TensorElementType::Int8: return 1;
        case TensorElementType::Int16: return 2;
        case TensorElementType::Int32: return 4;
        case TensorElementType::Int64: return 8;
        case TensorElementType::Float16: return 2;
        case TensorElementType::Float32: return 4;
        default: return 0;
    }
}

TensorInterface Tensor::interface() const
{
    return TensorInterface {
        .type = type_,
        .dimensions = Span<const int64_t>(dimensions_.data(), num_dimensions_),
    };
}

#ifdef MADRONA_LINUX
void PyExecMode::key_() {}
void Tensor::key_() {}
void TrainInterface::key_() {}
#endif

}
