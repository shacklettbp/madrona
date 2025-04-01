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
    HeapArray<NamedTensor> namedTensors;

    TrainStepInputInterface inputs;
    TrainStepOutputInterface outputs;
    Optional<TrainCheckpointingInterface> checkpointing;

    static inline Impl * init(
        TrainStepInputInterface inputs,
        TrainStepOutputInterface outputs,
        Optional<TrainCheckpointingInterface> checkpointing);
};

TrainInterface::Impl * TrainInterface::Impl::init(
    TrainStepInputInterface inputs,
    TrainStepOutputInterface outputs,
    Optional<TrainCheckpointingInterface> checkpointing)
{
    CountT num_total_name_chars = 0;
    CountT num_total_named_tensors = 0;

    auto sumStorageRequirements = [
        &num_total_name_chars, &num_total_named_tensors
    ](Span<const NamedTensor> tensors)
    {
        for (NamedTensor named_tensor : tensors) {
            num_total_name_chars += strlen(named_tensor.name) + 1;
        }

        num_total_named_tensors += tensors.size();
    };

    sumStorageRequirements(inputs.actions);
    sumStorageRequirements(inputs.pbt);

    sumStorageRequirements(outputs.observations);
    sumStorageRequirements(outputs.stats);
    sumStorageRequirements(outputs.pbt);

    HeapArray<char> name_buffer(num_total_name_chars);
    HeapArray<NamedTensor> named_tensors(num_total_named_tensors);

    char *cur_name_ptr = name_buffer.data();
    NamedTensor *cur_named_tensor_ptr = named_tensors.data();

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

    auto makeOwnedNamedTensors = [
        &makeOwnedName, &cur_named_tensor_ptr
    ](Span<const NamedTensor> inputs)
    {
        NamedTensor *out_start = cur_named_tensor_ptr;

        for (const NamedTensor &named_in : inputs) {
            const char *owned_name = makeOwnedName(named_in.name);
            *(cur_named_tensor_ptr++) = NamedTensor {
                .name = owned_name,
                .tensor = named_in.tensor,
            };
        }

        return Span<const NamedTensor>(out_start, inputs.size());
    };

    TrainStepInputInterface owned_inputs {
        .actions = makeOwnedNamedTensors(inputs.actions),
        .resets = inputs.resets,
        .simCtrl = inputs.simCtrl,
        .pbt = makeOwnedNamedTensors(inputs.pbt),
    };

    TrainStepOutputInterface owned_outputs {
        .observations = makeOwnedNamedTensors(outputs.observations),
        .rewards = outputs.rewards,
        .dones = outputs.dones,
        .stats = makeOwnedNamedTensors(outputs.stats),
        .pbt = makeOwnedNamedTensors(outputs.pbt),
    };

    Optional<TrainCheckpointingInterface> owned_checkpointing =
        Optional<TrainCheckpointingInterface>::none();

    if (checkpointing.has_value()) {
        owned_checkpointing = TrainCheckpointingInterface {
            .checkpointData = checkpointing->checkpointData,
        };
    }

    assert(cur_name_ptr == name_buffer.data() + name_buffer.size());
    assert(cur_named_tensor_ptr ==
           named_tensors.data() + named_tensors.size());

    return new Impl {
        .nameBuffer = std::move(name_buffer),
        .namedTensors = std::move(named_tensors),
        .inputs = owned_inputs,
        .outputs = owned_outputs,
        .checkpointing = std::move(owned_checkpointing),
    };
}

TrainInterface::TrainInterface()
  : impl_(nullptr)
{}

TrainInterface::TrainInterface(
        TrainStepInputInterface step_inputs,
        TrainStepOutputInterface step_outputs,
        Optional<TrainCheckpointingInterface> checkpointing)
    : impl_(Impl::init(step_inputs, step_outputs, std::move(checkpointing)))
{}
    
TrainInterface::TrainInterface(TrainInterface &&) = default;
TrainInterface::~TrainInterface() = default;

TrainInterface & TrainInterface::operator=(TrainInterface &&o) = default;

TrainStepInputInterface TrainInterface::stepInputs() const
{
    return impl_->inputs;
}

TrainStepOutputInterface TrainInterface::stepOutputs() const
{
    return impl_->outputs;
}

Optional<TrainCheckpointingInterface> TrainInterface::checkpointing() const
{
    return impl_->checkpointing;
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

void Tensor::Printer::print(int64_t flatten_dim) const
{
    void *print_ptr;
    if (print_ptr_ == nullptr) {
        print_ptr = dev_ptr_;
    } else {
#ifdef MADRONA_CUDA_SUPPORT
        cudaMemcpy(print_ptr_, dev_ptr_,
                   num_total_bytes_,
                   cudaMemcpyDeviceToHost);
#else
        (void)num_total_bytes_;
        FATAL("Trying to print CUDA tensor, no CUDA support");
#endif
        print_ptr = print_ptr_;
    }

    int64_t num_inner_items = 1;
    for (int64_t i = flatten_dim + 1; i < num_dimensions_; i++) {
        int64_t dim_size = dimensions_[i];
        num_inner_items *= dim_size;
    }

    printOuterDim(0, flatten_dim, print_ptr, num_inner_items, 0);
}

int64_t Tensor::Printer::printInnerDims(void *print_ptr,
                                        int64_t num_inner_items,
                                        int64_t cur_offset) const
{
    switch (type_) {
    case TensorElementType::Int32: {
        auto base = (int32_t *)print_ptr + cur_offset;
    
        for (int64_t i = 0; i < num_inner_items; i++) {
            printf("%d ", base[i]);
        }
    } break;
    case TensorElementType::Float32: {
        auto base = (float *)print_ptr + cur_offset;
    
        for (int64_t i = 0; i < num_inner_items; i++) {
            printf("%.3f ", base[i]);
        }
    } break;
    default: break;
    }

    return cur_offset + num_inner_items;
}

int64_t Tensor::Printer::printOuterDim(int64_t dim,
                                       int64_t flatten_dim,
                                       void *print_ptr,
                                       int64_t num_inner_items,
                                       int64_t cur_offset) const
{
    int64_t dim_size = dimensions_[dim];
    if (dim == flatten_dim) {
        for (CountT i = 0; i < dim_size; i++) {
            for (int64_t j = 0; j < dim; j++) {
                printf("  ");
            }

            if (num_dimensions_ - flatten_dim > 1) {
                printf("[ ");
            }
            cur_offset = printInnerDims(
                print_ptr, num_inner_items, cur_offset);

            if (num_dimensions_ - flatten_dim > 1) {
                printf("]");
            }
            printf("\n");
        }
    } else {
        for (CountT i = 0; i < dim_size; i++) {
            for (int64_t j = 0; j < dim; j++) {
                printf("  ");
            }

            printf("[\n");
            cur_offset = printOuterDim(dim + 1, flatten_dim, print_ptr,
                                       num_inner_items, cur_offset);

            for (int64_t j = 0; j < dim; j++) {
                printf("  ");
            }
            printf("]\n");
        }
    }

    return cur_offset;
}

Tensor::Printer::Printer(void *dev_ptr,
                         void *print_ptr,
                         TensorElementType type,
                         Span<const int64_t> dimensions,
                         int64_t num_total_bytes)
    : dev_ptr_(dev_ptr),
      print_ptr_(print_ptr),
      type_(type),
      num_dimensions_(dimensions.size()),
      dimensions_(),
      num_total_bytes_(num_total_bytes)
{
    for (int64_t i = 0; i < num_dimensions_; i++) {
        dimensions_[i] = dimensions[i];
    }
}

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
    int64_t num_total_items = dimensions_[num_dimensions_ - 1];
    for (int64_t i = num_dimensions_ - 2; i >= 0; i--) {
        num_total_items *= dimensions_[i];
    }
    int64_t num_total_bytes = num_total_items * numBytesPerItem();

    void *print_ptr;
    if (!isOnGPU()) {
        print_ptr = nullptr;
    } else {
#ifdef MADRONA_CUDA_SUPPORT
        print_ptr = cu::allocReadback(num_total_bytes);
#else
        print_ptr = nullptr;
#endif
    }

    return Printer {
        dev_ptr_,
        print_ptr,
        type_,
        Span(dimensions_.data(), num_dimensions_),
        num_total_bytes,
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

#ifdef MADRONA_CUDA_SUPPORT
[[maybe_unused]] static inline  uint64_t numTensorBytes(const Tensor &t)
{
  uint64_t num_items = 1;
  uint64_t num_dims = t.numDims();
  for (uint64_t i = 0; i < num_dims; i++) {
    num_items *= t.dims()[i];
  }

  return num_items * (uint64_t)t.numBytesPerItem();
}

void ** TrainInterface::cudaCopyStepInputs(cudaStream_t strm, void **buffers)
{

  auto copyToSim = [&strm](const Tensor &dst, void *src) {
    uint64_t num_bytes = numTensorBytes(dst);

    REQ_CUDA(cudaMemcpyAsync(dst.devicePtr(), src, num_bytes,
      dst.isOnGPU() ? cudaMemcpyDeviceToDevice : cudaMemcpyDeviceToHost,
      strm));
  };

  TrainStepInputInterface &inputs = impl_->inputs;

  for (const NamedTensor &t : inputs.actions) {
    copyToSim(t.tensor, *buffers++);
  }

  copyToSim(inputs.resets, *buffers++);
  copyToSim(inputs.simCtrl, *buffers++);

  for (const NamedTensor &t : inputs.pbt) {
    copyToSim(t.tensor, *buffers++);
  }

  return buffers;
}

void TrainInterface::cudaCopyObservations(cudaStream_t strm, void **buffers)
{
  auto copyFromSim = [&strm](void *dst, const Tensor &src) {
    uint64_t num_bytes = numTensorBytes(src);

    REQ_CUDA(cudaMemcpyAsync(dst, src.devicePtr(), num_bytes,
      src.isOnGPU() ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice, strm));
  };

  for (const NamedTensor &t : impl_->outputs.observations) {
    copyFromSim(*buffers++, t.tensor);
  }
}

void TrainInterface::cudaCopyStepOutputs(cudaStream_t strm, void **buffers)
{
  auto copyFromSim = [&strm](void *dst, const Tensor &src) {
    uint64_t num_bytes = numTensorBytes(src);

    REQ_CUDA(cudaMemcpyAsync(dst, src.devicePtr(), num_bytes,
      src.isOnGPU() ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice, strm));
  };

  TrainStepOutputInterface &outputs = impl_->outputs;

  for (const NamedTensor &t : outputs.observations) {
    copyFromSim(*buffers++, t.tensor);
  }

  copyFromSim(*buffers++, outputs.rewards);
  copyFromSim(*buffers++, outputs.dones);

  for (const NamedTensor &t : outputs.stats) {
    copyFromSim(*buffers++, t.tensor);
  }

  for (const NamedTensor &t : outputs.pbt) {
    copyFromSim(*buffers++, t.tensor);
  }
}

#endif

}
