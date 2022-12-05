#include <madrona/python.hpp>
#include <madrona/crash.hpp>
#include <madrona/cuda_utils.hpp>

#include <cassert>
#include <cstring>

namespace madrona {
namespace py {

MADRONA_EXPORT ExternalSync::ExternalSync(cudaExternalSemaphore_t sema)
    : sema_(sema)
{}

MADRONA_EXPORT void ExternalSync::wait(uint64_t strm)
{
    // Get the current CUDA stream from pytorch and force it to wait
    // on an external semaphore to finish
    cudaStream_t cuda_strm = (cudaStream_t)strm;
    cudaExternalSemaphoreWaitParams params {};
    REQ_CUDA(cudaWaitExternalSemaphoresAsync(&sema_, &params, 1, cuda_strm));
}

MADRONA_EXPORT GPUTensor::GPUTensor(void *dev_ptr, ElementType type,
                                    Span<const int64_t> dimensions, int gpu_id)
    : dev_ptr_(dev_ptr),
      type_(type),
      gpu_id_(gpu_id),
      num_dimensions_(dimensions.size()),
      dimensions_()
{
    assert(num_dimensions_ <= maxDimensions);
    memcpy(dimensions_.data(), dimensions.data(),
           num_dimensions_ * sizeof(int64_t));
}

}
}
