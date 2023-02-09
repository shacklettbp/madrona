#define MADRONA_PYTHON_VISIBILITY MADRONA_EXPORT
#include <madrona/python.hpp>

#ifdef MADRONA_CUDA_SUPPORT
#include <madrona/cuda_utils.hpp>
#endif

#include <cassert>
#include <cstring>
#include <typeindex>

namespace madrona::py {

#ifdef MADRONA_CUDA_SUPPORT
MADRONA_EXPORT CudaSync::CudaSync(cudaExternalSemaphore_t sema)
    : sema_(sema)
{}

MADRONA_EXPORT void CudaSync::wait(uint64_t strm)
{
    // Get the current CUDA stream from pytorch and force it to wait
    // on an external semaphore to finish
    cudaStream_t cuda_strm = (cudaStream_t)strm;
    cudaExternalSemaphoreWaitParams params {};
    REQ_CUDA(cudaWaitExternalSemaphoresAsync(&sema_, &params, 1, cuda_strm));
}

#ifdef MADRONA_LINUX
MADRONA_EXPORT void CudaSync::key_() {}
#endif
#endif

MADRONA_EXPORT Tensor::Tensor(void *dev_ptr, ElementType type,
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

#ifdef MADRONA_LINUX
MADRONA_EXPORT void Tensor::key_() {}
#endif

}
