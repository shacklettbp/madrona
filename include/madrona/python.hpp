#pragma once

#include <madrona/macros.hpp>
#include <madrona/span.hpp>
#include <madrona/optional.hpp>

#include <array>
#include <cuda_runtime.h>

namespace madrona {
namespace py {

class ExternalSync {
public:
    MADRONA_IMPORT ExternalSync(cudaExternalSemaphore_t sema);

    MADRONA_IMPORT void wait(uint64_t strm);

private:
    cudaExternalSemaphore_t sema_;
};

class Tensor {
public:
    enum class ElementType {
        UInt8,
        Int8,
        Int16,
        Int32,
        Int64,
        Float16,
        Float32,
    };

    MADRONA_IMPORT Tensor(void *dev_ptr, ElementType type,
                          Span<const int64_t> dimensions,
                          Optional<int> gpu_id);
    
    inline void * devicePtr() const { return dev_ptr_; }
    inline ElementType type() const { return type_; }
    inline bool isOnGPU() const { return gpu_id_ != -1; }
    inline int gpuID() const { return gpu_id_; }
    inline int64_t numDims() const { return num_dimensions_; }
    inline const int64_t *dims() const { return dimensions_.data(); }

    static inline constexpr int64_t maxDimensions = 16;
private:
    void *dev_ptr_;
    ElementType type_;
    int gpu_id_;

    int64_t num_dimensions_;
    std::array<int64_t, maxDimensions> dimensions_;
};

}
}
