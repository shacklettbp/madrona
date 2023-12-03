#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include <madrona/render/vk/backend.hpp>

namespace madrona::render::vk {

class CudaImportedBuffer {
public:
    CudaImportedBuffer(const Device &dev,
                       VkDeviceMemory mem,
                       uint64_t num_bytes,
                       bool dedicated = true);

    CudaImportedBuffer(const CudaImportedBuffer &) = delete;
    CudaImportedBuffer(CudaImportedBuffer &&);
    ~CudaImportedBuffer();

    inline void *getDevicePointer() const { return dev_ptr_; }

private:
    int ext_fd_;
    cudaExternalMemory_t ext_mem_;
    void *dev_ptr_;
};

}
