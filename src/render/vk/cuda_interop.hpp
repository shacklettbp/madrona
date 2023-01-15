#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include "core.hpp"

namespace madrona {
namespace render {
namespace vk {

class CudaImportedBuffer {
public:
    CudaImportedBuffer(const DeviceState &dev,
                       int cuda_id,
                       VkDeviceMemory mem,
                       uint64_t num_bytes,
                       bool dedicated = true);

    CudaImportedBuffer(const CudaImportedBuffer &) = delete;
    CudaImportedBuffer(CudaImportedBuffer &&);
    ~CudaImportedBuffer();

    void *getDevicePointer() const { return dev_ptr_; }

private:
    int ext_fd_;
    cudaExternalMemory_t ext_mem_;
    void *dev_ptr_;
};

DeviceUUID getUUIDFromGPUID(int gpu_id);

}
}
}
