#include "cuda_interop.hpp"

#include "utils.hpp"

#include <iostream>
#include <cstring>
#include <unistd.h>

using namespace std;

namespace madrona {
namespace render {
namespace vk {

static void setActiveDevice(int cuda_id)
{
    cudaError_t res = cudaSetDevice(cuda_id);

    if (res != cudaSuccess) {
        FATAL("CUDA failed to set device");
    }
}

static cudaExternalMemory_t importBuffer(int buf_fd, uint64_t num_bytes,
                                         bool dedicated)
{
    cudaExternalMemoryHandleDesc cuda_ext_info {};
    cuda_ext_info.type = cudaExternalMemoryHandleTypeOpaqueFd;
    cuda_ext_info.handle.fd = buf_fd;
    cuda_ext_info.size = num_bytes;
    cuda_ext_info.flags = dedicated ? cudaExternalMemoryDedicated : 0;

    cudaExternalMemory_t ext_mem;
    cudaError_t res = cudaImportExternalMemory(&ext_mem, &cuda_ext_info);

    if (res != cudaSuccess) {
        FATAL("CUDA failed to import vulkan buffer");
    }

    return ext_mem;
}

static void *mapExternal(cudaExternalMemory_t ext_mem, uint64_t num_bytes)
{
    void *dev_ptr;
    cudaExternalMemoryBufferDesc ext_info;
    ext_info.offset = 0;
    ext_info.size = num_bytes;
    ext_info.flags = 0;

    cudaError_t res =
        cudaExternalMemoryGetMappedBuffer(&dev_ptr, ext_mem, &ext_info);
    if (res != cudaSuccess) {
        FATAL("CUDA failed to map vulkan buffer");
    }

    return dev_ptr;
}

CudaImportedBuffer::CudaImportedBuffer(const Device &dev,
                                       int cuda_id,
                                       VkDeviceMemory mem,
                                       uint64_t num_bytes,
                                       bool dedicated)
    : ext_fd_(),
      ext_mem_(),
      dev_ptr_()
{
    setActiveDevice(cuda_id);

    VkMemoryGetFdInfoKHR fd_info;
    fd_info.sType = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR;
    fd_info.pNext = nullptr;
    fd_info.memory = mem;
    fd_info.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;

    REQ_VK(dev.dt.getMemoryFdKHR(dev.hdl, &fd_info, &ext_fd_));

    ext_mem_ = importBuffer(ext_fd_, num_bytes, dedicated);
    dev_ptr_ = mapExternal(ext_mem_, num_bytes);
}

CudaImportedBuffer::CudaImportedBuffer(CudaImportedBuffer &&o)
    : ext_fd_(o.ext_fd_),
      ext_mem_(o.ext_mem_),
      dev_ptr_(o.dev_ptr_)
{
    o.dev_ptr_ = nullptr;
}

CudaImportedBuffer::~CudaImportedBuffer()
{
    if (!dev_ptr_) return;

    cudaFree(dev_ptr_);
    cudaDestroyExternalMemory(ext_mem_);
    close(ext_fd_);
}

DeviceID getVkUUIDFromCudaID(int gpu_id)
{
    DeviceID vk_id;

    int device_count;
    cudaError_t res = cudaGetDeviceCount(&device_count);
    if (res != cudaSuccess) {
        FATAL("CUDA failed to enumerate devices");
    }

    if (device_count <= gpu_id) {
        FATAL("%d is not a valid CUDA ID given %d supported device\n",
              gpu_id, device_count);
    }

    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, gpu_id);

    if (props.computeMode == cudaComputeModeProhibited) {
        FATAL("%d corresponds to a prohibited device\n", gpu_id);
    }

    memcpy(vk_id.data(), &props.uuid,
           sizeof(DeviceID::value_type) * vk_id.size());

    return vk_id;
}

}
}
}
