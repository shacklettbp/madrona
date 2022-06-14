#include "cuda_utils.hpp"

namespace madrona {
namespace cu {

void printCudaError(cudaError_t res, const char *msg)
{
    std::cerr << msg << ": " << cudaGetErrorString(res) << std::endl;
}

}
}
