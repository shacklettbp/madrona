enable_language(CUDA)

add_library(madrona_train STATIC
    cuda_utils.hpp cuda_utils.inl cuda_utils.cpp)

add_executable(train train.cu)
target_link_libraries(train PRIVATE
    madrona_train CUDA::nvrtc CUDA::cudart)
