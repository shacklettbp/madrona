set(CMAKE_THREAD_PREFER_PTHREAD TRUE)
set(THREADS_PREFER_PTHREAD_FLAG TRUE)
find_package(Threads REQUIRED)

if (${MADRONA_REQUIRE_CUDA})
    set(CUDA_REQUIRED_ARG REQUIRED)
else()
    set(CUDA_REQUIRED_ARG QUIET)
endif()

find_package(CUDAToolkit ${CUDA_REQUIRED_ARG})
find_library(CUDA_NVJITLINK_LIBRARY nvJitLink_static
    PATHS
        ${CUDAToolkit_LIBRARY_DIR}
    ${CUDA_REQUIRED_ARG}
)

find_library(CUDA_PTXCOMPILER_LIBRARY nvptxcompiler_static
    PATHS
        ${CUDAToolkit_LIBRARY_DIR}
    ${CUDA_REQUIRED_ARG}
)

if (${MADRONA_REQUIRE_PYTHON})
    set(PYTHON_REQUIRED_ARG REQUIRED)
else()
    set(PYTHON_REQUIRED_ARG)
endif()

find_package(Python 3.9 COMPONENTS Interpreter Development.Module
    ${PYTHON_REQUIRED_ARG})
