set(CMAKE_THREAD_PREFER_PTHREAD TRUE)
set(THREADS_PREFER_PTHREAD_FLAG TRUE)
find_package(Threads REQUIRED)

if (${MADRONA_REQUIRE_CUDA})
    set(CUDA_REQUIRED_ARG REQUIRED)

    if (MADRONA_DISABLE_CUDA)
        message(FATAL_ERROR "CUDA set as required build MADRONA_DISABLE_CUDA set to ${MADRONA_DISABLE_CUDA}")
    endif()
else()
    set(CUDA_REQUIRED_ARG QUIET)
endif()

if (MADRONA_DISABLE_CUDA)
    set(CUDAToolkit_FOUND FALSE)
else ()
    find_package(CUDAToolkit ${CUDA_REQUIRED_ARG})
    find_library(CUDA_NVJITLINK_LIBRARY nvJitLink
        PATHS
            ${CUDAToolkit_LIBRARY_DIR}
        ${CUDA_REQUIRED_ARG}
    )
endif ()

if (${MADRONA_REQUIRE_PYTHON})
    set(PYTHON_REQUIRED_ARG REQUIRED)
else()
    set(PYTHON_REQUIRED_ARG)
endif()

find_package(Python 3.9 COMPONENTS Interpreter Development.Module
    ${PYTHON_REQUIRED_ARG})
