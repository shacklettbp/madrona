add_library(madrona_train STATIC
    cuda_utils.hpp cuda_utils.inl cuda_utils.cpp
    cpp_compile.hpp cpp_compile.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include/madrona/train.hpp train.cpp)

target_link_libraries(madrona_train PUBLIC
    madrona_core CUDA::nvrtc CUDA::cudart)

file(GLOB CUDA_DEVICE_RUNTIME "${CUDAToolkit_LIBRARY_DIR}/*cudadevrt*")

set(MADRONA_TRAIN_GPU_COMPILE_FLAGS
    -std=c++20
    -default-device
    -rdc=true
    -use_fast_math
    -lineinfo
    -dopt=on
    -DMADRONA_TRAIN_MODE=1
)

set(GPU_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/gpu)
set(TRAIN_GPU_SOURCES
    ${GPU_SRC_DIR}/job.cpp
)
    
set(INTERNAL_SRC_LIST)
foreach(f ${TRAIN_GPU_SOURCES})
    set(INTERNAL_SRC_LIST "${INTERNAL_SRC_LIST}\"${f}\", ")
endforeach()
set(INTERNAL_SRC_LIST "${INTERNAL_SRC_LIST}")

get_target_property(MADRONA_CORE_INCLUDES madrona_core INCLUDE_DIRECTORIES)
get_target_property(MADRONA_TRAIN_INCLUDES madrona_train INCLUDE_DIRECTORIES)

set(NVRTC_INCLUDE_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}/gpu
    ${CMAKE_CURRENT_SOURCE_DIR}/gpu/overrides
    ${MADRONA_TRAIN_INCLUDES}
    ${MADRONA_CORE_INCLUDES}
    ${CUDAToolkit_INCLUDE_DIRS}
)

set(NVRTC_OPTIONS)
foreach(f ${NVRTC_INCLUDE_DIRS})
    set(NVRTC_OPTIONS "${NVRTC_OPTIONS}\"-I${f}\", ")
endforeach()
foreach(f ${MADRONA_TRAIN_GPU_COMPILE_FLAGS})
    set(NVRTC_OPTIONS "${NVRTC_OPTIONS}\"${f}\", ")
endforeach()
set(NVRTC_OPTIONS "${NVRTC_OPTIONS}")

target_compile_definitions(madrona_train PRIVATE
    MADRONA_NVRTC_OPTIONS=${NVRTC_OPTIONS}
    MADRONA_TRAIN_INTERNAL_CPP=${INTERNAL_SRC_LIST}
    MADRONA_CUDADEVRT_PATH=\"${CUDA_DEVICE_RUNTIME}\"
)

# Helper function for library users to get list of sources / compileflags
function(madrona_build_compile_defns OUT_TARGET_NAME OUT_SRC_DEFN OUT_FLAG_DEFN)
    set(ALL_TARGET_SOURCES "")
    set(ALL_TARGET_INCLUDES "")
    foreach(t ${ARGN})
        get_target_property(TARGET_DIR ${t} SOURCE_DIR) 

        get_target_property(TARGET_SOURCES ${t} SOURCES) 
        list(FILTER TARGET_SOURCES INCLUDE REGEX ".+\.cpp")
        list(TRANSFORM TARGET_SOURCES PREPEND "${TARGET_DIR}/")
        list(APPEND ALL_TARGET_SOURCES ${TARGET_SOURCES})

        get_target_property(TARGET_INCLUDES ${t} INCLUDE_DIRECTORIES) 
        list(APPEND ALL_TARGET_INCLUDES ${TARGET_INCLUDES})
    endforeach()

    set(SRC_LIST)
    foreach(src ${ALL_TARGET_SOURCES})
        set(SRC_LIST "${SRC_LIST}\"${src}\", ")
    endforeach()
    set(SRC_LIST "${SRC_LIST}")

    set(INCLUDE_LIST)
    foreach(i ${ALL_TARGET_INCLUDES})
        set(INCLUDE_LIST "${INCLUDE_LIST}\"-I${i}\", ")
    endforeach()
    set(INCLUDE_LIST "${INCLUDE_LIST}")

    add_library(${OUT_TARGET_NAME} INTERFACE)

    target_compile_definitions(${OUT_TARGET_NAME} INTERFACE
        ${OUT_SRC_DEFN}=${SRC_LIST}
        ${OUT_FLAG_DEFN}=${INCLUDE_LIST}
    )
endfunction()
