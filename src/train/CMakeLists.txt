add_library(madrona_train STATIC
    cuda_utils.hpp cuda_utils.inl cuda_utils.cpp
    cpp_compile.hpp cpp_compile.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include/madrona/train.hpp train.cpp)

target_link_libraries(madrona_train PUBLIC
    madrona_core CUDA::nvrtc CUDA::cudart)

set(NVRTC_COMPILE_FLAGS -std=c++20 -default-device -rdc true
    -use_fast_math -lineinfo -extended-lambda -dopt=on)

set(GPU_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/gpu/)
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
    ${MADRONA_TRAIN_INCLUDES}
    ${MADRONA_CORE_INCLUDES}
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

set(NVRTC_OPTIONS)
foreach(f ${NVRTC_INCLUDE_DIRS})
    set(NVRTC_OPTIONS "${NVRTC_OPTIONS}\"-I${f}\", ")
endforeach()
foreach(f ${NVRTC_COMPILE_FLAGS})
    set(NVRTC_OPTIONS "${NVRTC_OPTIONS}\"${f}\", ")
endforeach()
set(NVRTC_OPTIONS "${NVRTC_OPTIONS}")

target_compile_definitions(madrona_train PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:MADRONA_NVRTC_OPTIONS=${NVRTC_OPTIONS}>
    $<$<COMPILE_LANGUAGE:CXX>:MADRONA_TRAIN_INTERNAL_CPP=${INTERNAL_SRC_LIST}>
)

#add_executable(train train.cu)
#target_link_libraries(train PRIVATE
#    madrona_train)
