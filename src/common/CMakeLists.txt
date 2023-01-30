set(MADRONA_INC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../include/madrona"
    CACHE INTERNAL "")

add_library(madrona_hdrs INTERFACE)
target_include_directories(madrona_hdrs INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
)
target_link_libraries(madrona_hdrs INTERFACE
    madrona_sys_defns madrona_libcxx)

if (FRONTEND_GCC)
    target_compile_options(madrona_hdrs INTERFACE
        -fdiagnostics-color=always  
    )
elseif (FRONTEND_CLANG)
    target_compile_options(madrona_hdrs INTERFACE
        -fcolor-diagnostics -Wshadow
    )
endif ()

if (FRONTEND_GCC OR FRONTEND_CLANG)
    target_compile_options(madrona_hdrs INTERFACE
        -pedantic -Wall -Wextra
    )

    if (MADRONA_X64)
        target_compile_options(madrona_hdrs INTERFACE
            -march=x86-64-v3
        )
    elseif (MADRONA_ARM)
        target_compile_options(madrona_hdrs INTERFACE
            -mcpu=apple-m1
        )
    endif()
endif()

add_library(madrona_common STATIC
    ${MADRONA_INC_DIR}/memory.hpp ${MADRONA_INC_DIR}/memory.inl memory.cpp
    ${MADRONA_INC_DIR}/heap_array.hpp
    ${MADRONA_INC_DIR}/span.hpp
    ${MADRONA_INC_DIR}/utils.hpp
    ${MADRONA_INC_DIR}/crash.hpp crash.cpp
    ${MADRONA_INC_DIR}/ecs.hpp ${MADRONA_INC_DIR}/ecs.inl
    ${MADRONA_INC_DIR}/type_tracker.hpp ${MADRONA_INC_DIR}/type_tracker.inl
        type_tracker.cpp
    ${MADRONA_INC_DIR}/hashmap.hpp ${MADRONA_INC_DIR}/hashmap.inl hashmap.cpp
    ${MADRONA_INC_DIR}/table.hpp ${MADRONA_INC_DIR}/table.inl table.cpp
    ${MADRONA_INC_DIR}/virtual.hpp virtual.cpp
    ${MADRONA_INC_DIR}/tracing.hpp tracing.cpp
    #${MADRONA_INC_DIR}/hash.hpp
    #${INC_DIR}/platform_utils.hpp ${INC_DIR}/platform_utils.inl
    #    platform_utils.cpp
)

option(MADRONA_ENABLE_TRACING "Enable tracing" OFF)
if (MADRONA_ENABLE_TRACING)
    target_compile_definitions(madrona_common PUBLIC
        MADRONA_TRACING=1
    )
endif()

# Disable exceptions & RTTI
if (FRONTEND_GCC OR FRONTEND_CLANG)
    target_compile_options(madrona_common PUBLIC
        -fno-exceptions -fno-rtti)
elseif (FRONTEND_MSVC)
    # Disabling exceptions in MSVC seems painful
    target_compile_options(madrona_common PUBLIC
        /GR-)
else ()
    message(FATAL_ERROR "Unsupported compiler frontend")
endif()

target_link_libraries(madrona_common
    PUBLIC 
        madrona_hdrs
)

set_property(TARGET madrona_common PROPERTY
    POSITION_INDEPENDENT_CODE TRUE)
set_property(TARGET madrona_common PROPERTY
    INTERFACE_POSITION_INDEPENDENT_CODE TRUE)

add_library(madrona_importer STATIC
    ${MADRONA_INC_DIR}/importer.hpp importer.cpp
)

target_link_libraries(madrona_importer PRIVATE
    madrona_common
    meshoptimizer
    fast_float
)

if (CUDAToolkit_FOUND)
    add_library(madrona_cuda STATIC
        ${MADRONA_INC_DIR}/cuda_utils.hpp ${MADRONA_INC_DIR}/cuda_utils.inl
            cuda_utils.cpp
    )
    
    target_compile_definitions(madrona_cuda
        PUBLIC
            MADRONA_CUDA_SUPPORT=1
    )
    
    target_link_libraries(madrona_cuda
        PUBLIC
            CUDA::cudart
        PRIVATE
            CUDA::cuda_driver CUDA::nvrtc
            madrona_common
    )
endif ()
