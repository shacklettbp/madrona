set(MADRONA_INC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../include/madrona"
    CACHE INTERNAL "")

set(FRONTEND_CLANG FALSE)
set(FRONTEND_GCC FALSE)
set(FRONTEND_MSVC FALSE)

if (CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
    set(FRONTEND_CLANG TRUE)
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set(FRONTEND_GCC TRUE)
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    set(FRONTEND_MSVC TRUE)
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    if (CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC")
        set(FRONTEND_MSVC TRUE)
    else ()
        set(FRONTEND_CLANG TRUE)
    endif ()
endif ()

add_library(madrona_hdrs INTERFACE)
target_include_directories(madrona_hdrs INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
)
target_link_libraries(madrona_hdrs INTERFACE
    madrona_sys_defns)

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
        -pedantic -Wall -Wextra -march=x86-64-v3
    )
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
    #${MADRONA_INC_DIR}/hash.hpp
    #${INC_DIR}/platform_utils.hpp ${INC_DIR}/platform_utils.inl
    #    platform_utils.cpp
)

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
