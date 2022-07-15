set(INC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../include/madrona")
add_library(madrona_core STATIC
    ${INC_DIR}/fwd.hpp
    ${INC_DIR}/memory.hpp  ${INC_DIR}/memory.inl
    ${INC_DIR}/heap_array.hpp ${INC_DIR}/span.hpp
    ${INC_DIR}/utils.hpp
    ${INC_DIR}/crash.hpp crash.cpp
    ${INC_DIR}/job.hpp ${INC_DIR}/job.inl job.cpp
    ${INC_DIR}/ecs.hpp ${INC_DIR}/ecs.inl
    ${INC_DIR}/state.hpp ${INC_DIR}/state.inl state.cpp
    ${INC_DIR}/context.hpp ${INC_DIR}/context.inl context.cpp
    #${INC_DIR}/table.hpp ${INC_DIR}/table.inl table.cpp
    #${INC_DIR}/hash.hpp
    #${INC_DIR}/platform_utils.hpp ${INC_DIR}/platform_utils.inl
    #    platform_utils.cpp
)

# Disable exceptions & RTTI
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

if (FRONTEND_GCC OR FRONTEND_CLANG)
    target_compile_options(madrona_core PUBLIC
        -fno-exceptions -fno-rtti)
elseif (FRONTEND_MSVC)
    # Disabling exceptions in MSVC seems painful
    target_compile_options(madrona_core PUBLIC
        /GR-)
else ()
    message(FATAL_ERROR "Unsupported compiler frontend")
endif()

target_link_libraries(madrona_core PUBLIC 
    madrona_sys_defns
)

target_include_directories(madrona_core
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../../include/
)

#add_library(madrona_wm STATIC
#    ${INC_DIR}/window.hpp ${INC_DIR}/window.inl window.cpp
#    ${INC_DIR}/wm.hpp ${INC_DIR}/wm.inl wm.cpp
#    ${INC_DIR}/input_button.hpp 
#)
#
#target_link_libraries(madrona_wm PUBLIC 
#    madrona_core
#    madrona_sdl2
#)
