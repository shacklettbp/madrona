unset(CMAKE_CXX_VISIBILITY_PRESET)

#add_library(glm INTERFACE)
#target_include_directories(glm SYSTEM
#    INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/glm")
#
#add_library(spirv_reflect STATIC
#    SPIRV-Reflect/spirv_reflect.h SPIRV-Reflect/spirv_reflect.c)
#target_include_directories(spirv_reflect
#    INTERFACE SPIRV-Reflect)
#
add_subdirectory(madrona-toolchain)
add_subdirectory(madrona-deps)
include(madrona-deps/cmake/sys-detect.cmake)
set(MADRONA_LINUX ${MADRONA_LINUX} PARENT_SCOPE)
set(MADRONA_MACOS ${MADRONA_MACOS} PARENT_SCOPE)
set(MADRONA_WINDOWS ${MADRONA_WINDOWS} PARENT_SCOPE)
set(MADRONA_IOS ${MADRONA_IOS} PARENT_SCOPE)
set(MADRONA_ANDROID ${MADRONA_ANDROID} PARENT_SCOPE)
set(MADRONA_X64 ${MADRONA_X64} PARENT_SCOPE)
set(MADRONA_ARM ${MADRONA_ARM} PARENT_SCOPE)

add_library(madrona_sys_defns INTERFACE)
if (MADRONA_LINUX)
    target_compile_definitions(madrona_sys_defns INTERFACE "MADRONA_LINUX=(1)")
elseif (MADRONA_WINDOWS)
    target_compile_definitions(madrona_sys_defns INTERFACE "MADRONA_WINDOWS=(1)")
elseif (MADRONA_MACOS)
    target_compile_definitions(madrona_sys_defns INTERFACE "MADRONA_MACOS=(1)")
elseif (MADRONA_IOS)
    target_compile_definitions(madrona_sys_defns INTERFACE "MADRONA_IOS=(1)")
elseif (MADRONA_ANDROID)
    target_compile_definitions(madrona_sys_defns INTERFACE "MADRONA_ANDROID=(1)")
endif ()

add_subdirectory(meshoptimizer EXCLUDE_FROM_ALL)
set_property(TARGET meshoptimizer PROPERTY
    POSITION_INDEPENDENT_CODE TRUE)

add_subdirectory(fast_float EXCLUDE_FROM_ALL)

if (MADRONA_ENABLE_TESTS)
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    add_subdirectory(googletest EXCLUDE_FROM_ALL)
endif ()

if (Python_FOUND)
    add_subdirectory(nanobind EXCLUDE_FROM_ALL)
endif ()
