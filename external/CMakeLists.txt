#add_library(glm INTERFACE)
#target_include_directories(glm SYSTEM
#    INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/glm")
#
#add_library(spirv_reflect STATIC
#    SPIRV-Reflect/spirv_reflect.h SPIRV-Reflect/spirv_reflect.c)
#target_include_directories(spirv_reflect
#    INTERFACE SPIRV-Reflect)
#
add_subdirectory(madrona-deps)
include(madrona-deps/cmake/sys-detect.cmake)
set(IS_LINUX ${IS_LINUX} PARENT_SCOPE)
set(IS_MAC ${IS_MAC} PARENT_SCOPE)
set(IS_WINDOWS ${IS_WINDOWS} PARENT_SCOPE)

add_library(madrona_sys_defns INTERFACE)
if (IS_LINUX)
    target_compile_definitions(madrona_sys_defns INTERFACE "IS_LINUX=1")
elseif (IS_WINDOWS)
    target_compile_definitions(madrona_sys_defns INTERFACE "IS_WINDOWS=1")
elseif (IS_MAC)
    target_compile_definitions(madrona_sys_defns INTERFACE "IS_MAC=1")
endif ()
