add_subdirectory(external/glslang EXCLUDE_FROM_ALL)

target_include_directories(glslang SYSTEM
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/external/glslang/glslang/Public>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/external/glslang/StandAlone>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/external/glslang/SPIRV>
)

add_library(spirv_reflect STATIC
    external/SPIRV-Reflect/spirv_reflect.h external/SPIRV-Reflect/spirv_reflect.c)
target_include_directories(spirv_reflect
    INTERFACE external/SPIRV-Reflect)

# Build vulkan dispatch table
get_directory_property(DIR_DEFINES DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} COMPILE_DEFINITIONS )
set(DEFINE_STR "")
foreach (f ${DIR_DEFINES})
    list(APPEND DEFINE_STR "-D${f}")
endforeach()

file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/dispatch")

add_custom_command(
    OUTPUT dispatch/dispatch.template
    COMMAND ${CMAKE_CXX_COMPILER} -x c -E -P -nostdinc ${DEFINE_STR} "${CMAKE_CURRENT_SOURCE_DIR}/dispatch.template" -o dispatch/dispatch.template
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/dispatch.template
)

add_custom_command(
    OUTPUT dispatch/dispatch_device_impl.hpp dispatch/dispatch_device_impl.cpp
           dispatch/dispatch_instance_impl.hpp dispatch/dispatch_instance_impl.cpp
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/scripts/gen_vk_dispatch.py
            ${CMAKE_CURRENT_BINARY_DIR}/dispatch/dispatch.template dispatch
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/dispatch/dispatch.template
            ${CMAKE_CURRENT_SOURCE_DIR}/scripts/gen_vk_dispatch.py
)

add_custom_target( generate_vk_dispatch DEPENDS
    dispatch/dispatch_device_impl.hpp dispatch/dispatch_device_impl.cpp
    dispatch/dispatch_instance_impl.hpp dispatch/dispatch_instance_impl.cpp
)

add_library(madrona_render_vk STATIC
    config.hpp
    core.hpp core.cpp
    cuda_interop.hpp cuda_interop.cpp
    descriptors.hpp descriptors.cpp descriptors.inl
    dispatch.hpp dispatch.cpp
    shader.hpp shader.cpp
    utils.hpp utils.cpp utils.inl
    memory.hpp memory.cpp memory.inl
    scene.hpp scene.cpp
    present.hpp present.cpp
)

target_compile_definitions(madrona_render_vk
    PUBLIC
        "-DVK_NO_PROTOTYPES"
)

target_include_directories(madrona_render_vk
    PUBLIC
        "${CMAKE_CURRENT_BINARY_DIR}/dispatch"
    PRIVATE
        $<TARGET_PROPERTY:madrona_sdl2,INTERFACE_INCLUDE_DIRECTORIES>
)

target_link_libraries(madrona_render_vk
    PUBLIC
        madrona_vk_hdrs
    PRIVATE
        CUDA::cudart
        glslang
        spirv_reflect
        SPIRV
        glslang-default-resource-limits
        madrona_common
        madrona_sdl2
)

add_dependencies(madrona_render_vk generate_vk_dispatch)
