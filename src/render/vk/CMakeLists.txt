# Build vulkan dispatch table
get_directory_property(DIR_DEFINES DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} COMPILE_DEFINITIONS )
set(DEFINE_STR "")
if (FRONTEND_MSVC)
    set(DEFINE_PREFIX "/D")
else()
    set(DEFINE_PREFIX "-D")
endif()
foreach (f ${DIR_DEFINES})
    list(APPEND DEFINE_STR "${DEFINE_PREFIX}${f}")
endforeach()

file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/dispatch")

if (FRONTEND_MSVC)
    set(CXX_PREPROCESSOR_CMD
        ${CMAKE_CXX_COMPILER} /nologo /TP /P /EP /C /X ${DEFINE_STR} "${CMAKE_CURRENT_SOURCE_DIR}/dispatch.template" /Fidispatch/dispatch.template)
else()
    set(CXX_PREPROCESSOR_CMD
        ${CMAKE_CXX_COMPILER} -x c -E -P -nostdinc ${DEFINE_STR} "${CMAKE_CURRENT_SOURCE_DIR}/dispatch.template" -o dispatch/dispatch.template)
endif()

add_custom_command(
    OUTPUT dispatch/dispatch.template
    COMMAND ${CXX_PREPROCESSOR_CMD} DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/dispatch.template
)

add_custom_command(
    OUTPUT dispatch/dispatch_device_impl.hpp dispatch/dispatch_device_impl.cpp
           dispatch/dispatch_instance_impl.hpp dispatch/dispatch_instance_impl.cpp
    COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/scripts/gen_vk_dispatch.py
            ${CMAKE_CURRENT_BINARY_DIR}/dispatch/dispatch.template dispatch
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/dispatch/dispatch.template
            ${CMAKE_CURRENT_SOURCE_DIR}/scripts/gen_vk_dispatch.py
)

add_custom_target(generate_vk_dispatch DEPENDS
    dispatch/dispatch_device_impl.hpp dispatch/dispatch_device_impl.cpp
    dispatch/dispatch_instance_impl.hpp dispatch/dispatch_instance_impl.cpp
)

add_library(madrona_render_vk STATIC
    ${MADRONA_INC_DIR}/render/vk/backend.hpp backend.cpp
    ${MADRONA_INC_DIR}/render/vk/device.hpp device.cpp
    ${MADRONA_INC_DIR}/render/vk/dispatch.hpp dispatch.cpp
    #${MADRONA_INC_DIR}/render/vk/shader.hpp shader.cpp
    config.hpp
    descriptors.hpp descriptors.cpp descriptors.inl
    pipeline_shaders.hpp pipeline_shaders.cpp
    utils.hpp utils.cpp utils.inl
    memory.hpp memory.cpp memory.inl
)

target_compile_definitions(madrona_render_vk
    PUBLIC
        "-DVK_NO_PROTOTYPES"
)

target_include_directories(madrona_render_vk
    PUBLIC
        "${CMAKE_CURRENT_BINARY_DIR}/dispatch"
)

target_link_libraries(madrona_render_vk
    PUBLIC
        madrona_vk_hdrs
    PRIVATE
        madrona_common
)

add_dependencies(madrona_render_vk generate_vk_dispatch)

if (MADRONA_LINUX)
    target_link_libraries(madrona_render_vk PRIVATE
        ${CMAKE_DL_LIBS}
    )
endif()

if (MADRONA_MACOS OR MADRONA_IOS)
    target_link_libraries(madrona_render_vk PRIVATE
        madrona_moltenvk
    )
endif()

if (TARGET madrona_cuda AND NOT MADRONA_WINDOWS)
    target_link_libraries(madrona_render_vk PRIVATE
        madrona_cuda
    )

    add_library(madrona_render_vk_cuda STATIC
        cuda_interop.hpp cuda_interop.cpp
    )

    target_link_libraries(madrona_render_vk_cuda PUBLIC
        madrona_render_vk madrona_cuda madrona_common)

    target_compile_definitions(madrona_render_vk_cuda
        PUBLIC
            "-DMADRONA_VK_CUDA_SUPPORT"
    )
endif()
