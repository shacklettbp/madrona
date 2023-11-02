add_library(madrona_viz_system STATIC
    ${MADRONA_INC_DIR}/viz/system.hpp viewer_system.cpp
    interop.hpp
)

target_link_libraries(madrona_viz_system PRIVATE
    madrona_mw_core
)

add_library(madrona_viz_recorder STATIC
    ${MADRONA_INC_DIR}/viz/recorder.hpp recorder.cpp
)

target_link_libraries(madrona_viz_recorder PRIVATE
    madrona_viz_system
)

if (NOT TARGET glfw)
    return()
endif()

set(IMGUI_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../external/imgui")
set(STB_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../external/stb")

add_library(imgui_impl STATIC
    ${IMGUI_DIR}/imgui.cpp
    ${IMGUI_DIR}/imgui_draw.cpp
    ${IMGUI_DIR}/imgui_tables.cpp
    ${IMGUI_DIR}/imgui_widgets.cpp
    ${IMGUI_DIR}/backends/imgui_impl_vulkan.h
    ${IMGUI_DIR}/backends/imgui_impl_vulkan.cpp
    ${IMGUI_DIR}/backends/imgui_impl_glfw.h
    imgui_impl_glfw.cpp
)

target_compile_definitions(imgui_impl
    PRIVATE "-DVK_NO_PROTOTYPES")

target_link_libraries(imgui_impl PRIVATE 
    imgui
    glfw
    madrona_libcxx
    madrona_noexceptrtti
    madrona_vk_hdrs
)

add_library(madrona_viewer STATIC
    ${MADRONA_INC_DIR}/viz/viewer.hpp viewer.cpp
    viewer_renderer.hpp viewer_renderer.cpp
    image_util.cpp
)

target_link_libraries(madrona_viewer PRIVATE
    madrona_render_vk
    madrona_render_shader_compiler
    madrona_common
    madrona_mw_core
    madrona_importer
    madrona_viz_system
    imgui
    imgui_impl
    glfw
)

target_include_directories(madrona_viewer SYSTEM PRIVATE
    ${STB_DIR}
)

if (TARGET madrona_render_vk_cuda)
    target_link_libraries(madrona_viewer PRIVATE
        madrona_cuda madrona_render_vk_cuda
    )
endif()

target_compile_definitions(madrona_viewer PRIVATE
    "-DVIEWER_DATA_DIR=${CMAKE_CURRENT_SOURCE_DIR}/"
)

target_include_directories(madrona_viewer PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/../render"
)
