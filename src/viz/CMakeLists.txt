set(IMGUI_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../external/imgui")

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
    madrona_vk_hdrs
)

add_library(madrona_viewer STATIC
    viewer.cpp
    viewer_renderer.hpp viewer_renderer.cpp
)

target_link_libraries(madrona_viewer PRIVATE
    madrona_render_vk
    madrona_render_shader_compiler
    madrona_common
    madrona_mw_core
    madrona_importer
    imgui
    imgui_impl
    glfw
)

if (TARGET madrona_cuda)
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
