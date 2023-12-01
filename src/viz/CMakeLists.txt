if (NOT TARGET madrona_window)
    return()
endif()

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
    madrona_noexceptrtti
    madrona_vk_hdrs
)

add_library(madrona_viz STATIC
    ${MADRONA_INC_DIR}/viz/viewer.hpp
        ${MADRONA_INC_DIR}/viz/viewer.inl
        viewer.cpp
    viewer_renderer.hpp viewer_renderer.cpp
    present.hpp present.cpp 
)

target_include_directories(madrona_viz PRIVATE 
    "${CMAKE_CURRENT_SOURCE_DIR}/../render"
)

target_link_libraries(madrona_viz
    PUBLIC
        madrona_window
        imgui
    PRIVATE
        madrona_render_core
        imgui_impl
        glfw
        stb
)
