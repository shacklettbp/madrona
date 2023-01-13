add_subdirectory(vk)

add_library(madrona_ecs_render STATIC
    render_ecs.cpp
)

target_link_libraries(madrona_ecs_render PUBLIC
    madrona_mw_core
)

add_library(madrona_mw_ecs_render STATIC
    render_ecs.cpp
)

target_link_libraries(madrona_mw_ecs_render PUBLIC
    madrona_mw_core
)

add_library(madrona_batch_render STATIC
    batch_renderer.cpp
)

target_link_libraries(madrona_batch_render PRIVATE
    CUDA::cudart
    madrona_render_vk
    madrona_common
)

target_compile_definitions(madrona_batch_render
    PRIVATE
        "-DSHADER_DIR=${CMAKE_CURRENT_SOURCE_DIR}/vk/shaders/"
)
