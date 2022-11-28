add_subdirectory(vk)

add_library(madrona_batch_render STATIC
    batch_renderer.cpp
)

target_link_libraries(madrona_batch_render PRIVATE
    CUDA::cudart
    madrona_render_vk
    madrona_common
)
