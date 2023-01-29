add_library(madrona_mw_render STATIC
    interop.hpp interop.cpp
)

target_link_libraries(madrona_mw_render PUBLIC
    madrona_mw_core
)

add_library(madrona_batch_render STATIC
    batch_renderer.cpp
    scene.hpp scene.cpp
)

target_include_directories(madrona_batch_render PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../render
)

target_link_libraries(madrona_batch_render PRIVATE
    madrona_common
)

if (TARGET madrona_render_vk)
    target_link_libraries(madrona_batch_render PRIVATE
        madrona_render_vk
    )

    target_compile_definitions(madrona_batch_render PRIVATE
        "-DSHADER_DIR=${CMAKE_CURRENT_SOURCE_DIR}/../../render/vk/shaders/"
    )
endif ()

if (TARGET madrona_cuda)
    target_link_libraries(madrona_batch_render PRIVATE
        CUDA::cudart
    )
endif ()
