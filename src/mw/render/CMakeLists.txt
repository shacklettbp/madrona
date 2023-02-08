add_library(madrona_mw_render STATIC
    interop.hpp interop.cpp
)

target_link_libraries(madrona_mw_render PUBLIC
    madrona_mw_core
)

if (TARGET madrona_render_vk)
    add_library(madrona_batch_render STATIC
        batch_renderer.hpp
        vk/batch_renderer.cpp
        vk/scene.hpp vk/scene.cpp
    )
    
    target_link_libraries(madrona_batch_render PRIVATE
        madrona_render_vk
    )
    
    target_compile_definitions(madrona_batch_render PRIVATE
        "-DSHADER_DIR=${CMAKE_CURRENT_SOURCE_DIR}/../../render/vk/shaders/"
    )
    
    if (CUDAToolkit_FOUND)
        target_link_libraries(madrona_batch_render PRIVATE
            CUDA::cudart
        )
    endif ()
elseif (TARGET madrona_render_metal)
    add_library(madrona_batch_render STATIC
        batch_renderer.hpp metal/batch_renderer.cpp
    )

    target_link_libraries(madrona_batch_render PRIVATE
        madrona_render_metal
    )
endif()

target_include_directories(madrona_batch_render PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../render
)
    
target_link_libraries(madrona_batch_render PRIVATE
    madrona_common
)
