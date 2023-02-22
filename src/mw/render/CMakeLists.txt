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
        metal/scene.hpp metal/scene.cpp
        batch_renderer.hpp metal/batch_renderer.cpp
    )

    function(compile_metal_shaders OUTPUT_DIR TGT_NAME)

        foreach(SHADER_SRC_FILE ${ARGN})
            cmake_path(GET SHADER_SRC_FILE FILENAME SHADER_SRC_FILENAME)
            cmake_path(GET SHADER_SRC_FILE STEM SHADER_SRC_FILE_STEM)

            set(MTLLIB_OUT "${OUTPUT_DIR}/${SHADER_SRC_FILE_STEM}.metallib")
            add_custom_command(
                OUTPUT "${MTLLIB_OUT}"
                DEPENDS "${SHADER_SRC_FILE}"
                COMMAND xcrun -sdk macosx metal -frecord-sources=flat "${SHADER_SRC_FILE}" -o "${MTLLIB_OUT}"
            )

            list(APPEND MTLLIBS
                "${MTLLIB_OUT}"
            )
        endforeach()

        add_custom_target(${TGT_NAME} DEPENDS ${MTLLIBS})
    endfunction()

    set(SHADER_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/metal/shaders")
    set(SHADER_BIN_DIR "${CMAKE_BINARY_DIR}/shaders/metal/madrona/")
    file(MAKE_DIRECTORY "${SHADER_BIN_DIR}")

    compile_metal_shaders("${SHADER_BIN_DIR}" 
        madrona_batch_render_metal_shaders
        "${SHADER_SRC_DIR}/present.metal" 
        "${SHADER_SRC_DIR}/draw.metal" 
    )

    target_compile_definitions(madrona_batch_render PRIVATE
        "MADRONA_BATCHRENDERER_MTL_SHADER_DIR=\"${SHADER_BIN_DIR}\""
    )

    target_link_libraries(madrona_batch_render PRIVATE
        madrona_render_metal
    )

    add_dependencies(madrona_batch_render madrona_batch_render_metal_shaders)
endif()

target_include_directories(madrona_batch_render PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../render
)
    
target_link_libraries(madrona_batch_render PRIVATE
    madrona_common
)
