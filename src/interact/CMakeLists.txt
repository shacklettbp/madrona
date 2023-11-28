if (TARGET glfw)
    add_library(madrona_window STATIC
        ${MADRONA_INC_DIR}/window.hpp window.cpp
    )

    target_link_libraries(madrona_window
        PRIVATE
            glfw
            madrona_common
            madrona_render
            madrona_render_vk
    )
endif()
