add_library(madrona_render_shader SHARED
    compiler.hpp compiler.cpp
)

target_link_libraries(madrona_render_shader PRIVATE
    madrona_dxcompiler
)
