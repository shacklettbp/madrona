add_library(madrona_render_metal_cpp_impl STATIC
    metal_cpp_impl.cpp
)

target_link_libraries(madrona_render_metal_cpp_impl PRIVATE
    madrona_metal_cpp
    madrona_metal_libs
    madrona_libcxx
    madrona_noexceptrtti
)

add_library(madrona_render_metal STATIC
    core.cpp
)

target_link_libraries(madrona_render_metal 
    PUBLIC
        madrona_metal_cpp
    PRIVATE
        madrona_render_metal_cpp_impl
        madrona_common
)
