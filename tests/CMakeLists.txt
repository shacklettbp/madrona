set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

add_executable(core_tests
    id_map.cpp
    state.cpp
    static_map.cpp
    math.cpp
    rand.cpp
)

target_link_libraries(core_tests
    gtest_main
    madrona_common
    madrona_core
)

add_executable(physics_tests
    gjk.cpp
)

target_link_libraries(physics_tests
    gtest_main
    madrona_common
    madrona_mw_core
    madrona_mw_physics
)

include(GoogleTest)
gtest_discover_tests(core_tests)
gtest_discover_tests(physics_tests)
