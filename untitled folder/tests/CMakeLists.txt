set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

add_executable(tests
    id_map.cpp
    state.cpp
    static_map.cpp
    math.cpp
)

target_link_libraries(tests
    gtest_main
    madrona_common
    madrona_core
)

include(GoogleTest)
gtest_discover_tests(tests)
