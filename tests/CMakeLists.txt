set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

add_executable(tests
    hashmap.cpp
    id_map.cpp
    static_map.cpp
    state.cpp
)

target_link_libraries(tests
    gtest_main
    madrona_common
    madrona_core
)

include(GoogleTest)
gtest_discover_tests(tests)
