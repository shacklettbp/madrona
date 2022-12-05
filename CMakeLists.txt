cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
cmake_policy(VERSION 3.18)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

include(build_type)

project(MadronaEngine LANGUAGES C CXX)

include(setup)

set(CMAKE_THREAD_PREFER_PTHREAD TRUE)
set(THREADS_PREFER_PTHREAD_FLAG TRUE)
find_package(Threads REQUIRED)
find_package(CUDAToolkit QUIET)
find_package(Python 3.9 COMPONENTS Interpreter Development.Module QUIET)

add_subdirectory(external)
add_subdirectory(src)

if (MADRONA_ENABLE_EXAMPLES)
    add_subdirectory(examples)
endif()

if (MADRONA_ENABLE_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
