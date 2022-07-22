#pragma once

#define MADRONA_STRINGIFY_HELPER(m) #m
#define MADRONA_STRINGIFY(m) MADRONA_STRINGIFY_HELPER(m)

#define MADRONA_LOC_APPEND(m) m ": " __FILE__ " @ " MADRONA_STRINGIFY(__LINE__)

#if defined(__clang__) or defined(__GNUC__) or defined(__CUDACC__)
#define MADRONA_COMPILER_FUNCTION_NAME __PRETTY_FUNCTION__
#elif defined(_MSC_VER)
#define MADRONA_COMPILER_FUNCTION_NAME __FUNCSIG__
#endif
