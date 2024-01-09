/*
 * Copyright 2021-2022 Brennan Shacklett and contributors
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 */
#pragma once

#if defined(_MSC_VER)
#define MADRONA_MSVC (1)
#if defined(__clang__)
#define MADRONA_CLANG_CL (1)
#endif
#elif defined(__clang__)
#define MADRONA_CLANG (1)
#elif defined(__GNUC__)
#define MADRONA_GCC (1)
#elif !defined(MADRONA_GPU_MODE)
#error "Unsupported compiler"
#endif

#if defined(__x86_64__) || defined(_M_X64)
#define MADRONA_X64 (1)
#elif defined (__arm64__) || defined(_M_ARM64)
#define MADRONA_ARM (1)
#elif !defined (MADRONA_GPU_MODE)
#error "Unsupported architecture"
#endif

#if defined(__WIN32__) or defined(_WIN32) or defined(WIN32)
#define MADRONA_WINDOWS (1)
#elif defined(__APPLE__)
#define MADRONA_APPLE (1)
#elif defined(__linux)
#define MADRONA_LINUX (1)
#endif

#define MADRONA_STRINGIFY_HELPER(m) #m
#define MADRONA_STRINGIFY(m) MADRONA_STRINGIFY_HELPER(m)

#define MADRONA_LOC_APPEND(m) m ": " __FILE__ " @ " MADRONA_STRINGIFY(__LINE__)

#if defined(__clang__) or defined(__GNUC__) or defined(__CUDACC__)
#define MADRONA_COMPILER_FUNCTION_NAME __PRETTY_FUNCTION__
#elif defined(_MSC_VER)
#define MADRONA_COMPILER_FUNCTION_NAME __FUNCSIG__
#endif

#ifdef MADRONA_MW_MODE
#define MADRONA_MW_COND(...) __VA_ARGS__
#else
#define MADRONA_MW_COND(...)
#endif

#ifdef MADRONA_GPU_MODE
#define MADRONA_GPU_COND(...) __VA_ARGS__
#else
#define MADRONA_GPU_COND(...)
#endif

#ifdef MADRONA_X64
#define MADRONA_CACHE_LINE (64)
#elif defined(MADRONA_ARM) && defined(MADRONA_MACOS)
#define MADRONA_CACHE_LINE (128)
#else
#define MADRONA_CACHE_LINE (64)
#endif

#if defined(MADRONA_MSVC)

#define MADRONA_NO_INLINE __declspec(noinline)
#if defined(MADRONA_CLANG_CL)
#define MADRONA_ALWAYS_INLINE __attribute__((always_inline))
#else
#define MADRONA_ALWAYS_INLINE [[msvc::forceinline]]
#endif

#elif defined(MADRONA_CLANG) || defined(MADRONA_GCC) || defined(MADRONA_GPU_MODE)

#define MADRONA_ALWAYS_INLINE __attribute__((always_inline))
#define MADRONA_NO_INLINE __attribute__((noinline))

#endif

#if defined _WIN32 || defined __CYGWIN__
#define MADRONA_IMPORT __declspec(dllimport)
#define MADRONA_EXPORT __declspec(dllexport)
#else
#define MADRONA_IMPORT __attribute__ ((visibility ("default")))
#define MADRONA_EXPORT __attribute__ ((visibility ("default")))
#endif

#if defined(MADRONA_MSVC)
#define MADRONA_UNREACHABLE() __assume(0)
#else
#define MADRONA_UNREACHABLE() __builtin_unreachable()
#endif

#if defined(MADRONA_CLANG) || defined(MADRONA_CLANG_CL)
#define MADRONA_LFBOUND [[clang::lifetimebound]]
#elif defined(MADRONA_MSVC)
#define MADRONA_LFBOUND [[msvc::lifetimebound]]
#else
#define MADRONA_LFBOUND
#endif

#define STATIC_UNIMPLEMENTED() \
    static_assert(false, "Unimplemented")

#if defined(MADRONA_GPU_MODE) || defined(MADRONA_CLANG)
#define MADRONA_UNROLL _Pragma("unroll")
#else
#define MADRONA_UNROLL
#endif
