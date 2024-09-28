/*
 * Copyright 2021-2022 Brennan Shacklett and contributors
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 */
#pragma once

#include <madrona/macros.hpp>

namespace madrona {

struct CrashInfo {
    const char *file;
    int line;
    const char *funcname;
    const char *msg;
};

[[noreturn]] void fatal(const char *file, int line,
    const char *funcname, const char *fmt, ...);
[[noreturn]] void fatal(const CrashInfo &crash);

void debuggerBreakPoint();

#if __cplusplus >= 202002L
#define FATAL(fmt, ...) ::madrona::fatal(__FILE__, __LINE__,\
    MADRONA_COMPILER_FUNCTION_NAME, fmt __VA_OPT__(,) __VA_ARGS__ )
#else
#define FATAL(fmt, ...) ::madrona::fatal(__FILE__, __LINE__,\
    MADRONA_COMPILER_FUNCTION_NAME, fmt ##__VA_ARGS__ )
#endif

}
