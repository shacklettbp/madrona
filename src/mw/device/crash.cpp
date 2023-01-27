/*
 * Copyright 2021-2022 Brennan Shacklett and contributors
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 */
#include <madrona/crash.hpp>

#include <array>
#include <cstdio>
#include <cstdlib>

extern "C" int vprintf(const char *, va_list);

namespace madrona {

void fatal(const char *file, int line, const char *funcname,
           const char *fmt, ...)
{
    va_list args;
    va_start(args, fmt);
    vprintf(fmt, args);
    printf("\n");

    __assertfail("Fatal error", file, line, funcname, 1);
    __builtin_unreachable();
}

}
