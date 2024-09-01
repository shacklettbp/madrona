/*
 * Copyright 2021-2022 Brennan Shacklett and contributors
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 */
#include <madrona/crash.hpp>

#include <array>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>

#if defined(MADRONA_LINUX) || defined(MADRONA_MACOS)
#include <csignal>
#elif defined(MADRONA_WINDOWS)
#include <windows.h>
#endif

using namespace std;

namespace madrona {

void fatal(const char *file, int line, const char *funcname,
           const char *fmt, ...)
{
    // Use a fixed size buffer for the error message. This sets an upper
    // bound on total memory size, and wastes 4kb on memory, but is very
    // robust to things going horribly wrong elsewhere.
    static array<char, 4096> buffer;

    va_list args;
    va_start(args, fmt);
    vsnprintf(buffer.data(), buffer.size(), fmt, args);

    fatal(CrashInfo {
        file,
        line,
        funcname,
        buffer.data(),
    });
}

void fatal(const CrashInfo &crash)
{
    fprintf(stderr, "Error at %s:%d in %s\n", crash.file, crash.line,
            crash.funcname);
    if (crash.msg) {
        fprintf(stderr, "%s\n", crash.msg);
    }

    fflush(stderr);
    abort();
}

void debuggerBreakPoint()
{
#if defined(MADRONA_LINUX) || defined(MADRONA_MACOS)
  signal(SIGTRAP, SIG_IGN);
  raise(SIGTRAP);
  signal(SIGTRAP, SIG_DFL);
#elif defined(MADRONA_WINDOWS)
  if (IsDebuggerPresent()) {
    DebugBreak();
  }
#endif
}

}
