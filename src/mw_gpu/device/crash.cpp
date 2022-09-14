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
