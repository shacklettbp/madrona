#pragma once

namespace madrona {

struct CrashInfo {
    const char *file;
    int line;
    const char *msg;
};

[[noreturn]] void fatal(const char *file, int line, const char *fmt, ...);
[[noreturn]] void fatal(const CrashInfo &crash);

#define FATAL_VA_ARGS(...) , ##__VA_ARGS__
#define FATAL(fmt, ...) fatal(__FILE__, __LINE__, fmt FATAL_VA_ARGS( __VA_ARGS__))

#define STATIC_UNIMPLEMENTED() \
    do { static_assert(false, "Unimplemented"); } while (false)

}
