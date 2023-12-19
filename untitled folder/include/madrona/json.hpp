#pragma once

#include <simdjson.h>

#include <madrona/crash.hpp>

namespace madrona::json {

void checkSIMDJSONResult(simdjson::error_code err,
                         const char *file, int line,
                         const char *func_name);

#define REQ_JSON(err) ::madrona::json::checkSIMDJSONResult(err, __FILE__, \
    __LINE__, MADRONA_COMPILER_FUNCTION_NAME);

}
