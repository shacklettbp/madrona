#include <madrona/json.hpp>

namespace madrona::json {

void checkSIMDJSONResult(
    simdjson::error_code err,
    const char *file,
    int line,
    const char *func_name)
{
    if (err) {
        fatal(file, line, func_name, "Failed to parse JSON: %s",
              simdjson::error_message(err));
    }
}

}
