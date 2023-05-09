#pragma once

#include <madrona/importer.hpp>

namespace madrona::imp {

bool loadOBJFile(const char *path, ImportedAssets &imported_assets,
                 Span<char> err_buf);

}
