#include <madrona/importer.hpp>
#include <madrona/dyn_array.hpp>
#include <madrona/heap_array.hpp>

#include <string_view>

#include <meshoptimizer.h>

#include "obj.hpp"
//#include "gltf.hpp"

namespace madrona::imp {

using namespace math;

Optional<ImportedAssets> ImportedAssets::importFromDisk(
    Span<const char * const> paths, Span<char> err_buf)
{
    ImportedAssets imported {
        .geoData = GeometryData {
            .positionArrays { 0 },
            .normalArrays { 0 },
            .tangentAndSignArrays { 0 },
            .uvArrays { 0 },
            .indexArrays { 0 },
            .faceCountArrays { 0 },
            .meshArrays { 0 },
        },
        .objects { 0 },
        .materials { 0 },
        .instances { 0 },
    };


    bool load_success = false;
    for (const char *path : paths) {
        std::string_view path_view(path);

        auto extension_pos = path_view.rfind('.');
        if (extension_pos == path_view.npos) {
            return Optional<ImportedAssets>::none();
        }
        auto extension = path_view.substr(extension_pos + 1);

        if (extension == "obj") {
            load_success = loadOBJFile(path, imported, err_buf);
        } else if (extension == "gltf" || extension == "glb") {
            //load_success = loadGLTFFile(path, imported);
        }

        if (!load_success) {
            break;
        }
    }

    if (!load_success) {
        return Optional<ImportedAssets>::none();
    }

    return imported;

}

}
