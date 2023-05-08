#include <madrona/importer.hpp>
#include <madrona/dyn_array.hpp>
#include <madrona/heap_array.hpp>

#include <string_view>

#include <meshoptimizer.h>

#include "obj.hpp"
//#include "gltf.hpp"

namespace madrona::imp {

using namespace math;

Optional<ImportedAssets> ImportedAssets::importFromDisk(const char *path)
{
    ImportedAssets imported {
        .objects = DynArray<ImportedObject>(1),
        .materials = DynArray<ImportedMaterial>(0),
        .instances = DynArray<ImportedInstance>(0),
    };

    std::string_view path_view(path);

    auto extension_pos = path_view.rfind('.');
    if (extension_pos == path_view.npos) {
        return Optional<ImportedAssets>::none();
    }
    auto extension = path_view.substr(extension_pos + 1);

    bool load_success = false;
    if (extension == "obj") {
        load_success = loadOBJFile(path, imported);
    } else if (extension == "gltf" || extension == "glb") {
        //load_success = loadGLTFFile(path, imported);
    }

    if (!load_success) {
        return Optional<ImportedAssets>::none();
    }

    return imported;

}

}
