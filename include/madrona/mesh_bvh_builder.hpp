#include <madrona/mesh_bvh.hpp>

#include <madrona/importer.hpp>

namespace madrona {

struct MeshBVHBuilder {
    static MeshBVH build(
        Span<const imp::SourceMesh> src_meshes);

};

}
