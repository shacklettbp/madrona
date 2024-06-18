#include <madrona/mesh_bvh.hpp>

#include <madrona/importer.hpp>

namespace madrona {

struct MeshBVHBuilder {
#if 0
    static MeshBVH build(
        Span<const imp::SourceMesh> src_meshes,
        StackAlloc &tmp_alloc,
        StackAlloc::Frame *out_alloc_frame);
#endif

    static MeshBVH build(
        Span<const imp::SourceMesh> src_meshes);

};

}
