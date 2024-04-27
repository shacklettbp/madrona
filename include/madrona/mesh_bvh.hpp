#pragma once

// We are not working with the compressed BVH for now.
// #define MADRONA_COMPRESSED_BVH

#ifdef MADRONA_COMPRESSED_BVH
#include "mesh_bvh_compressed.hpp"
#else
#include "mesh_bvh_uncompressed.hpp"
#endif

namespace madrona::render {
    
#ifdef MADRONA_COMPRESSED_BVH
using MeshBVH = MeshBVHCompressed;
#else
using MeshBVH = MeshBVHUncompressed;
#endif

}
