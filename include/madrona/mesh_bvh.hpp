#pragma once

// We are not working with the compressed BVH for now.
#define MADRONA_COMPRESSED_BVH
//#define MADRONA_COMPRESSED_DEINDEXED

#ifdef MADRONA_COMPRESSED_BVH
#include "mesh_bvh_compressed.hpp"
#elif defined(MADRONA_COMPRESSED_DEINDEXED)
#include "mesh_bvh_comp_deindex.hpp"
#else
#include "mesh_bvh_uncompressed.hpp"
#endif

namespace madrona::render {
    
#ifdef MADRONA_COMPRESSED_BVH
using MeshBVH = MeshBVHCompressed;
#elif defined(MADRONA_COMPRESSED_DEINDEXED)
using MeshBVH = MeshBVHCompUnIndexed;
#else
using MeshBVH = MeshBVHUncompressed;
#endif

}
