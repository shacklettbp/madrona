#pragma once

// We are not working with the compressed BVH for now.
//#define MADRONA_COMPRESSED_BVH
//#define MADRONA_COMPRESSED_DEINDEXED
#define MADRONA_COMPRESSED_DEINDEXED_TEX

#ifdef MADRONA_COMPRESSED_BVH
#include "mesh_bvh_compressed.hpp"
#elif defined(MADRONA_COMPRESSED_DEINDEXED)
#include "mesh_bvh_comp_deindex.hpp"
#elif defined(MADRONA_COMPRESSED_DEINDEXED_TEX)
#include "mesh_bvh_comp_deindex_tex.hpp"
#else
#include "mesh_bvh_uncompressed.hpp"
#endif

namespace madrona::render {

#ifdef MADRONA_COMPRESSED_BVH
using MeshBVH = MeshBVHCompressed;
#elif defined(MADRONA_COMPRESSED_DEINDEXED)
using MeshBVH = MeshBVHCompUnIndexed;
#elif defined(MADRONA_COMPRESSED_DEINDEXED_TEX)
using MeshBVH = MeshBVHCompUnIndexedTex;
#else
using MeshBVH = MeshBVHUncompressed;
#endif

}
