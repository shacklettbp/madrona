#include <vector>
#include <fstream>
#include <iostream>
#include <embree4/rtcore.h>
#include <madrona/mesh_bvh.hpp>
#include <madrona/importer.hpp>
#include <embree4/rtcore_common.h>

#include "embree.hpp"

namespace madrona::imp {

using namespace math;

struct EmbreeLoader::Impl {
    // Right now, this is unused.
};

constexpr int numTrisPerLeaf = 8;
constexpr int nodeWidth = 4;
static constexpr inline int32_t sentinel = (int32_t)0xFFFF'FFFF;

struct NodeCompressed {
    float minX;
    float minY;
    float minZ;
    int8_t expX;
    int8_t expY;
    int8_t expZ;
    uint8_t internalNodes;
    uint8_t qMinX[nodeWidth];
    uint8_t qMinY[nodeWidth];
    uint8_t qMinZ[nodeWidth];
    uint8_t qMaxX[nodeWidth];
    uint8_t qMaxY[nodeWidth];
    uint8_t qMaxZ[nodeWidth];
    int32_t children[nodeWidth];
    int32_t parentID;
};

struct RTC_ALIGN(16) BoundingBox {
    float lower_x, lower_y, lower_z, align0;
    float upper_x, upper_y, upper_z, align1;
};

inline float area(BoundingBox box)
{
    float spanX = box.upper_x - box.lower_x;
    float spanY = box.upper_y - box.lower_y;
    float spanZ = box.upper_z - box.lower_z;
    return spanX * spanY * 2 + spanY * spanZ * 2 + spanX * spanZ * 2;
}

inline BoundingBox merge(BoundingBox box1, BoundingBox box2)
{
    return BoundingBox {
        std::min(box1.lower_x, box2.lower_x),
        std::min(box1.lower_y, box2.lower_y),
        std::min(box1.lower_z, box2.lower_z),
        0,
        std::max(box1.upper_x, box2.upper_x),
        std::max(box1.upper_y, box2.upper_y),
        std::max(box1.upper_z, box2.upper_z),
        0
    };
}

static bool buildProgress(void* userPtr, double f)
{
    (void)userPtr;
    (void)f;

    return true;
}

static void splitPrimitive(const RTCBuildPrimitive* prim, 
                           unsigned int dim,
                           float pos,
                           RTCBounds* lprim,
                           RTCBounds* rprim,
                           void* userPtr)
{
    (void)userPtr;

    assert(dim < 3);
    assert(prim->geomID == 0);
    *(BoundingBox *)lprim = *(BoundingBox *)prim;
    *(BoundingBox *)rprim = *(BoundingBox *)prim;
    (&lprim->upper_x)[dim] = pos;
    (&rprim->lower_x)[dim] = pos;
}

struct Node {
    bool isLeaf;
    virtual float sah() const = 0;
};

struct InnerNode : public Node {
    BoundingBox bounds[4];
    Node* children[4];
    int numChildren;
    int id = -1;

    InnerNode()
    {
        bounds[0] = bounds[1] = bounds[2] = bounds[3] = {};
        children[0] = children[1] = children[2] = children[3] = nullptr;
        numChildren = 0;
        isLeaf = false;
    }

    float sah() const
    {
        float cost = 0;
        BoundingBox total {
            INFINITY,
            INFINITY,
            INFINITY,
            0,
            -INFINITY,
            -INFINITY,
            -INFINITY,
            0
        };

        for(int i = 0; i < 4; i++){
            if(children[i] != nullptr){
                cost += children[i]->sah() * area(bounds[i]);
                total = merge(bounds[i],total);
            }
        }

        assert(area(total) >= 0);

        if(area(total) == 0){
            return 1;
        }

        return 1+ cost/area(total);
    }

    static void * create(RTCThreadLocalAllocator alloc, 
                         unsigned int numChildren,
                         void* userPtr)
    {
        (void)userPtr;

        assert(numChildren > 0);
        void* ptr = rtcThreadLocalAlloc(alloc,sizeof(InnerNode),16);
        return (void*) new (ptr) InnerNode;
    }

    static void setChildren(void *nodePtr, void **childPtr, 
                            unsigned int numChildren, void* userPtr)
    {
        (void)userPtr;

        assert(numChildren > 0);
        for (size_t i=0; i<numChildren; i++)
            ((InnerNode*)nodePtr)->children[i] = (Node*) childPtr[i];
        ((InnerNode*)nodePtr)->numChildren = numChildren;
    }

    static void setBounds(void* nodePtr, const RTCBounds** bounds, 
                          unsigned int numChildren, void* userPtr)
    {
        (void)userPtr;

        assert(numChildren > 0);
        for (size_t i = 0; i < numChildren; i++)
            ((InnerNode*)nodePtr)->bounds[i] = *(const BoundingBox*) bounds[i];
    }
};

struct LeafNode : public Node {
    unsigned int id[8];
    unsigned int numPrims;
    BoundingBox bounds;
    int lid = -1;

    LeafNode (const BoundingBox& bounds)
        : bounds(bounds)
    {
        isLeaf=true;
    }

    float sah() const
    {
        return 1.0f;
    }

    static void * create(RTCThreadLocalAllocator alloc,
                         const RTCBuildPrimitive* prims, 
                         size_t numPrims, void* userPtr)
    {
        (void)userPtr;

        assert(numPrims > 0);
        void* ptr = rtcThreadLocalAlloc(alloc,sizeof(LeafNode),16);
        LeafNode* leaf = new (ptr) LeafNode(*(BoundingBox*)prims);
        leaf->numPrims = numPrims;

        for(int i = 0; i < (int)numPrims; i++){
            leaf->id[i] = prims[i].primID;
        }

        return (void *)leaf;
    }
};

Optional<render::MeshBVH> EmbreeLoader::load(const SourceObject& object)
{
    using render::MeshBVH;

    DynArray<render::MeshBVH::Node> nodes { 0 };
    DynArray<render::MeshBVH::LeafGeometry> leaf_geos { 0 };
    DynArray<render::MeshBVH::LeafMaterial> leaf_materials { 0 };

    math::AABB aabb_out;

    render::MeshBVH bvh_out;

    uint32_t current_node_offset = nodes.size();

    int numTriangles = 0;
    int numVertices = 0;
    std::vector<long> offsets;
    offsets.resize(object.meshes.size()+1);
    std::vector<long>  triOffsets;
    triOffsets.resize(object.meshes.size()+1);

    offsets[0] = 0;
    triOffsets[0] = 0;

    for(int i = 0; i < object.meshes.size(); i++) {
        numTriangles += object.meshes[i].numFaces;
        numVertices += object.meshes[i].numVertices;
        offsets[i+1] = object.meshes[i].numVertices+offsets[i];
        triOffsets[i+1] = object.meshes[i].numFaces+triOffsets[i];
    }

    RTCDevice device = rtcNewDevice(NULL);
    RTCBVH bvh = rtcNewBVH(device);
    std::vector<RTCBuildPrimitive> prims_i;
    prims_i.resize(numTriangles);

    DynArray<math::Vector3> vertices { 0 };
    vertices.resize(numVertices, [](math::Vector3 *) {});

    std::vector<madrona::render::TriangleIndices> prims_compressed;
    prims_compressed.resize(numTriangles);

    int index = 0;
    int counter = 0;

    for(int mesh_idx = 0; mesh_idx < object.meshes.size(); mesh_idx++){
        auto& mesh = object.meshes[mesh_idx];

        for(uint32_t vert_idx = 0; vert_idx < mesh.numVertices; vert_idx++) {
            madrona::math::Vector3 v1 = mesh.positions[vert_idx];
            assert(vert_idx + offsets[mesh_idx] < vertices.size());
            assert(counter < vertices.size());
            vertices[vert_idx + offsets[mesh_idx]] = {v1.x,v1.y,v1.z};
            counter++;
        }

        for(int face_idx = 0; face_idx < (int)mesh.numFaces; face_idx++){
            if (mesh.faceCounts != nullptr) {
                FATAL("MeshBVH only supports triangular meshes");
            }
            int32_t base = 3 * face_idx;

            uint32_t mesh_a_idx = mesh.indices[base + 0];
            uint32_t mesh_b_idx = mesh.indices[base + 1];
            uint32_t mesh_c_idx = mesh.indices[base + 2];

            auto v1 = mesh.positions[mesh_a_idx];
            auto v2 = mesh.positions[mesh_b_idx];
            auto v3 = mesh.positions[mesh_c_idx];

            uint32_t global_a_idx = mesh_a_idx + offsets[mesh_idx];
            uint32_t global_b_idx = mesh_b_idx + offsets[mesh_idx];
            uint32_t global_c_idx = mesh_c_idx + offsets[mesh_idx];

            int32_t b_diff = (int32_t)global_b_idx - (int32_t)global_a_idx;
            int32_t c_diff = (int32_t)global_c_idx - (int32_t)global_a_idx;
            // assert(abs(b_diff) < 32767 && abs(c_diff) < 32767);


            // For now, we are just doing it dumbly
            (void)b_diff, (void)c_diff;

            prims_compressed[triOffsets[mesh_idx] + face_idx] = {
                { global_a_idx, global_b_idx, global_c_idx }
            };

            float minX = std::min(std::min(v1.x,v2.x),v3.x);
            float minY = std::min(std::min(v1.y,v2.y),v3.y);
            float minZ = std::min(std::min(v1.z,v2.z),v3.z);

            float maxX = std::max(std::max(v1.x,v2.x),v3.x);
            float maxY = std::max(std::max(v1.y,v2.y),v3.y);
            float maxZ = std::max(std::max(v1.z,v2.z),v3.z);

            RTCBuildPrimitive prim;
            prim.lower_x = minX;
            prim.lower_y = minY;
            prim.lower_z = minZ;
            prim.geomID = 0;
            prim.upper_x = maxX;
            prim.upper_y = maxY;
            prim.upper_z = maxZ;
            prim.primID = index;
            prims_i[index] = prim;
            index++;
        }
    }

    std::vector<RTCBuildPrimitive> prims;
    prims.reserve(numTriangles);
    prims.resize(numTriangles);

    /* settings for BVH build */
    RTCBuildArguments arguments = rtcDefaultBuildArguments();
    arguments.byteSize = sizeof(arguments);
    arguments.buildFlags = RTC_BUILD_FLAG_NONE;
    arguments.buildQuality = RTC_BUILD_QUALITY_HIGH;
    arguments.maxBranchingFactor = 4;
    arguments.maxDepth = 1024;
    arguments.sahBlockSize = 1;
    arguments.minLeafSize = 4;
    arguments.maxLeafSize = 8;
    arguments.traversalCost = 1.0f;
    arguments.intersectionCost = 8.0f;
    arguments.bvh = bvh;
    arguments.primitives = prims.data();
    arguments.primitiveCount = prims.size();
    arguments.primitiveArrayCapacity = prims.capacity();
    arguments.createNode = InnerNode::create;
    arguments.setNodeChildren = InnerNode::setChildren;
    arguments.setNodeBounds = InnerNode::setBounds;
    arguments.createLeaf = LeafNode::create;
    arguments.splitPrimitive = splitPrimitive;
    arguments.buildProgress = buildProgress;
    arguments.userPtr = nullptr;

    Node* root;
    for (size_t i=0; i<10; i++)
    {
        /* we recreate the prims array here, as the builders modify this array */
        for (size_t j=0; j<prims.size(); j++) prims[j] = prims_i[j];

        root = (Node*) rtcBuildBVH(&arguments);
    }

    std::vector<Node*> stack;
    stack.push_back(root);

    std::vector<InnerNode*> innerNodes;
    std::vector<LeafNode*> leafNodes;

    int childrenCounts[]{0,0,0,0,0};

    int leafID = 0;
    int innerID = 0;

    while(!stack.empty()){
        Node* node = stack.back();
        stack.pop_back();
        if(!node->isLeaf){
            auto* inner = (InnerNode*)node;
            for(int i=0;i<4;i++){
                if(inner->children[i] != nullptr){
                    stack.push_back(inner->children[i]);
                }
            }
            if(inner->id == -1){
                inner->id = innerID;
                innerNodes.push_back(inner);
                innerID++;
            }
            childrenCounts[inner->numChildren]++;
        }else{
            auto* leaf = (LeafNode*)node;

            if(leaf->lid == -1){
                leaf->lid = leafID;
                leafNodes.push_back(leaf);
                leafID++;
            }
        }
    }

    madrona::Optional<std::ofstream> out = madrona::Optional<std::ofstream>::none();

    if(innerID == 0){
        render::MeshBVH::Node node;
        for(int j=0;j<nodeWidth;j++){
            int32_t child;
            if(j < (int)leafNodes.size()) {
                LeafNode *iNode = (LeafNode *) leafNodes[j];
                child = 0x80000000 | iNode->lid;
                BoundingBox box = iNode->bounds;
                node.minX[j] = box.lower_x;
                node.minY[j] = box.lower_y;
                node.minZ[j] = box.lower_z;
                node.maxX[j] = box.upper_x;
                node.maxY[j] = box.upper_y;
                node.maxZ[j] = box.upper_z;
            }else{
                child = sentinel;
            }
            node.children[j] = child;
            //node.children[j] = 0xBBBBBBBB;
        }
        nodes.push_back(node);
    }

#ifdef MADRONA_COMPRESSED_BVH
    float rootMaxX = FLT_MIN;
    float rootMaxY = FLT_MIN;
    float rootMaxZ = FLT_MIN;

    if(innerID == 0) {
        float minX = FLT_MAX,
              minY = FLT_MAX,
              minZ = FLT_MAX,
              maxX = FLT_MIN,
              maxY = FLT_MIN,
              maxZ = FLT_MIN;

        for(int i2 = 0; i2 < nodeWidth; i2++) {
            if(i2 < leafNodes.size()) {
                LeafNode *iNode = (LeafNode *) leafNodes[i2];
                BoundingBox box = iNode->bounds;
                minX = fminf(minX, box.lower_x);
                minY = fminf(minY, box.lower_y);
                minZ = fminf(minZ, box.lower_z);
                maxX = fmaxf(maxX, box.upper_x);
                maxY = fmaxf(maxY, box.upper_y);
                maxZ = fmaxf(maxZ, box.upper_z);
            }
        }

        rootMaxX = maxX;
        rootMaxY = maxY;
        rootMaxZ = maxZ;

        MeshBVH::Node node;
        int8_t ex = ceilf(log2f((maxX-minX) / (powf(2, 8) - 1)));
        int8_t ey = ceilf(log2f((maxY-minY) / (powf(2, 8) - 1)));
        int8_t ez = ceilf(log2f((maxZ-minZ) / (powf(2, 8) - 1)));
        
        node.minX = minX;
        node.minY = minY;
        node.minZ = minZ;
        node.expX = ex;
        node.expY = ey;
        node.expZ = ez;
        for(int j = 0; j < nodeWidth; j++){
            int32_t child;
            if(j < leafNodes.size()) {
                LeafNode *iNode = (LeafNode *) leafNodes[j];
                child = 0x80000000 | iNode->lid;
                BoundingBox box = iNode->bounds;
                node.qMinX[j] = floorf((box.lower_x - minX) / powf(2, ex));
                node.qMinY[j] = floorf((box.lower_y - minY) / powf(2, ey));
                node.qMinZ[j] = floorf((box.lower_z - minZ) / powf(2, ez));
                node.qMaxX[j] = ceilf((box.upper_x - minX) / powf(2, ex));
                node.qMaxY[j] = ceilf((box.upper_y - minY) / powf(2, ey));
                node.qMaxZ[j] = ceilf((box.upper_z - minZ) / powf(2, ez));
            } else {
                child = sentinel;
            }
            node.children[j] = child;
            //node.children[j] = 0xBBBBBBBB;
        }
        nodes.push_back(node);
    }
    for(int i = 0; i < innerID; i++){
        MeshBVH::Node node;
        float minX = FLT_MAX,
              minY = FLT_MAX,
              minZ = FLT_MAX,
              maxX = FLT_MIN,
              maxY = FLT_MIN,
              maxZ = FLT_MIN;

        for(int i2 = 0; i2 < nodeWidth; i2++){
            if(innerNodes[i]->children[i2] != nullptr) {
                minX = fminf(minX, innerNodes[i]->bounds[i2].lower_x);
                minY = fminf(minY, innerNodes[i]->bounds[i2].lower_y);
                minZ = fminf(minZ, innerNodes[i]->bounds[i2].lower_z);
                maxX = fmaxf(maxX, innerNodes[i]->bounds[i2].upper_x);
                maxY = fmaxf(maxY, innerNodes[i]->bounds[i2].upper_y);
                maxZ = fmaxf(maxZ, innerNodes[i]->bounds[i2].upper_z);
            }
        }

        rootMaxX = fmaxf(maxX,rootMaxX);
        rootMaxY = fmaxf(maxY,rootMaxY);
        rootMaxZ = fmaxf(maxZ,rootMaxZ);
        //printf("%f,%f,%f | %f,%f,%f\n",minX,minY,minZ,maxX,maxY,maxZ);

        int8_t ex = ceilf(log2f((maxX-minX)/(powf(2, 8) - 1)));
        int8_t ey = ceilf(log2f((maxY-minY)/(powf(2, 8) - 1)));
        int8_t ez = ceilf(log2f((maxZ-minZ)/(powf(2, 8) - 1)));
        //printf("%d,%d,%d\n",ex,ey,ez);
        node.minX = minX;
        node.minY = minY;
        node.minZ = minZ;
        node.expX = ex;
        node.expY = ey;
        node.expZ = ez;
        node.parentID = -1;
        for(int i2 = 0; i2 < nodeWidth; i2++) {
            node.qMinX[i2] = floorf((innerNodes[i]->bounds[i2].lower_x - minX) / powf(2, ex));
            node.qMinY[i2] = floorf((innerNodes[i]->bounds[i2].lower_y - minY) / powf(2, ey));
            node.qMinZ[i2] = floorf((innerNodes[i]->bounds[i2].lower_z - minZ) / powf(2, ez));
            node.qMaxX[i2] = ceilf((innerNodes[i]->bounds[i2].upper_x - minX) / powf(2, ex));
            node.qMaxY[i2] = ceilf((innerNodes[i]->bounds[i2].upper_y - minY) / powf(2, ey));
            node.qMaxZ[i2] = ceilf((innerNodes[i]->bounds[i2].upper_z - minZ) / powf(2, ez));
        }
        for(int j = 0; j < nodeWidth; j++){
            int32_t child;
            if(j < innerNodes[i]->numChildren) {
                Node *node2 = innerNodes[i]->children[j];
                if (!node2->isLeaf) {
                    InnerNode *iNode = (InnerNode *) node2;
                    child = iNode->id;
                } else {
                    LeafNode *iNode = (LeafNode *) node2;
                    child = 0x80000000 | iNode->lid;
                }
            } else {
                child = sentinel;
            }
            node.children[j] = child;
        }
        nodes.push_back(node);
    }

    auto *root_node = &nodes[current_node_offset];

    // Create root AABB
    madrona::math::AABB merged = {
        .pMin = { root_node->minX, root_node->minY, root_node->minZ},
        .pMax = { rootMaxX, rootMaxY, rootMaxZ },
    };

    aabbOut = merged;
#else
    if(innerID == 0){
        MeshBVH::Node node;
        for(int j = 0; j < nodeWidth; j++){
            int32_t child;
            if(j < leafNodes.size()) {
                LeafNode *iNode = (LeafNode *) leafNodes[j];
                child = 0x80000000 | iNode->lid;
                BoundingBox box = iNode->bounds;
                node.minX[j] = box.lower_x;
                node.minY[j] = box.lower_y;
                node.minZ[j] = box.lower_z;
                node.maxX[j] = box.upper_x;
                node.maxY[j] = box.upper_y;
                node.maxZ[j] = box.upper_z;
            } else {
                child = sentinel;
            }
            node.children[j] = child;
        }
        nodes.push_back(node);
    }

    for(int i = 0; i < innerID; i++){
        render::MeshBVH::Node node;
        node.parentID = -1;
        for (int i2 = 0; i2 < nodeWidth; i2++){
            BoundingBox box = innerNodes[i]->bounds[i2];
            node.minX[i2] = box.lower_x;
            node.minY[i2] = box.lower_y;
            node.minZ[i2] = box.lower_z;
            node.maxX[i2] = box.upper_x;
            node.maxY[i2] = box.upper_y;
            node.maxZ[i2] = box.upper_z;
        }
        for(int j = 0; j < nodeWidth; j++){
            int32_t child;
            if(j < innerNodes[i]->numChildren) {
                Node *node2 = innerNodes[i]->children[j];
                if (!node2->isLeaf) {
                    InnerNode *iNode = (InnerNode *) node2;
                    child = iNode->id;
                } else {
                    LeafNode *iNode = (LeafNode *) node2;
                    child = 0x80000000 | iNode->lid;
                }
            }else{
                child = sentinel;
            }
            node.children[j] = child;
        }

        nodes.push_back(node);
    }

    auto *root_node = &nodes[current_node_offset];

    // Create root AABB
    madrona::math::AABB merged = {
        .pMin = { root_node->minX[0], root_node->minY[0], root_node->minZ[0] },
        .pMax = { root_node->maxX[0], root_node->maxY[0], root_node->maxZ[0] },
    };

    for (int aabb_idx = 1; aabb_idx < nodeWidth; ++aabb_idx) {
        if (root_node->hasChild(aabb_idx)) {
            madrona::math::AABB child_aabb = {
                .pMin = { root_node->minX[aabb_idx], root_node->minY[aabb_idx], root_node->minZ[aabb_idx] },
                .pMax = { root_node->maxX[aabb_idx], root_node->maxY[aabb_idx], root_node->maxZ[aabb_idx] },
            };

            merged = madrona::math::AABB::merge(merged, child_aabb);
        }
    }

    aabb_out = merged;

#endif
    for(int i=0;i<leafID;i++){
        LeafNode* node = leafNodes[i];
        render::MeshBVH::LeafGeometry geos;
        for(int i2=0;i2<(int)numTrisPerLeaf;i2++){
            if(i2<(int)node->numPrims){
                geos.packedIndices[i2] = prims_compressed[node->id[i2]];
            }else{
                // geos.packedIndices[i2] = 0xFFFF'FFFF'FFFF'FFFF;
                geos.packedIndices[i2] = {
                    { 0xFFFF'FFFF, 0xFFFF'FFFF, 0xFFFF'FFFF }
                };
            }
        }
        leaf_geos.push_back(geos);
    }
    for(int i=0;i<leafID;i++){
        render::MeshBVH::LeafMaterial geos;
        for(int i2=0;i2<numTrisPerLeaf;i2++){
            geos.material[i2] = 0xAAAAAAAA;
        }
        leaf_materials.push_back(geos);
    }

    rtcReleaseBVH(bvh);

    bvh_out.numNodes = nodes.size();
    bvh_out.numLeaves = leaf_geos.size();
    bvh_out.numVerts = vertices.size();
    bvh_out.magic = MeshBVH::magicSignature;

    bvh_out.nodes = nodes.release(true);
    bvh_out.leafGeos = leaf_geos.release(true);
    bvh_out.leafMats = leaf_materials.release(true);
    bvh_out.vertices = vertices.release(true);
    bvh_out.rootAABB = aabb_out;

    return bvh_out;
}

EmbreeLoader::EmbreeLoader(Span<char> err_buf)
{
    (void)err_buf;
}

EmbreeLoader::~EmbreeLoader()
{
}

}
