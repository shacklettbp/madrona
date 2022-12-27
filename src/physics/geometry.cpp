#ifndef MADRONA_GPU_MODE
#include <stdlib.h>
#include <memory.h>
#include <map>
#endif

#include <madrona/physics.hpp>
#include <madrona/context.hpp>

namespace madrona {

namespace phys::geometry {
// Physics geometry representation
FastPolygonList &FastPolygonList::operator=(const FastPolygonList &other) {
    allocate(other.maxIndices);
    this->size = other.size;
    this->edgeCount = other.edgeCount;
    this->polygonCount = other.polygonCount;

    memcpy(buffer, other.buffer, sizeof(uint32_t) * maxIndices);

    return *this;
}

void FastPolygonList::constructCube() {
    allocate(5 * 6);

    // anti clockwise
    addPolygon(4, 0, 1, 2, 3); // -Z
    addPolygon(4, 7, 6, 5, 4); // +Z
    addPolygon(4, 3, 2, 6, 7); // +Y
    addPolygon(4, 4, 5, 1, 0); // -Y
    addPolygon(4, 5, 6, 2, 1); // +X
    addPolygon(4, 0, 3, 7, 4); // -X
}

void FastPolygonList::allocate(uint32_t maxIdx) {
    maxIndices = maxIdx;
    buffer = (uint32_t *)malloc(sizeof(uint32_t) * maxIndices);
    size = 0;
}

void FastPolygonList::free() {
    ::free(buffer);
}

// Creation
template <typename ...T> void FastPolygonList::addPolygon(uint32_t count, T &&...vertexIndices) {
    uint32_t indexCount = 1 + sizeof...(vertexIndices);
    uint32_t indices[] = {count, (uint32_t)vertexIndices...};

    memcpy(buffer + size, indices, sizeof(uint32_t) * indexCount);

    size += indexCount;

    polygonCount += 1;

    // Should be the amount of half edges there are
    edgeCount += count;
}


void HalfEdgeMesh::construct(
        FastPolygonList &polygons,
        uint32_t vertexCount, math::Vector3 *vertices) {
#ifndef MADRONA_GPU_MODE
    static HalfEdge dummy = {};

    // Allocate all temporary things
    struct Temporary {
        PolygonData *polygons;
        EdgeData *edges;
        HalfEdge *halfEdges;

        // Counters are just for sanity purposes
        uint32_t polygonCount;
        uint32_t edgeCount;
        uint32_t halfEdgeCount;
    };

    Temporary tmp = {};

    // We already know how many polygons there are
    tmp.polygons = (PolygonData *)malloc(sizeof(PolygonData) * polygons.polygonCount);
    tmp.halfEdges = (HalfEdge *)malloc(sizeof(HalfEdge) * polygons.edgeCount);
    // This will be excessive allocation but it's temporary so not big deal
    tmp.edges = (EdgeData *)malloc(sizeof(EdgeData) * polygons.edgeCount);

    std::map<std::pair<VertexID, VertexID>, HalfEdgeID> vtxPairToHalfEdge;

    // Proceed with construction
    for (
            uint32_t polygonIndex = 0, *polygon = polygons.begin();
            polygon != polygons.end();
            polygon = polygons.next(polygon), ++polygonIndex) {
        PolygonData *newPolygon = &tmp.polygons[tmp.polygonCount++];

        uint32_t vtxCount = polygons.getPolygonVertexCount(polygon);

        HalfEdge *prev = &dummy;
        uint32_t firstPolygonHalfEdgeIdx = tmp.halfEdgeCount;

        // Create a half edge for each of these
        for (int vIdx = 0; vIdx < vtxCount; ++vIdx) {
            VertexID a = polygon[vIdx];
            VertexID b = polygon[(vIdx + 1) % vtxCount];

            std::pair<VertexID, VertexID> edge = {a, b};

            if (vtxPairToHalfEdge.find(edge) != vtxPairToHalfEdge.end()) {
                // This should never happen - most likely something wrong with polygon construction
                // and the orientation of the faces' vertex indices
                // PANIC_AND_EXIT
            }
            else {
                // We can allocate a new half edge
                uint32_t hedgeIdx = tmp.halfEdgeCount++;
                HalfEdge *newHalfEdge = &tmp.halfEdges[hedgeIdx];
                newHalfEdge->rootVertex = a;
                newHalfEdge->polygon = polygonIndex;

                // Only set this if the twin was allocated
                std::pair<VertexID, VertexID> twinEdge = {b, a};
                if (auto twin = vtxPairToHalfEdge.find(twinEdge); twin != vtxPairToHalfEdge.end()) {
                    // The twin was allocated!
                    newHalfEdge->twin = twin->second;
                    tmp.halfEdges[twin->second].twin = hedgeIdx;

                    // Only call allocate a new "edge" if the twin was already allocated
                    EdgeData *newEdge = &tmp.edges[tmp.edgeCount++];
                    *newEdge = twin->second;

                    // assert(newHalfEdge->polygon != tmp.halfEdges[newHalfEdge->twin].polygon);
                }

                prev->next = hedgeIdx;
                prev = newHalfEdge;

                // Insert this half edge into temporary set
                vtxPairToHalfEdge[edge] = hedgeIdx;

                // Just make the polygon point to some random half edge in the polygon
                *newPolygon = hedgeIdx;
            }
        }

        prev->next = firstPolygonHalfEdgeIdx;
    }

    // Copy all these to permanent storage in member pointers
#if 0
    mVertices = flAllocv<math::Vector3>(vertexCount);
    memcpy(mVertices, vertices, sizeof(math::Vector3) * vertexCount);
    mVertexCount = vertexCount;

    mTransformedVertices = flAllocv<math::Vector3>(vertexCount);
#endif

    mPolygons = (PolygonData *)malloc(sizeof(PolygonData) * tmp.polygonCount);
    memcpy(mPolygons, tmp.polygons, sizeof(PolygonData) * tmp.polygonCount);
    mPolygonCount = tmp.polygonCount;

    mEdges = (EdgeData *)malloc(sizeof(EdgeData) * tmp.edgeCount);
    memcpy(mEdges, tmp.edges, sizeof(EdgeData) * tmp.edgeCount);
    mEdgeCount = tmp.edgeCount;

    mHalfEdges = (HalfEdge *)malloc(sizeof(HalfEdge) * tmp.halfEdgeCount);
    memcpy(mHalfEdges, tmp.halfEdges, sizeof(HalfEdge) * tmp.halfEdgeCount);
    mHalfEdgeCount = tmp.halfEdgeCount;

    mVertices = (math::Vector3 *)malloc(sizeof(math::Vector3) * vertexCount);
    memcpy(mVertices, vertices, sizeof(math::Vector3) * vertexCount);
    mVertexCount = vertexCount;

    // validate
    for (int i = 0; i < mHalfEdgeCount; ++i) {
        auto &hEdge = halfEdge(i);

        // assert(hEdge.polygon != halfEdge(hEdge.twin).polygon);

        math::Vector3 normal1 = getFaceNormal(hEdge.polygon, vertices);
        math::Vector3 normal2 = getFaceNormal(halfEdge(hEdge.twin).polygon, vertices);

        float d = normal1.dot(normal2);
    }
#endif
}

void HalfEdgeMesh::constructCube() {
    float r = 1.0f;

    math::Vector3 vertices[] = {
        { -r, -r, -r }, // 0
        { +r, -r, -r }, // 1

        { +r, +r, -r }, // 2
        { -r, +r, -r }, // 3

        { -r, -r, +r }, // 4
        { +r, -r, +r }, // 5

        { +r, +r, +r }, // 6
        { -r, +r, +r }, // 7
    };

    FastPolygonList polygons;
    // 5 per face, 6 faces
    polygons.allocate(5 * 6);

    // anti clockwise
    polygons.addPolygon(4, 0, 1, 2, 3); // -Z
    polygons.addPolygon(4, 7, 6, 5, 4); // +Z
    polygons.addPolygon(4, 3, 2, 6, 7); // +Y

    polygons.addPolygon(4, 4, 5, 1, 0); // -Y
    polygons.addPolygon(4, 5, 6, 2, 1); // +X
    polygons.addPolygon(4, 0, 3, 7, 4); // -X

    construct(polygons, 8, vertices);

    polygons.free();
}

math::Vector3 HalfEdgeMesh::getFaceNormal(const PolygonID &polygon, const math::Vector3 *vertices) const {
    math::Vector3 points[3] = {};

    auto *hEdge = &halfEdge(mPolygons[polygon]);
    for (int i = 0; i < 3; ++i) {
        points[i] = vertices[hEdge->rootVertex];
        hEdge = &halfEdge(hEdge->next);
    }

    math::Vector3 a = points[1] - points[0];
    math::Vector3 b = points[2] - points[0];

    return math::cross(b, a).normalize();
}

uint32_t HalfEdgeMesh::getPolygonVertices(
        const PolygonID &polygon, math::Vector3 *polygonVertices, const math::Vector3 *vertices) const {
    // Half edge of the polygon
    uint32_t hEdge = mPolygons[polygon];
    uint32_t start = hEdge;

    uint32_t vertexCounter = 0;
    polygonVertices[vertexCounter++] = vertices[halfEdge(hEdge).rootVertex];

    while (halfEdge(hEdge).next != start) {
        polygonVertices[vertexCounter++] = vertices[halfEdge(halfEdge(hEdge).next).rootVertex];
        hEdge = halfEdge(hEdge).next;
    }

    return vertexCounter;
}

void HalfEdgeMesh::getPolygonVertices(
        math::Vector3 *outVertices,
        const PolygonID &polygon,
        const math::Vector3 *vertices) const {
    // Half edge of the polygon
    uint32_t hEdge = mPolygons[polygon];
    uint32_t start = hEdge;

    uint32_t vertexCounter = 0;
    outVertices[vertexCounter++] = vertices[halfEdge(hEdge).rootVertex];

    while (halfEdge(hEdge).next != start) {
        outVertices[vertexCounter++] = vertices[halfEdge(halfEdge(hEdge).next).rootVertex];
        hEdge = halfEdge(hEdge).next;
    }
}

uint32_t HalfEdgeMesh::getPolygonVertexCount(const PolygonID &polygon) const {
    // Half edge of the polygon
    uint32_t hEdge = mPolygons[polygon];
    uint32_t start = hEdge;

    uint32_t vertexCounter = 1;

    while (halfEdge(hEdge).next != start) {
        vertexCounter++;
        hEdge = halfEdge(hEdge).next;
    }

    return vertexCounter;
}

void HalfEdgeMesh::getPolygonSidePlanes(Plane *planes, const PolygonID &polygon, const math::Vector3 *vertices) const {
    // Half edge of the polygon
    uint32_t hEdge = mPolygons[polygon];
    uint32_t start = hEdge;

    uint32_t vertexCounter = 0;
    planes[vertexCounter++] = getPlane(halfEdge(halfEdge(hEdge).twin).polygon, vertices);

    while (halfEdge(hEdge).next != start) {
        // Should probably improve this readability...
        // Basically get the side plane by getting the twins of each halfedge and querying
        // the plane data of the polygon that the twin is attached to
        planes[vertexCounter++] = getPlane(halfEdge(halfEdge(halfEdge(hEdge).next).twin).polygon, vertices);
        hEdge = halfEdge(hEdge).next;
    }
}

math::Vector3 HalfEdgeMesh::getEdgeDirection(const EdgeData &edge, math::Vector3 *vertices) const {
    auto *hEdge = &halfEdge(edge);

    math::Vector3 a = vertices[hEdge->rootVertex];
    math::Vector3 b = vertices[halfEdge(hEdge->next).rootVertex];

    return (b - a).normalize();
}

Segment HalfEdgeMesh::getEdgeSegment(const EdgeData &edge, math::Vector3 *vertices) const {
    auto *hEdge = &halfEdge(edge);

    math::Vector3 a = vertices[hEdge->rootVertex];
    math::Vector3 b = vertices[halfEdge(hEdge->next).rootVertex];

    return {a, b};
}

math::Vector3 HalfEdgeMesh::getEdgeDirection(const HalfEdge &edge, math::Vector3 *vertices) const {
    math::Vector3 a = vertices[edge.rootVertex];
    math::Vector3 b = vertices[halfEdge(edge.next).rootVertex];

    return (b - a).normalize();
}

math::Vector3 HalfEdgeMesh::getEdgeOrigin(const EdgeData &edge, math::Vector3 *vertices) const {
    return vertices[halfEdge(edge).rootVertex];
}

math::Vector3 HalfEdgeMesh::getEdgeOrigin(const HalfEdge &edge, math::Vector3 *vertices) const {
    return vertices[edge.rootVertex];
}

Plane HalfEdgeMesh::getPlane(const PolygonID &polygon, const math::Vector3 *vertices) const {
    Plane plane = {};
    plane.normal = getFaceNormal(polygon, vertices);
    plane.point = vertices[halfEdge(mPolygons[polygon]).rootVertex];
    return plane;
}

std::pair<math::Vector3, math::Vector3> HalfEdgeMesh::getEdgeNormals(const HalfEdge &hEdge, math::Vector3 *vertices) const {
    math::Vector3 normal1 = getFaceNormal(hEdge.polygon, vertices);
    math::Vector3 normal2 = getFaceNormal(halfEdge(hEdge.twin).polygon, vertices);

    return {normal1, normal2};
}

uint32_t HalfEdgeMesh::getPolygonCount() const {
    return mPolygonCount;
}

const PolygonData &HalfEdgeMesh::polygon(uint32_t id) const {
    return mPolygons[id];
}

uint32_t HalfEdgeMesh::getEdgeCount() const {
    return mEdgeCount;
}

const EdgeData &HalfEdgeMesh::edge(uint32_t id) const {
    return mEdges[id];
}

// Can pass a polygon data into here - polygon data is just half edge ID
const HalfEdge &HalfEdgeMesh::halfEdge(HalfEdgeID id) const {
    return mHalfEdges[id];
}

uint32_t HalfEdgeMesh::getVertexCount() const {
    return mVertexCount;
}

const math::Vector3 &HalfEdgeMesh::vertex(uint32_t id) const {
    return mVertices[id];
}

}

}

