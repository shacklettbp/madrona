#ifndef MADRONA_GPU_MODE
#include <cstdlib>
#include <memory.h>
#include <map>
#endif

#include <madrona/physics.hpp>
#include <madrona/context.hpp>

namespace madrona::phys::geometry {

using namespace math;

// Physics geometry representation
FastPolygonList &FastPolygonList::operator=(const FastPolygonList &other) {
    allocate(other.maxIndices);
    this->size = other.size;
    this->edgeCount = other.edgeCount;
    this->polygonCount = other.polygonCount;

    memcpy(buffer, other.buffer, sizeof(uint32_t) * maxIndices);

    return *this;
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
void FastPolygonList::addPolygon(Span<const uint32_t> indices)
{
    uint32_t index_count = indices.size();

    buffer[size] = index_count;

    memcpy(buffer + size + 1, indices.data(), sizeof(uint32_t) * index_count);

    size += index_count + 1;

    polygonCount += 1;

    // Should be the amount of half edges there are
    edgeCount += index_count;
}

void HalfEdgeMesh::construct(
        FastPolygonList &polygons,
        uint32_t vertexCount, const math::Vector3 *vertices) {
#ifndef MADRONA_GPU_MODE
    static HalfEdge dummy = {};

    // Allocate all temporary things
    struct Temporary {
        PolygonData *polygons;
        Plane *facePlanes;
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
    tmp.facePlanes = (Plane *)malloc(sizeof(Plane ) * polygons.polygonCount);
    tmp.halfEdges = (HalfEdge *)malloc(sizeof(HalfEdge) * polygons.edgeCount);
    // This will be excessive allocation but it's temporary so not big deal
    tmp.edges = (EdgeData *)malloc(sizeof(EdgeData) * polygons.edgeCount);

    // FIXME: std::map??
    std::map<std::pair<VertexID, VertexID>, HalfEdgeID> vtxPairToHalfEdge;

    // Proceed with construction
    for (uint32_t poly_idx = 0, *polygon = polygons.begin();
            polygon != polygons.end();
            polygon = polygons.next(polygon), ++poly_idx) {
        PolygonData *newPolygon = &tmp.polygons[tmp.polygonCount++];

        CountT vtx_count = polygons.getPolygonVertexCount(polygon);

        HalfEdge *prev = &dummy;
        uint32_t firstPolygonHalfEdgeIdx = tmp.halfEdgeCount;

        // Create a half edge for each of these
        for (CountT v_idx = 0; v_idx < vtx_count; ++v_idx) {
            VertexID a = polygon[v_idx];
            VertexID b = polygon[(v_idx + 1) % vtx_count];

            std::pair<VertexID, VertexID> edge = {a, b};

            if (vtxPairToHalfEdge.find(edge) != vtxPairToHalfEdge.end()) {
                // This should never happen - most likely something wrong with
                // polygon construction and the orientation of the faces'
                // vertex indices
                FATAL("Invalid input mesh to halfedge construction");
            }
            else {
                // We can allocate a new half edge
                uint32_t hedge_idx = tmp.halfEdgeCount++;
                HalfEdge *new_half_edge = &tmp.halfEdges[hedge_idx];
                new_half_edge->rootVertex = a;
                new_half_edge->polygon = poly_idx;

                // Only set this if the twin was allocated
                std::pair<VertexID, VertexID> twin_edge = {b, a};
                if (auto twin = vtxPairToHalfEdge.find(twin_edge);
                        twin != vtxPairToHalfEdge.end()) {
                    // The twin was allocated!
                    new_half_edge->twin = twin->second;
                    tmp.halfEdges[twin->second].twin = hedge_idx;

                    // Only call allocate a new "edge" if the twin was already allocated
                    EdgeData *new_edge = &tmp.edges[tmp.edgeCount++];
                    *new_edge = twin->second;

                    // assert(new_half_edge->polygon != tmp.halfEdges[new_half_edge->twin].polygon);
                }

                prev->next = hedge_idx;
                prev = new_half_edge;

                // Insert this half edge into temporary set
                vtxPairToHalfEdge[edge] = hedge_idx;

                // Just make the polygon point to some random half edge in the polygon
                *newPolygon = hedge_idx;
            }
        }
        prev->next = firstPolygonHalfEdgeIdx;

        math::Vector3 face_points[3];
        auto *h_edge = &tmp.halfEdges[*newPolygon];
        for (CountT i = 0; i < 3; ++i) {
            face_points[i] = vertices[h_edge->rootVertex];
            h_edge = &tmp.halfEdges[h_edge->next];
        }

        math::Vector3 a = face_points[1] - face_points[0];
        math::Vector3 b = face_points[2] - face_points[0];

        Vector3 n = math::cross(a, b).normalize();

        tmp.facePlanes[poly_idx] = Plane {
            n,
            dot(n, face_points[0]),
        };
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

    mFacePlanes = (geometry::Plane *)malloc(
        sizeof(geometry::Plane) * tmp.polygonCount);
    memcpy(mFacePlanes, tmp.facePlanes, sizeof(geometry::Plane) * tmp.polygonCount);

    mEdges = (EdgeData *)malloc(sizeof(EdgeData) * tmp.edgeCount);
    memcpy(mEdges, tmp.edges, sizeof(EdgeData) * tmp.edgeCount);
    mEdgeCount = tmp.edgeCount;

    mHalfEdges = (HalfEdge *)malloc(sizeof(HalfEdge) * tmp.halfEdgeCount);
    memcpy(mHalfEdges, tmp.halfEdges, sizeof(HalfEdge) * tmp.halfEdgeCount);
    mHalfEdgeCount = tmp.halfEdgeCount;

    mVertices = (math::Vector3 *)malloc(sizeof(math::Vector3) * vertexCount);
    memcpy(mVertices, vertices, sizeof(math::Vector3) * vertexCount);
    mVertexCount = vertexCount;

    free(tmp.polygons);
    free(tmp.facePlanes);
    free(tmp.halfEdges);
    free(tmp.edges);
#endif
}

// FIXME: validate where this function needs and doesn't need to
// return a normalized result
math::Vector3 HalfEdgeMesh::getFaceNormal(PolygonID polygon,
                                          const math::Vector3 *vertices) const
{
    math::Vector3 points[3];

    auto *h_edge = &halfEdge(mPolygons[polygon]);
#pragma unroll
    for (CountT i = 0; i < 3; ++i) {
        points[i] = vertices[h_edge->rootVertex];
        h_edge = &halfEdge(h_edge->next);
    }

    math::Vector3 a = points[1] - points[0];
    math::Vector3 b = points[2] - points[0];

    return math::cross(a, b).normalize();
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

// FIXME: not clear if this belongs in HalfEdgeMesh - current interface forces
// temporary planes buffer
void HalfEdgeMesh::getPolygonSidePlanes(Plane *planes, const PolygonID &polygon,
                                        const math::Vector3 *vertices) const
{
    math::Vector3 polygonNormal = getFaceNormal(polygon, vertices);

    // Half edge of the polygon
    uint32_t hEdge = mPolygons[polygon];
    uint32_t start = hEdge;

    uint32_t vertexCounter = 0;

    do {
        Vector3 plane_point = vertices[halfEdge(hEdge).rootVertex];
        Vector3 plane_normal = 
            cross(getEdgeDirection(hEdge, vertices), polygonNormal);

        float d = dot(plane_normal, plane_point);

        planes[vertexCounter++] = {
            plane_normal,
            d,
        };

        hEdge = halfEdge(hEdge).next;
    } while (hEdge != start);
}

math::Vector3 HalfEdgeMesh::getEdgeDirection(const EdgeData &edge, const math::Vector3 *vertices) const {
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

Plane HalfEdgeMesh::getPlane(PolygonID polygon,
                             const math::Vector3 *vertices) const
{
    // FIXME: this doesn't need to be normalized, but getFaceNormal normalizes
    // the output currently
    Vector3 normal = getFaceNormal(polygon, vertices);
    Vector3 point = vertices[halfEdge(mPolygons[polygon]).rootVertex];
    float d = dot(normal, point);

    return {
        normal,
        d,
    };
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

}
