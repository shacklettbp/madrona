#ifndef MADRONA_GPU_MODE
#include <cstdlib>
#include <map>
#endif

#include <madrona/physics.hpp>
#include <madrona/context.hpp>

namespace madrona::phys::geometry {

using namespace math;

void HalfEdgeMesh::construct(
        const math::Vector3 *vert_positions,
        CountT num_vertices, 
        const uint32_t *indices,
        const uint32_t *face_counts,
        CountT num_faces)
{
#ifndef MADRONA_GPU_MODE
    static HalfEdge dummy = {};

    mHalfEdgeCount = 0;
    mPolygonCount = num_faces;

    for (CountT face_idx = 0; face_idx < num_faces; face_idx++) {
        mHalfEdgeCount += face_counts[face_idx];
    }

    assert(mHalfEdgeCount % 2 == 0);

    mEdgeCount = mHalfEdgeCount / 2;
    mVertexCount = num_vertices;
    
    // We already know how many polygons there are
    mPolygons = (PolygonData *)malloc(sizeof(PolygonData) * num_faces);
    mFacePlanes = (Plane *)malloc(sizeof(Plane) * num_faces);
    mEdges = (EdgeData *)malloc(sizeof(EdgeData) * mEdgeCount);
    mHalfEdges = (HalfEdge *)malloc(sizeof(HalfEdge) * mHalfEdgeCount);
    mVertices = (math::Vector3 *)malloc(sizeof(math::Vector3) * mVertexCount);

    memcpy(mVertices, vert_positions, sizeof(math::Vector3) * mVertexCount);

    // FIXME: std::map??
    std::map<std::pair<VertexID, VertexID>, HalfEdgeID> vtxPairToHalfEdge;

    // Proceed with construction
    const uint32_t *cur_face_indices = indices;
    uint32_t num_initialized_hedges = 0;
    uint32_t num_initialized_edges = 0;
    for (CountT face_idx = 0; face_idx < num_faces; face_idx++) {
        PolygonData *newPolygon = &mPolygons[face_idx];

        CountT vtx_count = face_counts[face_idx];

        HalfEdge *prev = &dummy;
        uint32_t firstPolygonHalfEdgeIdx = num_initialized_hedges;

        // Create a half edge for each of these
        for (CountT v_idx = 0; v_idx < vtx_count; ++v_idx) {
            VertexID a = cur_face_indices[v_idx];
            VertexID b = cur_face_indices[((v_idx + 1) % vtx_count)];

            std::pair<VertexID, VertexID> edge = {a, b};

            if (vtxPairToHalfEdge.find(edge) != vtxPairToHalfEdge.end()) {
                // This should never happen - most likely something wrong with
                // polygon construction and the orientation of the faces'
                // vertex indices
                FATAL("Invalid input mesh to halfedge construction");
            }
            else {
                // We can allocate a new half edge
                uint32_t hedge_idx = num_initialized_hedges++;
                HalfEdge *new_half_edge = &mHalfEdges[hedge_idx];
                new_half_edge->rootVertex = a;
                new_half_edge->polygon = face_idx;

                // Only set this if the twin was allocated
                std::pair<VertexID, VertexID> twin_edge = {b, a};
                if (auto twin = vtxPairToHalfEdge.find(twin_edge);
                        twin != vtxPairToHalfEdge.end()) {
                    // The twin was allocated!
                    new_half_edge->twin = twin->second;
                    mHalfEdges[twin->second].twin = hedge_idx;

                    // Only call allocate a new "edge" if the twin was already allocated
                    EdgeData *new_edge = &mEdges[num_initialized_edges++];
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
        auto *h_edge = &mHalfEdges[*newPolygon];
        for (CountT i = 0; i < 3; ++i) {
            face_points[i] = vert_positions[h_edge->rootVertex];
            h_edge = &mHalfEdges[h_edge->next];
        }

        math::Vector3 a = face_points[1] - face_points[0];
        math::Vector3 b = face_points[2] - face_points[0];

        Vector3 n = math::cross(a, b).normalize();

        mFacePlanes[face_idx] = Plane {
            n,
            dot(n, face_points[0]),
        };

        cur_face_indices += vtx_count;
    }
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
