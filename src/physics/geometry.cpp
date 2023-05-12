#include <madrona/physics.hpp>
#include <madrona/context.hpp>

namespace madrona::phys::geometry {

using namespace math;

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
