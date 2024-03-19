#pragma once

#include <madrona/math.hpp>

namespace madrona::geo {

struct HalfEdge {
    uint32_t next;
    uint32_t rootVertex;
    uint32_t face;
};

struct Plane {
    math::Vector3 normal; // Potentially unnormalized
    float d;
};

struct Segment {
    math::Vector3 p1;
    math::Vector3 p2;
};

struct HalfEdgeMesh {
    template <typename Fn>
    inline void iterateFaceIndices(uint32_t face, Fn &&fn) const;
    inline uint32_t twinIDX(uint32_t half_edge_id) const;
    inline uint32_t numEdges() const;
    inline uint32_t edgeToHalfEdge(uint32_t edge_id) const;

    HalfEdge *halfEdges;
    uint32_t *faceBaseHalfEdges;
    Plane *facePlanes;
    math::Vector3 *vertices;

    uint32_t numHalfEdges;
    uint32_t numFaces;
    uint32_t numVertices;
};

// Sphere at origin, ray_d must be normalized
inline float intersectRayOriginSphere(
    math::Vector3 ray_o,
    math::Vector3 ray_d,
    float r);

// Assumes (0, 0, 0) is at the base of the capsule line segment, and that
// ray_d is normalized.
// h is the length of the line segment (not overall height of the capsule).
inline float intersectRayZOriginCapsule(
    math::Vector3 ray_o,
    math::Vector3 ray_d,
    float r,
    float h);

// Returns non-unit normal
inline math::Vector3 computeTriangleGeoNormal(
    math::Vector3 ab,
    math::Vector3 ac,
    math::Vector3 bc);

inline math::Vector3 triangleClosestPointToOrigin(
    math::Vector3 a,
    math::Vector3 b,
    math::Vector3 c,
    math::Vector3 ab,
    math::Vector3 ac);

// Returns distance to closest point squared + closest point itself
// in *closest_point. If the hull is touching the origin returns 0 and
// *closest_point is invalid.
float hullClosestPointToOriginGJK(
    HalfEdgeMesh &hull,
    float err_tolerance2,
    math::Vector3 *closest_point);

}

#include "geo.inl"
