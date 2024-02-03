#pragma once

#include <madrona/math.hpp>

namespace madrona::geo {

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

}

#include "geo.inl"
