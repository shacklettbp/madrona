#pragma once

#include <madrona/math.hpp>

namespace madrona::geo {

inline float intersect2DRayOriginCircle(
    math::Vector2 o,
    math::Vector2 d,
    float r);

inline float intersectRayOriginSphere(
    math::Vector3 ray_o,
    math::Vector3 ray_d,
    float r);

// Assumes (0, 0, 0) is at the base of the capsule line segment.
// h is the length of the line segment (not overall height of the capsule).
inline float intersectRayZOriginCapsule(
    math::Vector3 ray_o,
    math::Vector3 ray_d,
    float r,
    float h);

}

#include "geo.inl"
