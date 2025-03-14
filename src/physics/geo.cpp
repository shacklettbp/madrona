#include <madrona/geo.hpp>

//#define MADRONA_GJK_DEBUG
#include "gjk.hpp"

namespace madrona::geo {

using namespace math;

MADRONA_ALWAYS_INLINE static inline
Vector3 getHullSupportPointGJK(
    HalfEdgeMesh &hull,
    Vector3 v)
{
    // FIXME, upgrade HalfEdgeMesh to support getting neighboring vertices to
    // speed this up with hill climbing. Note that triangulated faces
    // are faster for hill climbing across edges since they allow large
    // faces to be shortcut rather than requiring traversal fully around the
    // edge. Refer Collision Detection in Interactive 3D environments 4.3.4
    // for more details / tips.

    float max_dot = -FLT_MAX;
    Vector3 support;
    const CountT num_verts = (CountT)hull.numVertices;
    for (CountT i = 0; i < num_verts; i++) {
        Vector3 w = hull.vertices[i];

        float w_dot_v = dot(w, v);
        if (w_dot_v > max_dot) {
            max_dot = w_dot_v;
            support = w;
        }
    }

    return support;
}

MADRONA_ALWAYS_INLINE static inline
Vector3 getHullSupportPointSegmentGJK(
    HalfEdgeMesh &hull,
    Vector3 v,
    Vector3 seg_p1,
    Vector3 seg_p2,
    float seg_len,
    Vector3 *support_hull_out = nullptr)
{
    float max_dot_hull = -FLT_MAX;
    Vector3 support_hull;
    const CountT num_verts = (CountT)hull.numVertices;
    for (CountT i = 0; i < num_verts; i++) {
        Vector3 w = hull.vertices[i];

        float w_dot_v = dot(w, v);
        if (w_dot_v > max_dot_hull) {
            max_dot_hull = w_dot_v;
            support_hull = w;
        }
    }

    Vector3 support_seg;
    { // Get support point for segment
        float p1_dot_minus_v = dot(seg_p1, -v);
        float p2_dot_minus_v = dot(seg_p2, -v);

        if (p1_dot_minus_v > p2_dot_minus_v) {
            support_seg = seg_p1;
        } else {
            support_seg = seg_p2;
        }
    }

    if (support_hull_out) {
        *support_hull_out = support_hull;
    }

    return support_hull - support_seg;
}

float hullClosestPointToOriginGJK(
    HalfEdgeMesh &hull,
    float err_tolerance2,
    Vector3 *closest_point)
{
    auto supportFn = [&hull] (Vector3 v, Vector3 *, Vector3 *)
    {
        return getHullSupportPointGJK(hull, v);
    };

    GJKWithoutPoints gjk;
    float dist2 = gjk.computeDistance2(
        supportFn, -hull.vertices[0], err_tolerance2);

    // Since out supportFn just returns points on the hull directly,
    // we can shortcut the closest point computation and just compute
    // it as -v, since v is the vector from the closest point to the origin.
    *closest_point = -gjk.v;

    return dist2;
}

float hullClosestPointToSegmentGJK(
    HalfEdgeMesh &hull,
    float err_tolerance2,
    Vector3 p1,
    Vector3 p2,
    Vector3 *closest_point)
{
#if 1
    float seg_len = (p2 - p1).length();
    auto supportFn = [&hull, &p1, &p2, seg_len] (Vector3 v, Vector3 *, Vector3 *)
    {
        return getHullSupportPointSegmentGJK(
                hull, v, p1, p2, seg_len);
    };
#endif

    GJKWithoutPoints gjk;
    gjk.computeDistance2(
        supportFn, -hull.vertices[0], err_tolerance2);

    getHullSupportPointSegmentGJK(
            hull, gjk.v, p1, p2, seg_len, closest_point);

    float dist2 = 0.f;
    {
        Vector3 a = p1;
        Vector3 b = p2;
        Vector3 ab = b - a;
        Vector3 ap = *closest_point - a;

        float abLength2 = ab.dot(ab);
        if (abLength2 == 0.0f) {
            Vector3 closestPoint = a;
            return (*closest_point - a).length();
        }

        float t = ap.dot(ab) / abLength2;
        t = std::clamp(t, 0.0f, 1.0f);

        Vector3 closestPoint = a + ab * t;

        dist2 = (*closest_point - closestPoint).length2();
    }

    // Since out supportFn just returns points on the hull directly,
    // we can shortcut the closest point computation and just compute
    // it as -v, since v is the vector from the closest point to the origin.
    //*closest_point = -gjk.v;
    //
    //V

    return dist2;
}

}
