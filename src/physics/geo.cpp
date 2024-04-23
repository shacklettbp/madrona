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

float hullClosestPointToOriginGJK(
    HalfEdgeMesh &hull,
    float err_tolerance2,
    Vector3 *closest_point)
{
    auto supportFn = [&hull]
    (Vector3 v, Vector3 *, Vector3 *)
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

}
