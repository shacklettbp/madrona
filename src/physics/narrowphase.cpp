#include <madrona/memory.hpp>
#include <madrona/physics.hpp>
#include <madrona/context.hpp>

#include "physics_impl.hpp"

#ifdef MADRONA_GPU_MODE
#include <madrona/mw_gpu/cu_utils.hpp>
#endif

namespace madrona::phys::narrowphase {

using namespace base;
using namespace math;
using namespace geometry;

enum class NarrowphaseTest : uint32_t {
    SphereSphere = 1,
    HullHull = 2,
    SphereHull = 3,
    PlanePlane = 4,
    SpherePlane = 5,
    HullPlane = 6,
};

struct FaceQuery {
    float separation;
    int32_t faceIdx;
};

struct EdgeQuery {
    float separation;
    math::Vector3 normal;
    int32_t edgeIdxA;
    int32_t edgeIdxB;
};

struct HullState {
    const Vector3 *vertices;
    const Plane *facePlanes;
    const HalfEdge *halfEdges;
    const EdgeData *edgeIndices; // FIXME: optimize HE mesh
    const PolygonData *faceEdgeIndices;
    int32_t numVertices;
    int32_t numFaces;
    int32_t numEdges;
    Vector3 center;
};

struct Manifold {
    math::Vector3 contactPoints[4];
    float penetrationDepths[4];
    int32_t numContactPoints;
    math::Vector3 normal;
    bool aIsReference;
};

static HullState makeHullState(
    MADRONA_GPU_COND(const int32_t mwgpu_lane_id,)
    const math::Vector3 *obj_vertices,
    const Plane *obj_planes,
    const HalfEdge *half_edges,
    const EdgeData *edge_indices,
    const PolygonData *face_edge_indices,
    CountT num_vertices,
    CountT num_faces,
    CountT num_edges,
    Vector3 translation,
    Quat rotation,
    Diag3x3 scale,
    math::Vector3 *dst_vertices,
    Plane *dst_planes)
{
    if (dst_vertices == nullptr) {
        return HullState {
            obj_vertices,
            obj_planes,
            half_edges,
            edge_indices,
            face_edge_indices,
            (int32_t)num_vertices,
            (int32_t)num_faces,
            (int32_t)num_edges,
            translation,
        };
    }

    Mat3x3 unscaled_rot = Mat3x3::fromQuat(rotation);
    Mat3x3 vertex_txfm = unscaled_rot * scale;
    Mat3x3 normal_txfm = unscaled_rot * scale.inv();

#ifdef MADRONA_GPU_MODE
    const CountT start_offset = mwgpu_lane_id;
    constexpr CountT elems_per_iter = mwGPU::numWarpThreads;
#else
    constexpr CountT start_offset = 0;
    constexpr CountT elems_per_iter = 1;
#endif

    for (CountT i = start_offset; i < num_vertices; i += elems_per_iter) {
        dst_vertices[i] = vertex_txfm * obj_vertices[i] + translation;
    }

    // FIXME: could significantly optimize this with a uniform scale
    // version
    for (CountT i = start_offset; i < num_faces; i += elems_per_iter) {
        Plane obj_plane = obj_planes[i];
        Vector3 plane_origin =
            vertex_txfm * (obj_plane.normal * obj_plane.d) + translation;

        Vector3 txfmed_normal = (normal_txfm * obj_plane.normal).normalize();
        float new_d = dot(txfmed_normal, plane_origin);

        dst_planes[i] = {
            txfmed_normal,
            new_d,
        };
    }

    return HullState {
        dst_vertices,
        dst_planes,
        half_edges,
        edge_indices,
        face_edge_indices,
        (int32_t)num_vertices,
        (int32_t)num_faces,
        (int32_t)num_edges,
        translation,
    };
}

static HullState makeHullState(
    MADRONA_GPU_COND(const int32_t mwgpu_lane_id,)
    const geometry::HalfEdgeMesh &he_mesh,
    Vector3 translation,
    Quat rotation,
    Diag3x3 scale,
    math::Vector3 *dst_vertices,
    Plane *dst_planes)
{
    return makeHullState(
        MADRONA_GPU_COND(mwgpu_lane_id,)
        he_mesh.mVertices,
        he_mesh.mFacePlanes,
        he_mesh.mHalfEdges,
        he_mesh.mEdges,
        he_mesh.mPolygons,
        he_mesh.mVertexCount,
        he_mesh.mPolygonCount,
        he_mesh.mEdgeCount,
        translation, rotation, scale,
        dst_vertices, dst_planes);
}

// Returns the signed distance
static inline float getDistanceFromPlane(
    const Plane &plane, const Vector3 &a)
{
    float adotn = a.dot(plane.normal);
    return (adotn - plane.d);
}

// Need to be normalized
static inline bool areParallel(const math::Vector3 &a,
                               const math::Vector3 &b)
{
    float d = fabsf(a.dot(b));

    return fabsf(d - 1.0f) < 0.0001f;
}

// Get intersection on plane of the line passing through 2 points
inline math::Vector3 planeIntersection(const geometry::Plane &plane, const math::Vector3 &p1, const math::Vector3 &p2) {
    float distance = getDistanceFromPlane(plane, p1);

    return p1 + (p2 - p1) * (-distance / plane.normal.dot(p2 - p1));
}

static Vector3 findFurthestPoint(const HullState &h,
                                 const math::Vector3 &d)
{
    Vector3 furthest = h.vertices[0];
    float max_dist = d.dot(h.vertices[0]);

    for (CountT i = 1; i < (CountT)h.numVertices; ++i) {
        Vector3 vertex = h.vertices[i];
        float dp = d.dot(vertex);
        if (dp > max_dist) {
            max_dist = dp;
            furthest = vertex;
        }
    }

    return furthest;
}

#ifdef MADRONA_GPU_MODE

static inline MADRONA_ALWAYS_INLINE std::pair<float, int32_t> 
warpFloatMaxAndIdx(float val, int32_t idx)
{
#pragma unroll
    for (int32_t w = 16; w >= 1; w /= 2) {
        float other_val =
            __shfl_xor_sync(mwGPU::allActive, val, w);
        int32_t other_idx =
            __shfl_xor_sync(mwGPU::allActive, idx, w);

        if (other_val > val) {
            val = other_val;
            idx = other_idx;
        }
    }

    return { val, idx };
}

static inline MADRONA_ALWAYS_INLINE std::pair<float, int32_t>
warpFloatMinAndIdx(float val, int32_t idx)
{
#pragma unroll
    for (int32_t w = 16; w >= 1; w /= 2) {
        float other_val =
            __shfl_xor_sync(mwGPU::allActive, val, w);
        int32_t other_idx =
            __shfl_xor_sync(mwGPU::allActive, idx, w);

        if (other_val < val) {
            val = other_val;
            idx = other_idx;
        }
    }

    return { val, idx };
}

static Vector3 findFurthestPointWide(
    const int32_t mwgpu_lane_id,
    const HullState &h, const math::Vector3 &d)
{
    Vector3 furthest;
    float max_dist = -FLT_MAX;

    const int32_t num_verts = (int32_t)h.numVertices;
    for (int32_t offset = 0; offset < num_verts; offset += 32) {
        int32_t idx = offset + mwgpu_lane_id;

        Vector3 vertex;
        float dp;
        if (idx < num_verts) {
            vertex = h.vertices[idx];
            dp = d.dot(vertex);
        } else {
            dp = -FLT_MAX;
        }

        auto [iter_max, max_lane] = warpFloatMaxAndIdx(dp, idx);

        if (iter_max > max_dist) {
            max_dist = iter_max;
            furthest.x =
                __shfl_sync(mwGPU::allActive, vertex.x, max_lane);
            furthest.y =
                __shfl_sync(mwGPU::allActive, vertex.y, max_lane);
            furthest.z =
                __shfl_sync(mwGPU::allActive, vertex.z, max_lane);
        }
    }

    return furthest;
}
#endif

static float getHullDistanceFromPlane(
    MADRONA_GPU_COND(const int32_t mwgpu_lane_id,)
    const Plane &plane, const HullState &h)
{
#ifdef MADRONA_GPU_MODE
    math::Vector3 supportA = findFurthestPointWide(
        mwgpu_lane_id, h, -plane.normal);
#else
    math::Vector3 supportA = findFurthestPoint(h, -plane.normal);
#endif
    
    return getDistanceFromPlane(plane, supportA);
}

static FaceQuery queryFaceDirections(
    MADRONA_GPU_COND(int32_t mwgpu_lane_id,)
    const HullState &a, const HullState &b)
{
#ifdef MADRONA_GPU_MODE
    constexpr CountT elems_per_iter = 32;
#else
    constexpr CountT elems_per_iter = 1;
#endif

    int polygonMaxDistance = 0;
    float maxDistance = -FLT_MAX;

    auto distanceFromAFace = [&a, &b](CountT face_idx) {
        Plane plane = a.facePlanes[face_idx];
        math::Vector3 supportB = findFurthestPoint(b, -plane.normal);
        return getDistanceFromPlane(plane, supportB);
    };

    CountT num_a_faces = (CountT)a.numFaces;
    for (CountT offset = 0; offset < num_a_faces; offset += elems_per_iter) {
#ifdef MADRONA_GPU_MODE
        CountT face_idx = offset + mwgpu_lane_id;

        float face_dist;
        if (face_idx < num_a_faces) {
            face_dist = distanceFromAFace(face_idx);
        } else {
            face_dist = -FLT_MAX;
        }

        auto [iter_max_dist, iter_max_idx] =
            warpFloatMaxAndIdx(face_dist, face_idx);
#else
        const CountT face_idx = offset;
        float iter_max_dist = distanceFromAFace(face_idx);
        const CountT iter_max_idx = face_idx;
#endif

        if (iter_max_dist > maxDistance) {
            maxDistance = iter_max_dist;
            polygonMaxDistance = iter_max_idx;
        }
    }

    return { maxDistance, polygonMaxDistance };
}

static bool isMinkowskiFace(
        const math::Vector3 &a, const math::Vector3 &b,
        const math::Vector3 &c, const math::Vector3 &d)
{
    math::Vector3 bxa = b.cross(a);
    math::Vector3 dxc = d.cross(c);

    float cba = c.dot(bxa);
    float dba = d.dot(bxa);
    float adc = a.dot(dxc);
    float bdc = b.dot(dxc);

    return cba * dba < 0.0f && adc * bdc < 0.0f && cba * bdc > 0.0f;
}

static inline std::pair<Vector3, Vector3> getEdgeNormals(
        const HullState &h, const HalfEdge &h_edge)
{
    Vector3 normal1 = h.facePlanes[h_edge.polygon].normal;
    CountT twin_poly = h.halfEdges[h_edge.twin].polygon;
    Vector3 normal2 = h.facePlanes[twin_poly].normal;

    return { normal1, normal2 };
}

static inline Segment getEdgeSegment(
        const HullState &h, const HalfEdge &h_edge)
{
    Vector3 a = h.vertices[h_edge.rootVertex];
    
    // FIXME: probably should put both vertex indices inline in the half edge
    Vector3 b = h.vertices[h.halfEdges[h_edge.next].rootVertex];

    return { a, b };
}

static inline bool buildsMinkowskiFace(
        const HullState &a, const HullState &b,
        const HalfEdge &edgeA, const HalfEdge &edgeB)
{
    auto [aNormal1, aNormal2] = getEdgeNormals(a, edgeA);
    auto [bNormal1, bNormal2] = getEdgeNormals(b, edgeB);

    return isMinkowskiFace(aNormal1, aNormal2, -bNormal1, -bNormal2);
}

struct EdgeTestResult {
    Vector3 normal;
    float separation;
};

static inline EdgeTestResult edgeDistance(
        const HullState &a, const HullState &b,
        const HalfEdge &edgeA, const HalfEdge &edgeB)
{
    Segment segment_a = getEdgeSegment(a, edgeA);
    Segment segment_b = getEdgeSegment(b, edgeB);

    Vector3 dir_a = segment_a.p2 - segment_a.p1;
    Vector3 dir_b = segment_b.p2 - segment_b.p1;

    if (areParallel(dir_a, dir_b)) {
        EdgeTestResult result;
        result.separation = -FLT_MAX;

        return result;
    }

    math::Vector3 normal = dir_a.cross(dir_b).normalize();

    if (normal.dot(segment_a.p1 - a.center) < 0.0f) {
        normal = -normal;
    }

    float separation = normal.dot(segment_b.p1 - segment_a.p1);

    return {
        normal,
        separation,
    };
}

static EdgeQuery queryEdgeDirections(
    MADRONA_GPU_COND(int32_t mwgpu_lane_id,)
    const HullState &a, const HullState &b)
{
    Vector3 normal;
    int edgeAMaxDistance = 0;
    int edgeBMaxDistance = 0;
    float maxDistance = -FLT_MAX;

    auto testEdgeSeparation = [&a, &b](const HalfEdge &hedge_a,
                                       const HalfEdge &hedge_b) {
        if (buildsMinkowskiFace(a, b, hedge_a, hedge_b)) {
            return edgeDistance(a, b, hedge_a, hedge_b);
        } else {
            EdgeTestResult result;
            result.separation = -FLT_MAX;
            return result;
        }
    };


#ifdef MADRONA_GPU_MODE
    int32_t a_num_edges = a.numEdges;
    int32_t b_num_edges = b.numEdges;
    int32_t num_edge_tests = a_num_edges * b_num_edges;
    for (int32_t edge_offset_linear = 0; edge_offset_linear < num_edge_tests;
         edge_offset_linear += 32) {
        int32_t edge_idx_linear = edge_offset_linear + mwgpu_lane_id;
        int32_t he_a_idx = edge_idx_linear / a_num_edges;
        int32_t he_b_idx = edge_idx_linear % a_num_edges;

        EdgeTestResult edge_cmp;

        if (edge_idx_linear >= num_edge_tests ) {
            edge_cmp.separation = -FLT_MAX;
        } else {
            const HalfEdge &hedge_a = a.halfEdges[he_a_idx]; // FIXME
            const HalfEdge &hedge_b = b.halfEdges[he_b_idx]; // FIXME

            edge_cmp = testEdgeSeparation(hedge_a, hedge_b);
        }

        auto [iter_max_separation, max_lane_idx] =
            warpFloatMaxAndIdx(edge_cmp.separation, mwgpu_lane_id);

        if (iter_max_separation > maxDistance) {
            maxDistance = iter_max_separation;
            normal.x =  __shfl_sync(mwGPU::allActive,
                edge_cmp.normal.x, max_lane_idx);
            normal.y =  __shfl_sync(mwGPU::allActive,
                edge_cmp.normal.y, max_lane_idx);
            normal.z =  __shfl_sync(mwGPU::allActive,
                edge_cmp.normal.z, max_lane_idx);

            edgeAMaxDistance = __shfl_sync(mwGPU::allActive,
                he_a_idx, max_lane_idx);
            edgeBMaxDistance = __shfl_sync(mwGPU::allActive,
                he_b_idx, max_lane_idx);
        }
    }
#else
    for (CountT edgeIdxA = 0; edgeIdxA < (CountT)a.numEdges; ++edgeIdxA) {
        int32_t he_a_idx = a.edgeIndices[edgeIdxA];
        const HalfEdge &hedge_a = a.halfEdges[he_a_idx]; // FIXME

        for (CountT edgeIdxB = 0; edgeIdxB < (CountT)b.numEdges; ++edgeIdxB) {
            int32_t he_b_idx = b.edgeIndices[edgeIdxB];
            const HalfEdge &hedge_b = b.halfEdges[he_b_idx]; // FIXME

            EdgeTestResult edge_cmp = testEdgeSeparation(hedge_a, hedge_b);

            if (edge_cmp.separation > maxDistance) {
                maxDistance = edge_cmp.separation;
                normal = edge_cmp.normal;
                edgeAMaxDistance = he_a_idx;
                edgeBMaxDistance = he_b_idx;
            }
        }
    }
#endif

    return { maxDistance, normal, edgeAMaxDistance, edgeBMaxDistance };
}

static CountT findIncidentFace(MADRONA_GPU_COND(int32_t mwgpu_lane_id,)
    const HullState &h, Vector3 ref_normal)
{
#ifdef MADRONA_GPU_MODE
    constexpr CountT elems_per_iter = 32;
#else
    constexpr CountT elems_per_iter = 1;
#endif

    auto computeFaceDotRef = [&h, ref_normal](CountT face_idx) {
        Plane face_plane = h.facePlanes[face_idx];
        return dot(face_plane.normal, ref_normal);
    };

    float min_dot = FLT_MAX;
    CountT minimizing_face = -1;

    const CountT num_faces = (CountT)h.numFaces;
    for (CountT offset = 0; offset < num_faces; offset += elems_per_iter) {
#ifdef MADRONA_GPU_MODE
        CountT face_idx = offset + mwgpu_lane_id;

        float face_dot_ref;
        if (face_idx < num_faces) {
            face_dot_ref = computeFaceDotRef(face_idx);
        } else{
            face_dot_ref = FLT_MAX;
        }

        auto [iter_min, iter_min_idx] =
            warpFloatMinAndIdx(face_dot_ref, face_idx);
#else
        const CountT face_idx = offset;
        float iter_min = computeFaceDotRef(face_idx);
        const CountT iter_min_idx = face_idx;
#endif

        if (iter_min < min_dot) {
            min_dot = iter_min;
            minimizing_face = iter_min_idx;
        }
    }

    assert(minimizing_face != -1);

    return minimizing_face;
}

static inline CountT clipPolygon(Vector3 *dst_vertices,
                                 Plane clipping_plane,
                                 const Vector3 *input_vertices,
                                 CountT num_input_vertices)
{
    CountT num_new_vertices = 0;

    Vector3 v1 = input_vertices[num_input_vertices - 1];
    float d1 = getDistanceFromPlane(clipping_plane, v1);

    for (CountT i = 0; i < num_input_vertices; ++i) {
        Vector3 v2 = input_vertices[i];
        float d2 = getDistanceFromPlane(clipping_plane, v2);

        if (d1 <= 0.0f && d2 <= 0.0f) {
            // Both vertices are behind the plane, keep the second vertex
            dst_vertices[num_new_vertices++] = v2;
        }
        else if (d1 <= 0.0f && d2 > 0.0f) {
            // v1 is behind the plane, the other is in front (out)
            Vector3 intersection = planeIntersection(clipping_plane, v1, v2);
            dst_vertices[num_new_vertices++] = intersection;
        }
        else if (d2 <= 0.0f && d1 > 0.0f) {
            math::Vector3 intersection = planeIntersection(clipping_plane, v1, v2);
            dst_vertices[num_new_vertices++] = intersection;
            dst_vertices[num_new_vertices++] = v2;
        }

        // Now use v2 as the starting vertex
        v1 = v2;
        d1 = d2;
    }

    return num_new_vertices;
}

static Manifold buildFaceContactManifold(
    MADRONA_GPU_COND(int32_t mwgpu_lane_id,)
    Vector3 contact_normal,
    Vector3 *contacts,
    float *penetration_depths,
    CountT num_contacts,
    bool a_is_ref,
    Vector3 world_offset,
    Quat to_world_frame)
{
#ifdef MADRONA_GPU_MODE
    if (mwgpu_lane_id != 0) {
        // FIXME: there is some warp-level parallelism below
        return Manifold {};
    }
#endif

    Manifold manifold;
    manifold.aIsReference = a_is_ref;
    if (num_contacts <= 4) {
        manifold.numContactPoints = num_contacts;
        for (CountT i = 0; i < num_contacts; i++) {
            manifold.contactPoints[i] = contacts[i];
            manifold.penetrationDepths[i] = penetration_depths[i];
        }
    } else {
        manifold.numContactPoints = 4;
        manifold.contactPoints[0] = contacts[0];
        manifold.penetrationDepths[0] = penetration_depths[0];
        Vector3 point0 = manifold.contactPoints[0];

        // Find furthest contact
        float largestD2 = 0.0f;
        int largestD2ContactPointIdx = 0;
        for (CountT i = 1; i < num_contacts; ++i) {
            Vector3 cur_contact = contacts[i];
            float d2 = point0.distance2(cur_contact);
            if (d2 > largestD2) {
                largestD2 = d2;
                manifold.contactPoints[1] = cur_contact;
                manifold.penetrationDepths[1] = penetration_depths[i];
                largestD2ContactPointIdx = i;
            }
        }

        contacts[largestD2ContactPointIdx] = manifold.contactPoints[0];

        math::Vector3 diff0 = manifold.contactPoints[1] - point0;

        // Find point which maximized area of triangle
        float largestArea = 0.0f;
        int largestAreaContactPointIdx = 0;
        for (CountT i = 1; i < num_contacts; ++i) {
            Vector3 cur_contact = contacts[i];
            math::Vector3 diff1 = cur_contact - point0;
            float area = contact_normal.dot(diff0.cross(diff1));
            if (area > largestArea) {
                manifold.contactPoints[2] = cur_contact;
                manifold.penetrationDepths[2] = penetration_depths[i];
                largestAreaContactPointIdx = i;
            }
        }

        contacts[largestAreaContactPointIdx] = manifold.contactPoints[0];

        for (CountT i = 1; i < num_contacts; ++i) {
            Vector3 cur_contact = contacts[i];
            math::Vector3 diff1 = cur_contact - point0;
            float area = contact_normal.dot(diff0.cross(diff1));
            if (area < largestArea) {
                manifold.contactPoints[3] = cur_contact;
                manifold.penetrationDepths[3] = penetration_depths[i];
            }
        }
    }

    for (CountT i = 0; i < (CountT)manifold.numContactPoints; i++) {
        manifold.contactPoints[i] =
            to_world_frame.rotateVec(manifold.contactPoints[i]) + world_offset;
    }

    manifold.normal = to_world_frame.rotateVec(contact_normal);

    return manifold;
}

static Manifold createFaceContact(MADRONA_GPU_COND(int32_t mwgpu_lane_id,)
                                  FaceQuery faceQueryA, const HullState &a,
                                  FaceQuery faceQueryB, const HullState &b,
                                  void *a_tmp_buf, void *b_tmp_buf,
                                  Vector3 world_offset, Quat to_world_frame)
{
    // Determine minimizing face
    bool a_is_ref = faceQueryA.separation > faceQueryB.separation;
    const FaceQuery &minimizing_query = a_is_ref ? faceQueryA : faceQueryB;

    const HullState &ref_hull = a_is_ref ? a : b;
    const HullState &other_hull = a_is_ref ? b : a;

    auto ref_tmp_buf = (Vector3 *)(a_is_ref ? a_tmp_buf : b_tmp_buf);
    auto other_tmp_buf = (Vector3 *)(a_is_ref ? b_tmp_buf : a_tmp_buf);

    CountT ref_face_idx = minimizing_query.faceIdx;
    Plane ref_plane = ref_hull.facePlanes[ref_face_idx];

    // Find incident face
    CountT incident_face_idx = findIncidentFace(
        MADRONA_GPU_COND(mwgpu_lane_id,) other_hull, ref_plane.normal);

#ifdef MADRONA_GPU_MODE
    // FIXME, there is warp level parallelism available in clipping,
    // contact point filtering etc
    if (mwgpu_lane_id != 0) {
        return Manifold {};
    }
#endif

    // Collect incident vertices: FIXME should have face indices
    CountT num_incident_vertices = 0;
    {
        CountT hedge_idx = other_hull.faceEdgeIndices[incident_face_idx];
        CountT start_hedge_idx = hedge_idx;

        do {
            const auto &cur_hedge = other_hull.halfEdges[hedge_idx];
            hedge_idx = cur_hedge.next;

            Vector3 cur_point = other_hull.vertices[cur_hedge.rootVertex];

            other_tmp_buf[num_incident_vertices++] = cur_point;
        } while (hedge_idx != start_hedge_idx);
    }

    Vector3 *clipping_input = other_tmp_buf;
    CountT num_clipped_vertices = num_incident_vertices;

    Vector3 *clipping_dst = ref_tmp_buf;

    // max output vertices is num_incident_vertices + num planes
    // but we don't know num planes ahead of time without iterating
    // through the reference face twice! Alternative would be to cache the
    // side planes, or store max face size in each mesh. The worst case
    // buffer sizes here is just the sum of the max face sizes - 1

    // FIXME, this code assumes that clipping_input & clipping_dst have space
    // to write incident_vertices + num_planes new vertices

    // Loop over side planes
    {
        CountT hedge_idx = ref_hull.faceEdgeIndices[ref_face_idx];
        CountT start_hedge_idx = hedge_idx;

        auto *cur_hedge = &ref_hull.halfEdges[hedge_idx];
        Vector3 cur_point = ref_hull.vertices[cur_hedge->rootVertex];
        do {
            hedge_idx = cur_hedge->next;
            cur_hedge = &ref_hull.halfEdges[hedge_idx];
            Vector3 next_point = ref_hull.vertices[cur_hedge->rootVertex];

            Vector3 edge = next_point - cur_point;
            Vector3 plane_normal = cross(edge, ref_plane.normal);

            float d = dot(plane_normal, cur_point);
            cur_point = next_point;

            Plane side_plane {
                plane_normal,
                d,
            };

            num_clipped_vertices = clipPolygon(
                clipping_dst, side_plane, clipping_input, num_clipped_vertices);

            std::swap(clipping_dst, clipping_input);
        } while (hedge_idx != start_hedge_idx);
    }

    // clipping_input has the result due to the final swap

    // Filter clipping_input to ones below ref_plane and save penetration depth
    float *penetration_depths = (float *)clipping_dst;

    CountT num_below_plane = 0;
    for (CountT i = 0; i < num_clipped_vertices; ++i) {
        Vector3 vertex = clipping_input[i];
        if (float d = getDistanceFromPlane(ref_plane, vertex); d < 0.0f) {
            // Project the point onto the reference plane (d guaranteed to be negative)
            clipping_input[num_below_plane] = vertex - d * ref_plane.normal;
            penetration_depths[num_below_plane] = -d;

            num_below_plane += 1;
        }
    }

    return buildFaceContactManifold(MADRONA_GPU_COND(mwgpu_lane_id,)
                                    ref_plane.normal, clipping_input,
                                    penetration_depths, num_below_plane,
                                    a_is_ref, world_offset, to_world_frame);
}

static Manifold createFaceContactPlane(MADRONA_GPU_COND(int32_t mwgpu_lane_id,)
                                       const HullState &h,
                                       Plane plane,
                                       Vector3 *contacts_tmp,
                                       float *penetration_depths_tmp,
                                       Vector3 world_offset,
                                       Quat to_world_frame)
{
    // Find incident face
    CountT incident_face_idx =
        findIncidentFace(MADRONA_GPU_COND(mwgpu_lane_id,) h, plane.normal);

#ifdef MADRONA_GPU_MODE
    // FIXME, there is warp level parallelism available below
    if (mwgpu_lane_id != 0) {
        return Manifold {};
    }
#endif

    // Collect incident vertices: FIXME should have face indices
    CountT num_incident_vertices = 0;
    {
        CountT hedge_idx = h.faceEdgeIndices[incident_face_idx];
        CountT start_hedge_idx = hedge_idx;

        do {
            const auto &cur_hedge = h.halfEdges[hedge_idx];
            hedge_idx = cur_hedge.next;
            Vector3 vertex = h.vertices[cur_hedge.rootVertex];

            if (float d = getDistanceFromPlane(plane, vertex); d < 0.0f) {
                // Project the point onto the reference plane
                // (d guaranteed to be negative)
                contacts_tmp[num_incident_vertices] =
                    vertex - d * plane.normal;
                penetration_depths_tmp[num_incident_vertices] = -d;

                num_incident_vertices += 1;
            }
        } while (hedge_idx != start_hedge_idx);
    }

    return buildFaceContactManifold(MADRONA_GPU_COND(mwgpu_lane_id,)
        plane.normal, contacts_tmp,
        penetration_depths_tmp, num_incident_vertices,
        false /* Plane is always reference, always b */,
        world_offset, to_world_frame);
}

static Segment shortestSegmentBetween(const Segment &seg1, const Segment &seg2)
{
    math::Vector3 v1 = seg1.p2 - seg1.p1;
    math::Vector3 v2 = seg2.p2 - seg2.p1;

    math::Vector3  v21 = seg2.p1 - seg1.p1;

    float dotv22 = v2.dot(v2);
    float dotv11 = v1.dot(v1); 
    float dotv21 = v2.dot(v1);
    float dotv211 = v21.dot(v1);
    float dotv212 = v21.dot(v2);

    float denom = dotv21 * dotv21 - dotv22 * dotv11;

    float s, t;

    // FIXME: validate this epsilon
    if (fabsf(denom) < 0.00001f) {
        s = 0.0f;
        t = (dotv11 * s - dotv211) / dotv21;
    }
    else {
        s = (dotv212 * dotv21 - dotv22 * dotv211) / denom;
        t = (-dotv211 * dotv21 + dotv11 * dotv212) / denom;
    }

    s = fmaxf(fminf(s, 1.0f), 0.0f);
    t = fmaxf(fminf(t, 1.0f), 0.0f);

    return { seg1.p1 + s * v1, seg2.p1 + t * v2 };
}

static Manifold createEdgeContact(const EdgeQuery &query,
                                  const HullState &a,
                                  const HullState &b,
                                  Vector3 world_offset,
                                  Quat to_world_frame)
{
    Segment segA = getEdgeSegment(a, a.halfEdges[query.edgeIdxA]);
    Segment segB = getEdgeSegment(b, b.halfEdges[query.edgeIdxB]);

#if 0
    Segment s = shortestSegmentBetween(segA, segB);
    Vector3 contact = 0.5f * (s.p1 + s.p2);
    float depth = 0.5f * (s.p2 - s.p1).length();
#endif

    // Deviation from Gregorius GDC 2015:
    // Currently the solver expects the contact point to be ON object A.
    // For Face-Face this means the point is on object A's face, which is
    // the same as the presentation. In the edge-edge case, the presentation
    // has the contact point between the edges.
    // Our solver currently reconstructs the contact points on point B
    // using the normal depth, so the presentation's method will 
    // reconstruct the wrong contact points.
    // FIXME: revisit this after modifying solver to handle multi-point
    // manifolds better

    // FIXME: don't need this full function call here.
    Segment s = shortestSegmentBetween(segA, segB);
    Vector3 contact = s.p1;

    Manifold manifold;
    manifold.contactPoints[0] = 
        to_world_frame.rotateVec(contact) + world_offset,
    manifold.penetrationDepths[0] = -query.separation;
    manifold.numContactPoints = 1;
    manifold.normal = to_world_frame.rotateVec(query.normal);

    // Normal always points away from object A (enforced in edgeDistance) &
    // contact points are on A.
    manifold.aIsReference = true;

    return manifold;
}

Manifold doSAT(MADRONA_GPU_COND(int32_t mwgpu_lane_id,)
               const HullState &a, const HullState &b,
               void *a_tmp_buf, void *b_tmp_buf,
               Vector3 world_offset, Quat to_world_frame)
{
    Manifold manifold;
    manifold.numContactPoints = 0;

    FaceQuery faceQueryA =
        queryFaceDirections(MADRONA_GPU_COND(mwgpu_lane_id,) a, b);
    if (faceQueryA.separation > 0.0f) {
        // There is a separating axis - no collision
        return manifold;
    }

    FaceQuery faceQueryB =
        queryFaceDirections(MADRONA_GPU_COND(mwgpu_lane_id,) b, a);
    if (faceQueryB.separation > 0.0f) {
        // There is a separating axis - no collision
        return manifold;
    }

    EdgeQuery edgeQuery =
        queryEdgeDirections(MADRONA_GPU_COND(mwgpu_lane_id,) a, b);
    if (edgeQuery.separation > 0.0f) {
        // There is a separating axis - no collision
        return manifold;
    }

    bool bIsFaceContactA = faceQueryA.separation > edgeQuery.separation;
    bool bIsFaceContactB = faceQueryB.separation > edgeQuery.separation;

    if (bIsFaceContactA || bIsFaceContactB) {
        // Create face contact
        manifold = createFaceContact(MADRONA_GPU_COND(mwgpu_lane_id,)
                                     faceQueryA, a, faceQueryB, b,
                                     a_tmp_buf, b_tmp_buf,
                                     world_offset, to_world_frame);
    }
    else {
        // Create edge contact
        manifold = createEdgeContact(edgeQuery, a, b,
                                     world_offset, to_world_frame);
    }

    return manifold;
}

Manifold doSATPlane(MADRONA_GPU_COND(const int32_t mwgpu_lane_id,)
                    const Plane &plane, const HullState &h,
                    void *tmp_buffer1, void *tmp_buffer2,
                    Vector3 world_offset, Quat to_world_frame)
{
    Manifold manifold;
    manifold.numContactPoints = 0;

    float separation = getHullDistanceFromPlane(
        MADRONA_GPU_COND(mwgpu_lane_id,) plane, h);

    if (separation > 0.0f) {
        return manifold;
    }

    return createFaceContactPlane(MADRONA_GPU_COND(mwgpu_lane_id,) h, plane,
                                  (Vector3 *)tmp_buffer1, (float *)tmp_buffer2,
                                  world_offset, to_world_frame);
}

static inline void addContactsToSolver(
    SolverData &solver_data,
    Span<const Contact> added_contacts)
{
    int32_t contact_idx = solver_data.numContacts.fetch_add(
        added_contacts.size(), std::memory_order_relaxed);

    assert(contact_idx < solver_data.maxContacts);
    
    for (CountT i = 0; i < added_contacts.size(); i++) {
        solver_data.contacts[contact_idx + i] = added_contacts[i];
    }
}

static inline void addManifoldToSolver(
    MADRONA_GPU_COND(const int32_t mwgpu_lane_id,)
    SolverData &solver_data,
    Manifold manifold,
    Entity a_entity, Entity b_entity)
{
#ifdef MADRONA_GPU_MODE
    if (mwgpu_lane_id != 0) {
        return;
    }
#endif

    addContactsToSolver(
        solver_data, {{
        manifold.aIsReference ? a_entity : b_entity,
        manifold.aIsReference ? b_entity : a_entity,
        {
            Vector4::fromVector3(manifold.contactPoints[0],
                                 manifold.penetrationDepths[0]),
            Vector4::fromVector3(manifold.contactPoints[1],
                                 manifold.penetrationDepths[1]),
            Vector4::fromVector3(manifold.contactPoints[2],
                                 manifold.penetrationDepths[2]),
            Vector4::fromVector3(manifold.contactPoints[3],
                                 manifold.penetrationDepths[3]),
        },
        manifold.numContactPoints,
        manifold.normal,
        {},
    }});
}

#ifdef MADRONA_GPU_MODE
namespace gpuImpl {
inline constexpr int32_t numSMemFloats =
    mwGPU::SharedMemStorage::numSMemBytesPerWarp / sizeof(float);

// FIXME: do something actually intelligent here
inline constexpr int32_t maxNumPlanes = 40;
inline constexpr int32_t numPlaneFloats = maxNumPlanes * 4;
inline constexpr int32_t numVertexFloats =
    numSMemFloats - numPlaneFloats;
inline constexpr int32_t maxNumVertices = numVertexFloats / 3;
}
#endif

inline void runNarrowphase(
    Context &ctx,
    const CandidateCollision &candidate_collision)
{
#ifdef MADRONA_GPU_MODE
    const int32_t mwgpu_warp_id = threadIdx.x / 32;
    const int32_t mwgpu_lane_id = threadIdx.x % 32;

    Plane *tmp_faces;
    Vector3 *tmp_vertices;
    constexpr int32_t max_num_tmp_faces = gpuImpl::maxNumPlanes;
    constexpr int32_t max_num_tmp_vertices =
        gpuImpl::maxNumVertices;
    {
        auto smem_buf = (char *)mwGPU::SharedMemStorage::buffer;
        char *warp_smem_base = smem_buf + 
            mwGPU::SharedMemStorage::numSMemBytesPerWarp * mwgpu_warp_id;

        tmp_faces = (Plane *)warp_smem_base;
        tmp_vertices = (Vector3 *)(tmp_faces + gpuImpl::maxNumPlanes);
    }

#else
    constexpr int32_t max_num_tmp_faces = 512;
    constexpr int32_t max_num_tmp_vertices = 512;

    Plane tmp_faces_buffer[max_num_tmp_faces];
    Vector3 tmp_vertices_buffer[max_num_tmp_vertices];

    Plane *tmp_faces = tmp_faces_buffer;
    Vector3 *tmp_vertices = tmp_vertices_buffer;
#endif

    Entity a_entity = candidate_collision.a;
    Entity b_entity = candidate_collision.b;

    const ObjectManager &obj_mgr = *ctx.getSingleton<ObjectData>().mgr;

    ObjectID a_obj = ctx.getUnsafe<ObjectID>(a_entity);
    ObjectID b_obj = ctx.getUnsafe<ObjectID>(b_entity);

    const CollisionPrimitive *a_prim = &obj_mgr.primitives[a_obj.idx];
    const CollisionPrimitive *b_prim = &obj_mgr.primitives[b_obj.idx];

    uint32_t raw_type_a = static_cast<uint32_t>(a_prim->type);
    uint32_t raw_type_b = static_cast<uint32_t>(b_prim->type);

    // Swap a & b to be properly ordered based on object type
    if (raw_type_a > raw_type_b) {
        std::swap(a_entity, b_entity);
        std::swap(a_obj, b_obj);
        std::swap(a_prim, b_prim);
        std::swap(raw_type_a, raw_type_b);
    }

    Vector3 a_pos = ctx.getUnsafe<Position>(a_entity);
    Vector3 b_pos = ctx.getUnsafe<Position>(b_entity);
    Quat a_rot = ctx.getUnsafe<Rotation>(a_entity);
    Quat b_rot = ctx.getUnsafe<Rotation>(b_entity);
    Diag3x3 a_scale(ctx.getUnsafe<Scale>(a_entity));
    Diag3x3 b_scale(ctx.getUnsafe<Scale>(b_entity));

    {
        AABB a_obj_aabb = obj_mgr.aabbs[a_obj.idx];
        AABB b_obj_aabb = obj_mgr.aabbs[b_obj.idx];

        AABB a_world_aabb = a_obj_aabb.applyTRS(a_pos, a_rot, a_scale);
        AABB b_world_aabb = b_obj_aabb.applyTRS(b_pos, b_rot, b_scale);

        if (!a_world_aabb.overlaps(b_world_aabb)) {
            return;
        }
    }

    SolverData &solver = ctx.getSingleton<SolverData>();

    NarrowphaseTest test_type {raw_type_a | raw_type_b};

    switch (test_type) {
    case NarrowphaseTest::SphereSphere: {
        assert(false);
        float a_radius = a_prim->sphere.radius;
        float b_radius = b_prim->sphere.radius;

        Vector3 to_b = b_pos - a_pos;
        float dist = to_b.length();

        if (dist > 0 && dist < a_radius + b_radius) {
            Vector3 mid = to_b / 2.f;

            Vector3 to_b_normal = to_b / dist;

            addContactsToSolver(
                solver, {{
                    a_entity,
                    b_entity,
                    { 
                        makeVector4(a_pos + mid, dist / 2.f),
                        {}, {}, {}
                    },
                    1,
                    to_b_normal,
                    {},
                }});
        }
    } break;
    case NarrowphaseTest::HullHull: {
        // Get half edge mesh for hull A and hull B
        const auto &a_he_mesh = a_prim->hull.halfEdgeMesh;
        const auto &b_he_mesh = b_prim->hull.halfEdgeMesh;

        assert(a_he_mesh.mPolygonCount + b_he_mesh.mPolygonCount < 
               max_num_tmp_faces);

        assert(a_he_mesh.mVertexCount + b_he_mesh.mVertexCount < 
               max_num_tmp_vertices);

        HullState a_hull_state = makeHullState(MADRONA_GPU_COND(mwgpu_lane_id,)
            a_he_mesh, a_pos, a_rot, a_scale, tmp_vertices, tmp_faces);

        tmp_vertices += a_hull_state.numVertices;
        tmp_faces += a_hull_state.numFaces;

        HullState b_hull_state = makeHullState(MADRONA_GPU_COND(mwgpu_lane_id,)
            b_he_mesh, b_pos, b_rot, b_scale, tmp_vertices, tmp_faces);

        MADRONA_GPU_COND(__syncwarp(mwGPU::allActive));

        Manifold manifold = doSAT(MADRONA_GPU_COND(mwgpu_lane_id,)
            a_hull_state, b_hull_state,
            (void *)a_hull_state.facePlanes, (void *)b_hull_state.facePlanes, // FIXME: if these aren't transformed we can't pass as temporaries!!
            {0, 0, 0}, {1, 0, 0, 0});

        if (manifold.numContactPoints > 0) {
            addManifoldToSolver(MADRONA_GPU_COND(mwgpu_lane_id,)
                                solver, manifold, a_entity, b_entity);
        }
    } break;
    case NarrowphaseTest::SphereHull: {
#if 0
        auto a_sphere = a_prim->sphere;
        const auto &b_he_mesh = b_prim->hull.halfEdgeMesh;
        Quat b_rot = ctx.getUnsafe<Rotation>(b_entity);
        Vector3 b_scale = ctx.getUnsafe<Rotation>(b_entity);

        geometry::CollisionMesh b_collision_mesh = 
            buildCollisionMesh(b_he_mesh, b_pos, b_rot, b_scale);
#endif
        assert(false);
    } break;
    case NarrowphaseTest::PlanePlane: {
        // Planes must be static, this should never be called
        assert(false);
    } break;
    case NarrowphaseTest::SpherePlane: {
        assert(false);
        auto sphere = a_prim->sphere;

        constexpr Vector3 base_normal = { 0, 0, 1 };
        Vector3 plane_normal = b_rot.rotateVec(base_normal);

        float d = plane_normal.dot(b_pos);
        float t = plane_normal.dot(a_pos) - d;

        float penetration = sphere.radius - t;
        if (penetration > 0) {
            Vector3 contact_point = a_pos - t * plane_normal;

            addContactsToSolver(
                solver, {{
                    b_entity,
                    a_entity,
                    {
                        makeVector4(contact_point, penetration),
                        {}, {}, {}
                    },
                    1,
                    plane_normal,
                    {},
                }});
        }
    } break;
    case NarrowphaseTest::HullPlane: {
        // Get half edge mesh for entity a (the hull)
        const auto &a_he_mesh = a_prim->hull.halfEdgeMesh;

        assert(a_he_mesh.mPolygonCount < max_num_tmp_faces);
        assert(a_he_mesh.mVertexCount <  max_num_tmp_vertices);

        HullState a_hull_state = makeHullState(MADRONA_GPU_COND(mwgpu_lane_id,)
            a_he_mesh, a_pos, a_rot, a_scale, tmp_vertices, tmp_faces);

        MADRONA_GPU_COND(__syncwarp(mwGPU::allActive));

        constexpr Vector3 base_normal = { 0, 0, 1 };
#if 0
        Quat inv_a_rot = a_rot.inv();
        Vector3 plane_origin_a_local = inv_a_rot.rotateVec(b_pos - a_pos);
        Quat to_a_local_rot = (inv_a_rot * b_rot).normalize();

        Vector3 plane_normal_a_local =
            (to_a_local_rot.rotateVec(base_normal)).normalize();
#endif

        Vector3 plane_normal = b_rot.rotateVec(base_normal);
            
        geometry::Plane plane {
            plane_normal,
            dot(plane_normal, b_pos),
        };

        Manifold manifold = doSATPlane(MADRONA_GPU_COND(mwgpu_lane_id,)
            plane, a_hull_state, tmp_faces, tmp_faces + a_hull_state.numFaces,
            {0, 0, 0}, {1, 0, 0, 0});

        if (manifold.numContactPoints > 0) {
            addManifoldToSolver(MADRONA_GPU_COND(mwgpu_lane_id,)
                solver, manifold, a_entity, b_entity);
        }
    } break;
    default: __builtin_unreachable();
    }
}

TaskGraph::NodeID setupTasks(
    TaskGraph::Builder &builder,
    Span<const TaskGraph::NodeID> deps)
{
#ifdef MADRONA_GPU_MODE
    auto narrowphase = builder.addToGraph<CustomParallelForNode<Context,
        runNarrowphase, 32, CandidateCollision>>(deps);
#else
    auto narrowphase = builder.addToGraph<ParallelForNode<Context,
        runNarrowphase, CandidateCollision>>(deps);
#endif

    // FIXME do some kind of scoped reset on tmp alloc
    return builder.addToGraph<ResetTmpAllocNode>({narrowphase});
}

}
