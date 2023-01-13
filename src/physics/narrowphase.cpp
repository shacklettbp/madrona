#include <madrona/memory.hpp>
#include <madrona/physics.hpp>
#include <madrona/context.hpp>

#include "physics_impl.hpp"

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

struct Manifold {
    math::Vector4 contactPoints[4];
    int32_t numContactPoints;
    math::Vector3 normal;
    bool aIsReference;
};

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


static uint32_t findFurthestPointIdx(const CollisionMesh &m,
                                     const math::Vector3 &d)
{
    float maxDistance = d.dot(m.vertices[0]);
    uint32_t vertexIdx = 0;

    for (CountT i = 1; i < (CountT)m.vertexCount; ++i) {
        float dp = d.dot(m.vertices[i]);
        if (dp > maxDistance) {
            maxDistance = dp;
            vertexIdx = i;
        }
    }

    return vertexIdx;
}

static math::Vector3 findFurthestPoint(const CollisionMesh &m,
                                       const math::Vector3 &d)
{
    return m.vertices[findFurthestPointIdx(m, d)];
}

static FaceQuery queryFaceDirectionsPlane(const Plane &plane,
                                          const CollisionMesh &a) {
    math::Vector3 supportA = findFurthestPoint(a, -plane.normal);
    float distance = getDistanceFromPlane(plane, supportA);

    return { distance, 0 };
}

static FaceQuery queryFaceDirections(const CollisionMesh &a,
                                     const CollisionMesh &b)
{
    const auto *hMesh = a.halfEdgeMesh;

    int polygonMaxDistance = 0;
    float maxDistance = -FLT_MAX;

    for (CountT i = 0; i < (CountT)hMesh->getPolygonCount(); ++i) {
        Plane plane = hMesh->getPlane(i, a.vertices);
        math::Vector3 supportB = findFurthestPoint(b, -plane.normal);
        float distance = getDistanceFromPlane(plane, supportB);

        if (distance > maxDistance) {
            maxDistance = distance;
            polygonMaxDistance = i;
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

bool buildsMinkowskiFace(
        const CollisionMesh &a, const CollisionMesh &b,
        const HalfEdge &edgeA, const HalfEdge &edgeB)
{
    auto [aNormal1, aNormal2] = a.halfEdgeMesh->getEdgeNormals(edgeA, a.vertices);
    auto [bNormal1, bNormal2] = b.halfEdgeMesh->getEdgeNormals(edgeB, b.vertices);

    return isMinkowskiFace(aNormal1, aNormal2, -bNormal1, -bNormal2);
}

static float edgeDistance(
        const CollisionMesh &a, const CollisionMesh &b,
        const HalfEdge &edgeA, const HalfEdge &edgeB)
{
    math::Vector3 dirA = a.halfEdgeMesh->getEdgeDirection(edgeA, a.vertices);
    math::Vector3 pointA = a.halfEdgeMesh->getEdgeOrigin(edgeA, a.vertices);

    math::Vector3 dirB = b.halfEdgeMesh->getEdgeDirection(edgeB, b.vertices);
    math::Vector3 pointB = b.halfEdgeMesh->getEdgeOrigin(edgeB, b.vertices);

    if (areParallel(dirA, dirB)) {
        return -FLT_MAX;
    }

    math::Vector3 normal = dirA.cross(dirB).normalize();

    if (normal.dot(pointA - a.center) < 0.0f) {
        normal = -normal;
    }

    return normal.dot(pointB - pointA);
}

static EdgeQuery queryEdgeDirections(const CollisionMesh &a, const CollisionMesh &b)
{
    const auto *hMeshA = a.halfEdgeMesh;
    const auto *hMeshB = b.halfEdgeMesh;

    Vector3 normal;
    int edgeAMaxDistance = 0;
    int edgeBMaxDistance = 0;
    float maxDistance = -FLT_MAX;

    for (CountT edgeIdxA = 0; edgeIdxA < (CountT)hMeshA->getEdgeCount(); ++edgeIdxA) {
        auto edgeDataA = hMeshA->edge(edgeIdxA);
        auto hEdgeA = hMeshA->halfEdge(edgeDataA);

        for (CountT edgeIdxB = 0; edgeIdxB < (CountT)hMeshB->getEdgeCount(); ++edgeIdxB) {
            auto edgeDataB = hMeshB->edge(edgeIdxB);
            auto hEdgeB = hMeshB->halfEdge(edgeDataB);

            math::Vector3 edgeDirectionA = hMeshA->getEdgeDirection(edgeDataA, a.vertices);
            math::Vector3 edgeDirectionB = hMeshB->getEdgeDirection(edgeDataB, b.vertices);

            math::Vector3 axis = edgeDirectionA.cross(edgeDirectionB).normalize();

            if (buildsMinkowskiFace(a, b, hEdgeA, hEdgeB)) {
                // FIXME: this edgeDistance function seems to duplicate a lot of the
                // above work (computing axis, specifically)
                float separation = edgeDistance(a, b, hEdgeA, hEdgeB);

                if (separation > maxDistance) {
                    maxDistance = separation;
                    normal = axis;
                    edgeAMaxDistance = edgeIdxA;
                    edgeBMaxDistance = edgeIdxB;
                }
            }
        }
    }

    return { maxDistance, normal, edgeAMaxDistance, edgeBMaxDistance };
}

static void clipPolygon(
        Context &ctx,
        const Plane &clippingPlane,
        uint32_t *vertexCount,
        math::Vector3 *vertices,
        const uint32_t kMaxVertices)
{
    uint32_t newVertexCount = 0;
    math::Vector3 *newVertices = (math::Vector3 *)ctx.tmpAlloc(sizeof(math::Vector3) * (*vertexCount) * 2);

    math::Vector3 v1 = vertices[*vertexCount - 1];
    float d1 = getDistanceFromPlane(clippingPlane, v1);

    for (CountT i = 0; i < (CountT)*vertexCount; ++i) {
        math::Vector3 v2 = vertices[i];
        float d2 = getDistanceFromPlane(clippingPlane, v2);

        if (d1 <= 0.0f && d2 <= 0.0f) {
            // Both vertices are behind the plane, keep the second vertex
            newVertices[newVertexCount++] = v2;
        }
        else if (d1 <= 0.0f && d2 > 0.0f) {
            // v1 is behind the plane, the other is in front (out)
            math::Vector3 intersection = planeIntersection(clippingPlane, v1, v2);
            newVertices[newVertexCount++] = intersection;
        }
        else if (d2 <= 0.0f && d1 > 0.0f) {
            math::Vector3 intersection = planeIntersection(clippingPlane, v1, v2);
            newVertices[newVertexCount++] = intersection;
            newVertices[newVertexCount++] = v2;
        }

        // assert(newVertexCount <= kMaxVertices);

        // Now use v2 as the starting vertex
        v1 = v2;
        d1 = d2;
    }

    *vertexCount = newVertexCount;
    memcpy(vertices, newVertices, sizeof(math::Vector3) * newVertexCount);
}

static int findIncidentFace(const CollisionMesh &referenceHull,
                            const CollisionMesh &otherHull,
                            int referenceFaceIdx)
{
    math::Vector3 referenceNormal = referenceHull.halfEdgeMesh->getFaceNormal(referenceFaceIdx, referenceHull.vertices);

    float minimizingDotProduct = FLT_MAX;
    int minimizingFace = -1;
    for (CountT i = 0; i < (CountT)otherHull.halfEdgeMesh->getPolygonCount(); ++i) {
        math::Vector3 incidentNormal = otherHull.halfEdgeMesh->getFaceNormal(i, otherHull.vertices);
        float dotProduct = incidentNormal.dot(referenceNormal);

        if (dotProduct < minimizingDotProduct) {
            minimizingDotProduct = dotProduct;
            minimizingFace = i;
        }
    }

    return minimizingFace;
}

static int findIncidentFace(const Plane &referencePlane,
                            const CollisionMesh &otherHull)
{
    float minimizingDotProduct = FLT_MAX;
    int minimizingFace = -1;
    for (CountT i = 0; i < (CountT)otherHull.halfEdgeMesh->getPolygonCount(); ++i) {
        math::Vector3 incidentNormal = otherHull.halfEdgeMesh->getFaceNormal(i, otherHull.vertices);
        float dotProduct = incidentNormal.dot(referencePlane.normal);

        if (dotProduct < minimizingDotProduct) {
            minimizingDotProduct = dotProduct;
            minimizingFace = i;
        }
    }

    return minimizingFace;
}

static Manifold createFaceContactPlane(Context &ctx,
                                       const Plane &plane,
                                       const CollisionMesh &hull)
{
    // Find incident face
    int incidentFaceIdx = findIncidentFace(plane, hull);

    uint32_t incidentFaceVertexCount = hull.halfEdgeMesh->getPolygonVertexCount(incidentFaceIdx);
    const uint32_t kMaxIncidentVertexCount = incidentFaceVertexCount * 2;
    math::Vector3 *incidentVertices = (math::Vector3 *)ctx.tmpAlloc(sizeof(math::Vector3) * kMaxIncidentVertexCount);
    hull.halfEdgeMesh->getPolygonVertices(incidentVertices, incidentFaceIdx, hull.vertices);

    math::Vector4 *contacts = (math::Vector4 *)ctx.tmpAlloc(sizeof(math::Vector4) * kMaxIncidentVertexCount);
    CountT contact_count = 0;

    for (CountT i = 0; i < (CountT)incidentFaceVertexCount; ++i) {
        if (float d = getDistanceFromPlane(plane, incidentVertices[i]); d < 0.0f) {
            // Project the point onto the reference plane (d guaranteed to be negative)
            contacts[contact_count++] = makeVector4(incidentVertices[i] - d * plane.normal, -d);
        }
    }

    Manifold manifold;
    manifold.normal = plane.normal;
    manifold.aIsReference = false;
    if (contact_count <= 4) {
        manifold.numContactPoints = contact_count;
        for (CountT i = 0; i < contact_count; i++) {
            manifold.contactPoints[i] = contacts[i];
        }
    } else {
        manifold.numContactPoints = 4;
        manifold.contactPoints[0] = contacts[0];

        // Find furthest contact
        float largestD2 = 0.0f;
        int largestD2ContactPointIdx = 0;
        for (CountT i = 1; i < contact_count; ++i) {
            Vector3 cur_contact = contacts[i].xyz();
            float d2 = manifold.contactPoints[0].xyz().distance2(cur_contact);
            if (d2 > largestD2) {
                largestD2 = d2;
                manifold.contactPoints[1] = makeVector4(cur_contact, contacts[i].w);
                largestD2ContactPointIdx = i;
            }
        }

        contacts[largestD2ContactPointIdx] = manifold.contactPoints[0];

        math::Vector3 diff0 =
            manifold.contactPoints[1].xyz() - manifold.contactPoints[0].xyz();

        // Find point which maximized area of triangle
        float largestArea = 0.0f;
        int largestAreaContactPointIdx = 0;
        for (CountT i = 1; i < contact_count; ++i) {
            Vector3 cur_contact = contacts[i].xyz();
            math::Vector3 diff1 = cur_contact - manifold.contactPoints[0].xyz();
            float area = plane.normal.dot(diff0.cross(diff1));
            if (area > largestArea) {
                manifold.contactPoints[2] = makeVector4(cur_contact, contacts[i].w);
                largestAreaContactPointIdx = i;
            }
        }

        contacts[largestAreaContactPointIdx] = manifold.contactPoints[0];

        for (CountT i = 1; i < contact_count; ++i) {
            Vector3 cur_contact = contacts[i].xyz();
            math::Vector3 diff1 = cur_contact - manifold.contactPoints[0].xyz();
            float area = plane.normal.dot(diff0.cross(diff1));
            if (area < largestArea) {
                manifold.contactPoints[3] = makeVector4(cur_contact, contacts[i].w);
            }
        }
    }

    return manifold;
}

static Manifold createFaceContact(Context &ctx,
                                  FaceQuery faceQueryA, const CollisionMesh &a,
                                  FaceQuery faceQueryB, const CollisionMesh &b)
{
    // Determine minimizing face
    bool a_is_ref = faceQueryA.separation > faceQueryB.separation;
    FaceQuery &minimizingQuery = a_is_ref ? faceQueryA : faceQueryB;

    const CollisionMesh &referenceHull = a_is_ref ? a : b;
    const CollisionMesh &otherHull = a_is_ref ? b : a;

    int referenceFaceIdx = minimizingQuery.faceIdx;

    // Find incident face
    int incidentFaceIdx = findIncidentFace(referenceHull, otherHull, referenceFaceIdx);

    // Get the side planes of the reference face
    uint32_t sidePlaneCount = referenceHull.halfEdgeMesh->getPolygonVertexCount(referenceFaceIdx);
    Plane *planes = (Plane *)ctx.tmpAlloc(sizeof(Plane) * sidePlaneCount);
    referenceHull.halfEdgeMesh->getPolygonSidePlanes(planes, referenceFaceIdx, referenceHull.vertices);

    uint32_t incidentFaceVertexCount = otherHull.halfEdgeMesh->getPolygonVertexCount(incidentFaceIdx);
    const uint32_t kMaxIncidentVertexCount = incidentFaceVertexCount * 2;
    math::Vector3 *incidentVertices = (math::Vector3 *)ctx.tmpAlloc(sizeof(math::Vector3) * kMaxIncidentVertexCount);
    otherHull.halfEdgeMesh->getPolygonVertices(incidentVertices, incidentFaceIdx, otherHull.vertices);

    // Clip the incident face against the side planes of the reference face
    for (CountT i = 0; i < (CountT)sidePlaneCount; ++i) {
        clipPolygon(ctx, planes[i], &incidentFaceVertexCount, incidentVertices, kMaxIncidentVertexCount);
    }

    // incidentVertices should now contain the only relevant vertices
    // Now we just keep the ones below the reference face
    Plane referencePlane = referenceHull.halfEdgeMesh->getPlane(referenceFaceIdx, referenceHull.vertices);

    math::Vector4 *contacts = (math::Vector4 *)ctx.tmpAlloc(sizeof(math::Vector4) * kMaxIncidentVertexCount);
    CountT contact_count = 0;

    for (CountT i = 0; i < (CountT)incidentFaceVertexCount; ++i) {
        if (float d = getDistanceFromPlane(referencePlane, incidentVertices[i]); d < 0.0f) {
            // Project the point onto the reference plane (d guaranteed to be negative)
            contacts[contact_count++] = makeVector4(incidentVertices[i] - d * referencePlane.normal, -d);
        }
    }

    Manifold manifold;
    manifold.normal = referencePlane.normal;
    manifold.aIsReference = a_is_ref;
    if (contact_count <= 4) {
        manifold.numContactPoints = contact_count;
        for (CountT i = 0; i < contact_count; i++) {
            manifold.contactPoints[i] = contacts[i];
        }
    } else {
        manifold.numContactPoints = 4;
        manifold.contactPoints[0] = contacts[0];
        Vector3 point0 = manifold.contactPoints[0].xyz();

        // Find furthest contact
        float largestD2 = 0.0f;
        int largestD2ContactPointIdx = 0;
        for (CountT i = 1; i < contact_count; ++i) {
            Vector4 cur_contact = contacts[i];
            float d2 = point0.distance2(cur_contact.xyz());
            if (d2 > largestD2) {
                largestD2 = d2;
                manifold.contactPoints[1] = cur_contact;
                largestD2ContactPointIdx = i;
            }
        }

        contacts[largestD2ContactPointIdx] = manifold.contactPoints[0];

        math::Vector3 diff0 = manifold.contactPoints[1].xyz() - point0;

        // Find point which maximized area of triangle
        float largestArea = 0.0f;
        int largestAreaContactPointIdx = 0;
        for (CountT i = 1; i < contact_count; ++i) {
            Vector4 cur_contact = contacts[i];
            math::Vector3 diff1 = cur_contact.xyz() - point0;
            float area = referencePlane.normal.dot(diff0.cross(diff1));
            if (area > largestArea) {
                manifold.contactPoints[2] = cur_contact;
                largestAreaContactPointIdx = i;
            }
        }

        contacts[largestAreaContactPointIdx] = manifold.contactPoints[0];

        for (CountT i = 1; i < contact_count; ++i) {
            Vector4 cur_contact = contacts[i];
            math::Vector3 diff1 = cur_contact.xyz() - point0;
            float area = referencePlane.normal.dot(diff0.cross(diff1));
            if (area < largestArea) {
                manifold.contactPoints[3] = cur_contact;
            }
        }
    }

    return manifold;
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
                                  const CollisionMesh &a,
                                  const CollisionMesh &b)
{
    Segment segA = a.halfEdgeMesh->getEdgeSegment(a.halfEdgeMesh->edge(query.edgeIdxA), a.vertices);
    Segment segB = b.halfEdgeMesh->getEdgeSegment(b.halfEdgeMesh->edge(query.edgeIdxB), b.vertices);

    Segment s = shortestSegmentBetween(segA, segB);
    Vector3 contact = 0.5f * (s.p1 + s.p2);
    float depth = (s.p2 - s.p1).length() / 2.0f;

    Manifold manifold;
    manifold.contactPoints[0] = makeVector4(contact, depth);
    manifold.numContactPoints = 1;
    manifold.normal = query.normal;
    manifold.aIsReference = true; // Is this guaranteed?
    
    if (manifold.normal.dot(contact - a.center) < 0.0f) {
        manifold.aIsReference = false;
    }

    return manifold;
}

Manifold doSAT(Context &ctx, const CollisionMesh &a, const CollisionMesh &b)
{
    Manifold manifold;
    manifold.numContactPoints = 0;

    FaceQuery faceQueryA = queryFaceDirections(a, b);
    if (faceQueryA.separation > 0.0f) {
        // There is a separating axis - no collision
        return manifold;
    }

    FaceQuery faceQueryB = queryFaceDirections(b, a);
    if (faceQueryB.separation > 0.0f) {
        // There is a separating axis - no collision
        return manifold;
    }

    EdgeQuery edgeQuery = queryEdgeDirections(a, b);
    if (edgeQuery.separation > 0.0f) {
        // There is a separating axis - no collision
        return manifold;
    }

    bool bIsFaceContactA = faceQueryA.separation > edgeQuery.separation;
    bool bIsFaceContactB = faceQueryB.separation > edgeQuery.separation;

    if (bIsFaceContactA || bIsFaceContactB) {
        // Create face contact
        manifold = createFaceContact(ctx, faceQueryA, a, faceQueryB, b);
    }
    else {
        // Create edge contact
        manifold = createEdgeContact(edgeQuery, a, b);
    }

    return manifold;
}

Manifold doSATPlane(Context &ctx, const Plane &plane, const CollisionMesh &a)
{
    Manifold manifold;
    manifold.numContactPoints = 0;

    FaceQuery faceQuery = queryFaceDirectionsPlane(plane, a);

    if (faceQuery.separation > 0.0f) {
        return manifold;
    }

    return createFaceContactPlane(ctx, plane, a);
}

// FIXME: Reduce redundant work on transforming point
static inline geometry::CollisionMesh buildCollisionMesh(
    Context &ctx,
    const geometry::HalfEdgeMesh &he_mesh,
    Vector3 pos, Quat rot, Vector3 scale)
{
    auto transformVertex = [pos, rot, scale] (math::Vector3 v) {
        return pos + rot.rotateVec((math::Vector3)scale * v);
    };

    geometry::CollisionMesh collision_mesh;
    collision_mesh.halfEdgeMesh = &he_mesh;
    collision_mesh.vertexCount = he_mesh.getVertexCount();
    collision_mesh.vertices = (Vector3 *)ctx.tmpAlloc(
        sizeof(math::Vector3) * collision_mesh.vertexCount);
    collision_mesh.center = pos;

    for (CountT v = 0; v < (CountT)collision_mesh.vertexCount; ++v) {
        collision_mesh.vertices[v] = transformVertex(he_mesh.vertex(v));
    }

    return collision_mesh;
}

static inline void addContactsToSolver(SolverData &solver_data,
                                       Span<const Contact> added_contacts)
{
    int32_t contact_idx = solver_data.numContacts.fetch_add(
        added_contacts.size(), std::memory_order_relaxed);

    assert(contact_idx < solver_data.maxContacts);
    
    for (CountT i = 0; i < added_contacts.size(); i++) {
        solver_data.contacts[contact_idx + i] = added_contacts[i];
    }
}

inline void runNarrowphase(
    Context &ctx,
    const CandidateCollision &candidate_collision)
{
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
    Vector3 a_scale = ctx.getUnsafe<Scale>(a_entity);
    Vector3 b_scale = ctx.getUnsafe<Scale>(b_entity);

    {
        // FIXME: Rechecking the AABBs here seems to only give a very small
        // performance improvement. Should revisit.
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
        float a_radius = a_prim->sphere.radius;
        float b_radius = b_prim->sphere.radius;

        Vector3 to_b = b_pos - a_pos;
        float dist = to_b.length();

        if (dist > 0 && dist < a_radius + b_radius) {
            Vector3 mid = to_b / 2.f;

            Vector3 to_b_normal = to_b / dist;
            addContactsToSolver(solver, {{
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

            Loc loc = ctx.makeTemporary<CollisionEventTemporary>();
            ctx.getUnsafe<CollisionEvent>(loc) = CollisionEvent {
                candidate_collision.a,
                candidate_collision.b,
            };
        }
    } break;
    case NarrowphaseTest::HullHull: {
        // Get half edge mesh for hull A and hull B
        const auto &a_he_mesh = a_prim->hull.halfEdgeMesh;
        const auto &b_he_mesh = b_prim->hull.halfEdgeMesh;

        geometry::CollisionMesh a_collision_mesh =
            buildCollisionMesh(ctx, a_he_mesh, a_pos, a_rot, a_scale);

        geometry::CollisionMesh b_collision_mesh =
            buildCollisionMesh(ctx, b_he_mesh, b_pos, b_rot, b_scale);

        Manifold manifold = doSAT(ctx, a_collision_mesh, b_collision_mesh);

        if (manifold.numContactPoints > 0) {
            addContactsToSolver(solver, {{
                manifold.aIsReference ? a_entity : b_entity,
                manifold.aIsReference ? b_entity : a_entity,
                {
                    manifold.contactPoints[0],
                    manifold.contactPoints[1],
                    manifold.contactPoints[2],
                    manifold.contactPoints[3],
                },
                manifold.numContactPoints,
                manifold.normal,
                {},
            }});
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
        auto sphere = a_prim->sphere;

        constexpr Vector3 base_normal = { 0, 0, 1 };
        Vector3 plane_normal = b_rot.rotateVec(base_normal);

        float d = plane_normal.dot(b_pos);
        float t = plane_normal.dot(a_pos) - d;

        float penetration = sphere.radius - t;
        if (penetration > 0) {
            Vector3 contact_point = a_pos - t * plane_normal;

            addContactsToSolver(solver, {{
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
        
        geometry::CollisionMesh a_collision_mesh =
            buildCollisionMesh(ctx, a_he_mesh, a_pos, a_rot, a_scale);

        constexpr Vector3 base_normal = { 0, 0, 1 };
        Vector3 plane_normal = b_rot.rotateVec(base_normal);

        geometry::Plane plane { plane_normal, dot(b_pos, plane_normal) };

        Manifold manifold = doSATPlane(ctx, plane, a_collision_mesh);

        if (manifold.numContactPoints > 0) {
            addContactsToSolver(solver, {{
                b_entity, // Plane is always reference
                a_entity,
                {
                    manifold.contactPoints[0],
                    manifold.contactPoints[1],
                    manifold.contactPoints[2],
                    manifold.contactPoints[3],
                },
                manifold.numContactPoints,
                manifold.normal,
                {},
            }});
        }
    } break;
    default: __builtin_unreachable();
    }
}

TaskGraph::NodeID setupTasks(
    TaskGraph::Builder &builder,
    Span<const TaskGraph::NodeID> deps)
{
    auto narrowphase = builder.addToGraph<ParallelForNode<Context,
        runNarrowphase, CandidateCollision>>(deps);

    // FIXME do some kind of scoped reset on tmp alloc
    return builder.addToGraph<ResetTmpAllocNode>({narrowphase});
}

}
