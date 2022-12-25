#include <stdint.h>
#include <madrona/memory.hpp>
#include <madrona/physics.hpp>
#include <madrona/context.hpp>

namespace madrona {
using namespace base;
using namespace math;

namespace phys::narrowphase {

using namespace geometry;

uint32_t findFurthestPointIdx(const CollisionMesh &m, const math::Vector3 &d) {
    float maxDistance = d.dot(m.vertices[0]);
    uint32_t vertexIdx = 0;

    for (int i = 1; i < m.vertexCount; ++i) {
        float dp = d.dot(m.vertices[i]);
        if (dp > maxDistance) {
            maxDistance = dp;
            vertexIdx = i;
        }
    }

    return vertexIdx;
}

math::Vector3 findFurthestPoint(const CollisionMesh &m, const math::Vector3 &d) {
    return m.vertices[findFurthestPointIdx(m, d)];
}

FaceQuery queryFaceDirections(const CollisionMesh &a, const CollisionMesh &b) {
    const auto *hMesh = a.halfEdgeMesh;

    int polygonMaxDistance = 0;
    float maxDistance = -FLT_MAX;

    for (int i = 0; i < hMesh->getPolygonCount(); ++i) {
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

bool isMinkowskiFace(
        const math::Vector3 &a, const math::Vector3 &b,
        const math::Vector3 &c, const math::Vector3 &d) {
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
        const HalfEdge &edgeA, const HalfEdge &edgeB) {
    auto [aNormal1, aNormal2] = a.halfEdgeMesh->getEdgeNormals(edgeA, a.vertices);
    auto [bNormal1, bNormal2] = b.halfEdgeMesh->getEdgeNormals(edgeB, b.vertices);

    return isMinkowskiFace(aNormal1, aNormal2, -bNormal1, -bNormal2);
}

float edgeDistance(
        const CollisionMesh &a, const CollisionMesh &b,
        const HalfEdge &edgeA, const HalfEdge &edgeB) {
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

EdgeQuery queryEdgeDirections(const CollisionMesh &a, const CollisionMesh &b) {
    const auto *hMeshA = a.halfEdgeMesh;
    const auto *hMeshB = b.halfEdgeMesh;

    int edgeAMaxDistance = 0;
    int edgeBMaxDistance = 0;
    float maxDistance = -FLT_MAX;

    for (int edgeIdxA = 0; edgeIdxA < hMeshA->getEdgeCount(); ++edgeIdxA) {
        auto edgeDataA = hMeshA->edge(edgeIdxA);
        auto hEdgeA = hMeshA->halfEdge(edgeDataA);

        for (int edgeIdxB = 0; edgeIdxB < hMeshB->getEdgeCount(); ++edgeIdxB) {
            auto edgeDataB = hMeshB->edge(edgeIdxB);
            auto hEdgeB = hMeshB->halfEdge(edgeDataB);

            math::Vector3 edgeDirectionA = hMeshA->getEdgeDirection(edgeDataA, a.vertices);
            math::Vector3 edgeDirectionB = hMeshB->getEdgeDirection(edgeDataB, b.vertices);

            math::Vector3 axis = edgeDirectionA.cross(edgeDirectionB).normalize();

            if (buildsMinkowskiFace(a, b, hEdgeA, hEdgeB)) {
                float separation = edgeDistance(a, b, hEdgeA, hEdgeB);

                if (separation > maxDistance) {
                    maxDistance = separation;
                    edgeAMaxDistance = edgeIdxA;
                    edgeBMaxDistance = edgeIdxB;
                }
            }
        }
    }

    return { maxDistance, edgeAMaxDistance, edgeBMaxDistance };
}

void clipPolygon(
        const Plane &clippingPlane,
        uint32_t *vertexCount,
        math::Vector3 *vertices,
        const uint32_t kMaxVertices) {
    uint32_t newVertexCount = 0;
    math::Vector3 *newVertices = (math::Vector3 *)TmpAllocator::get().alloc(sizeof(math::Vector3) * (*vertexCount) * 2);

    math::Vector3 v1 = vertices[*vertexCount - 1];
    float d1 = getDistanceFromPlane(clippingPlane, v1);

    for (int i = 0; i < *vertexCount; ++i) {
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

int findIncidentFace(const CollisionMesh &referenceHull, const CollisionMesh &otherHull, int referenceFaceIdx) {
    math::Vector3 referenceNormal = referenceHull.halfEdgeMesh->getFaceNormal(referenceFaceIdx, referenceHull.vertices);

    float minimizingDotProduct = FLT_MAX;
    int minimizingFace = -1;
    for (int i = 0; i < otherHull.halfEdgeMesh->getPolygonCount(); ++i) {
        math::Vector3 incidentNormal = otherHull.halfEdgeMesh->getFaceNormal(i, otherHull.vertices);
        float dotProduct = incidentNormal.dot(referenceNormal);

        if (dotProduct < minimizingDotProduct) {
            minimizingDotProduct = dotProduct;
            minimizingFace = i;
        }
    }

    return minimizingFace;
}

Manifold createFaceContact(FaceQuery faceQueryA, const CollisionMesh &a, FaceQuery faceQueryB, const CollisionMesh &b) {
    // Determine minimizing face
    FaceQuery &minimizingQuery = (faceQueryA.separation < faceQueryB.separation) ? faceQueryA : faceQueryB;
    FaceQuery &otherQuery = (faceQueryA.separation < faceQueryB.separation) ? faceQueryB : faceQueryA;

    const CollisionMesh &referenceHull = (faceQueryA.separation < faceQueryB.separation) ? a : b;
    const CollisionMesh &otherHull = (faceQueryA.separation < faceQueryB.separation) ? b : a;

    int referenceFaceIdx = minimizingQuery.faceIdx;

    // Find incident face
    int incidentFaceIdx = findIncidentFace(referenceHull, otherHull, referenceFaceIdx);

    // Get the side planes of the reference face
    uint32_t sidePlaneCount = referenceHull.halfEdgeMesh->getPolygonVertexCount(referenceFaceIdx);
    Plane *planes = (Plane *)TmpAllocator::get().alloc(sizeof(Plane) * sidePlaneCount);
    referenceHull.halfEdgeMesh->getPolygonSidePlanes(planes, referenceFaceIdx, referenceHull.vertices);

    uint32_t incidentFaceVertexCount = otherHull.halfEdgeMesh->getPolygonVertexCount(incidentFaceIdx);
    const uint32_t kMaxIncidentVertexCount = incidentFaceVertexCount * 2;
    math::Vector3 *incidentVertices = (math::Vector3 *)TmpAllocator::get().alloc(sizeof(math::Vector3) * kMaxIncidentVertexCount);
    otherHull.halfEdgeMesh->getPolygonVertices(incidentVertices, incidentFaceIdx, otherHull.vertices);

    // Clip the incident face against the side planes of the reference face
    for (int i = 0; i < sidePlaneCount; ++i) {
        clipPolygon(planes[i], &incidentFaceVertexCount, incidentVertices, kMaxIncidentVertexCount);
    }

    // incidentVertices should now contain the only relevant vertices
    // Now we just keep the ones below the reference face
    Plane referencePlane = referenceHull.halfEdgeMesh->getPlane(referenceFaceIdx, referenceHull.vertices);

    math::Vector3 *contacts = (math::Vector3 *)TmpAllocator::get().alloc(sizeof(math::Vector3) * kMaxIncidentVertexCount);
    uint32_t contactCount = 0;

    for (int i = 0; i < incidentFaceVertexCount; ++i) {
        if (float d = getDistanceFromPlane(referencePlane, incidentVertices[i]); d < 0.0f) {
            // Project the point onto the reference plane
            contacts[contactCount++] = incidentVertices[i] - d * referencePlane.normal;
        }
    }

    if (contactCount > 4) {
        math::Vector3 *reducedContacts = (math::Vector3 *)TmpAllocator::get().alloc(sizeof(math::Vector3) * 4);
        reducedContacts[0] = contacts[0];
        uint32_t reducedCount = 1;

        // Find furthest contact
        float largestD2 = 0.0f;
        int largestD2ContactPointIdx = 0;
        for (int i = 1; i < contactCount; ++i) {
            math::Vector3 diff = reducedContacts[0] - contacts[i];
            float d2 = diff.dot(diff);
            if (d2 > largestD2) {
                largestD2 = d2;
                reducedContacts[1] = contacts[i];
                largestD2ContactPointIdx = i;
            }
        }

        contacts[largestD2ContactPointIdx] = contacts[0];

        math::Vector3 diff0 = reducedContacts[1] - reducedContacts[0];

        // Find point which maximized area of triangle
        float largestArea = 0.0f;
        int largestAreaContactPointIdx = 0;
        for (int i = 1; i < contactCount; ++i) {
            math::Vector3 diff1 = contacts[i] - reducedContacts[0];
            float area = referencePlane.normal.dot(diff0.cross(diff1));
            if (area > largestArea) {
                reducedContacts[2] = contacts[i];
                largestAreaContactPointIdx = i;
            }
        }

        contacts[largestAreaContactPointIdx] = contacts[0];

        float mostNegative = 0.0f;
        int mostNegativeAreaContactPointIdx = 0;
        for (int i = 1; i < contactCount; ++i) {
            math::Vector3 diff1 = contacts[i] - reducedContacts[0];
            float area = referencePlane.normal.dot(diff0.cross(diff1));
            if (area < largestArea) {
                reducedContacts[3] = contacts[i];
                mostNegativeAreaContactPointIdx = i;
            }
        }

        for (int i = 0; i < 4; ++i) {
            contacts[i] = reducedContacts[i];
        }
    }

    return {contacts, contactCount};
}

Segment shortestSegmentBetween(const Segment &seg1, const Segment &seg2) {
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

    if (abs(denom) < 0.00001f) {
        s = 0.0f;
        t = (dotv11 * s - dotv211) / dotv21;
    }
    else {
        s = (dotv212 * dotv21 - dotv22 * dotv211) / denom;
        t = (-dotv211 * dotv21 + dotv11 * dotv212) / denom;
    }

    s = max(min(s, 1.0f), 0.0f);
    t = max(min(t, 1.0f), 0.0f);

    return { seg1.p1 + s * v1, seg2.p1 + t * v2 };
}


Manifold createEdgeContact(const EdgeQuery &query, const CollisionMesh &a, const CollisionMesh &b) {
    Segment segA = a.halfEdgeMesh->getEdgeSegment(a.halfEdgeMesh->edge(query.edgeIdxA), a.vertices);
    Segment segB = b.halfEdgeMesh->getEdgeSegment(b.halfEdgeMesh->edge(query.edgeIdxB), b.vertices);

    math::Vector3 *contact = (math::Vector3 *)TmpAllocator::get().alloc(sizeof(math::Vector3));
    Segment s = shortestSegmentBetween(segA, segB);
    *contact = 0.5f * (s.p1 + s.p2);

    return {contact, 1};
}

Manifold doSAT(const CollisionMesh &a, const CollisionMesh &b) {
    Manifold manifold = { nullptr, 0 };

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
        manifold = createFaceContact(faceQueryA, a, faceQueryB, b);
    }
    else {
        // Create edge contact
        manifold = createEdgeContact(edgeQuery, a, b);
    }

    return manifold;
}

}
}

