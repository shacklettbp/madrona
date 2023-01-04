#include <madrona/physics.hpp>
#include <madrona/context.hpp>

namespace madrona {

using namespace base;
using namespace math;

namespace phys {

struct SolverData {
    Contact *contacts;
    std::atomic<CountT> numContacts;
    CountT maxContacts;
    float deltaT;
    float h;
    Vector3 g;
    float gMagnitude;
    float restitutionThreshold;

    inline SolverData(CountT max_contacts_per_step,
                      float delta_t,
                      CountT num_substeps,
                      Vector3 gravity)
        : contacts((Contact *)rawAlloc(sizeof(Contact) * max_contacts_per_step)),
          numContacts(0),
          maxContacts(max_contacts_per_step),
          deltaT(delta_t),
          h(delta_t / (float)num_substeps),
          g(gravity),
          gMagnitude(gravity.length()),
          restitutionThreshold(2.f * gMagnitude * h)
    {}

    inline void addContacts(Span<const Contact> added_contacts)
    {
        int32_t contact_idx = numContacts.fetch_add(added_contacts.size(),
                                                    std::memory_order_relaxed);
        assert(contact_idx < maxContacts);

        for (CountT i = 0; i < added_contacts.size(); i++) {
            contacts[contact_idx + i] = added_contacts[i];
        }
    }
};

struct ObjectData {
    ObjectManager *mgr;
};

inline void updateCollisionAABB(Context &ctx,
                                const Position &pos,
                                const Rotation &rot,
                                const ObjectID &obj_id,
                                const Velocity &vel,
                                CollisionAABB &out_aabb)
{
    // FIXME: this could all be more efficient with a center + width
    // AABB representation
    ObjectManager &obj_mgr = *ctx.getSingleton<ObjectData>().mgr;

    Mat3x3 rot_mat = Mat3x3::fromQuat(rot);
    AABB obj_aabb = obj_mgr.aabbs[obj_id.idx];

    AABB world_aabb;

    // RTCD page 86
#pragma unroll
    for (CountT i = 0; i < 3; i++) {
        world_aabb.pMin[i] = world_aabb.pMax[i] = pos[i];

#pragma unroll
        for (CountT j = 0; j < 3; j++) {
            float e = rot_mat[i][j] * obj_aabb.pMin[j];
            float f = rot_mat[i][j] * obj_aabb.pMax[j];

            if (e < f) {
                world_aabb.pMin[i] += e;
                world_aabb.pMax[i] += f;
            } else {
                world_aabb.pMin[i] += f;
                world_aabb.pMax[i] += e;
            }
        }
    }

    constexpr float expansion_factor = 2.f;
    constexpr float max_accel = 100.f;

    float delta_t = ctx.getSingleton<SolverData>().deltaT;
    float min_pos_change = max_accel * delta_t * delta_t;

    Vector3 linear_velocity = vel.linear;

#pragma unroll
    for (int32_t i = 0; i < 3; i++) {
        float pos_delta = expansion_factor * linear_velocity[i] * delta_t;

        float min_delta = pos_delta - min_pos_change;
        float max_delta = pos_delta + min_pos_change;

        if (min_delta < 0.f) {
            world_aabb.pMin[i] += min_delta;
        }
        if (max_delta > 0.f) {
            world_aabb.pMax[i] += max_delta;
        }
    }

    out_aabb = world_aabb;
}

namespace narrowphase {

enum class NarrowphaseTest : uint32_t {
    SphereSphere = 1,
    HullHull = 2,
    SphereHull = 3,
    PlanePlane = 4,
    SpherePlane = 5,
    HullPlane = 6,
};

// FIXME: Reduce redundant work on transforming point
static inline geometry::CollisionMesh buildCollisionMesh(
    const geometry::HalfEdgeMesh &he_mesh,
    Vector3 pos, Quat rot, Vector3 scale)
{
    auto transformVertex = [pos, rot, scale] (math::Vector3 v) {
        return pos + rot.rotateVec((math::Vector3)scale * v);
    };

    geometry::CollisionMesh collision_mesh;
    collision_mesh.halfEdgeMesh = &he_mesh;
    collision_mesh.vertexCount = he_mesh.getVertexCount();
    collision_mesh.vertices = (math::Vector3 *)TmpAllocator::get().alloc(
        sizeof(math::Vector3) * collision_mesh.vertexCount);
    collision_mesh.center = pos;

    for (int v = 0; v < collision_mesh.vertexCount; ++v) {
        collision_mesh.vertices[v] = transformVertex(he_mesh.vertex(v));
    }

    return collision_mesh;
}

inline void runNarrowphase(
    Context &ctx,
    const CandidateCollision &candidate_collision)
{
    ObjectID a_obj = ctx.getUnsafe<ObjectID>(candidate_collision.a);
    ObjectID b_obj = ctx.getUnsafe<ObjectID>(candidate_collision.b);

    SolverData &solver = ctx.getSingleton<SolverData>();
    const ObjectManager &obj_mgr = *ctx.getSingleton<ObjectData>().mgr;

    const CollisionPrimitive *a_prim = &obj_mgr.primitives[a_obj.idx];
    const CollisionPrimitive *b_prim = &obj_mgr.primitives[b_obj.idx];

    uint32_t raw_type_a = static_cast<uint32_t>(a_prim->type);
    uint32_t raw_type_b = static_cast<uint32_t>(b_prim->type);

    Entity a_entity = candidate_collision.a;
    Entity b_entity = candidate_collision.b;

    if (raw_type_a > raw_type_b) {
        std::swap(raw_type_a, raw_type_b);
        std::swap(a_entity, b_entity);
        std::swap(a_prim, b_prim);
    }

    NarrowphaseTest test_type {raw_type_a | raw_type_b};

    Vector3 a_pos = ctx.getUnsafe<Position>(a_entity);
    Vector3 b_pos = ctx.getUnsafe<Position>(b_entity);

    switch (test_type) {
    case NarrowphaseTest::SphereSphere: {
        float a_radius = a_prim->sphere.radius;
        float b_radius = b_prim->sphere.radius;

        Vector3 to_b = b_pos - a_pos;
        float dist = to_b.length();

        if (dist > 0 && dist < a_radius + b_radius) {
            Vector3 mid = to_b / 2.f;

            Vector3 to_b_normal = to_b / dist;
            solver.addContacts({{
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

        Quat a_rot = ctx.getUnsafe<Rotation>(a_entity);
        Quat b_rot = ctx.getUnsafe<Rotation>(b_entity);
        Vector3 a_scale = ctx.getUnsafe<Scale>(a_entity);
        Vector3 b_scale = ctx.getUnsafe<Scale>(b_entity);

        geometry::CollisionMesh a_collision_mesh =
            buildCollisionMesh(a_he_mesh, a_pos, a_rot, a_scale);

        geometry::CollisionMesh b_collision_mesh =
            buildCollisionMesh(b_he_mesh, b_pos, b_rot, b_scale);

        Manifold manifold = doSAT(a_collision_mesh, b_collision_mesh);

        if (manifold.numContactPoints > 0) {
            solver.addContacts({{
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
        // Do nothing, planes must be static.
        // Should rework this entire setup so static objects
        // aren't checked against the BVH
    } break;
    case NarrowphaseTest::SpherePlane: {
        auto sphere = a_prim->sphere;
        Quat b_rot = ctx.getUnsafe<Rotation>(b_entity);

        constexpr Vector3 base_normal = { 0, 0, 1 };
        Vector3 plane_normal = b_rot.rotateVec(base_normal);

        float d = plane_normal.dot(b_pos);
        float t = plane_normal.dot(a_pos) - d;

        float penetration = sphere.radius - t;
        if (penetration > 0) {
            Vector3 contact_point = a_pos - t * plane_normal;

            solver.addContacts({{
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
        Quat a_rot = ctx.getUnsafe<Rotation>(a_entity);
        Quat b_rot = ctx.getUnsafe<Rotation>(b_entity);
        Vector3 a_scale = ctx.getUnsafe<Scale>(a_entity);

        // Get half edge mesh for entity a (the hull)
        const auto &a_he_mesh = a_prim->hull.halfEdgeMesh;
        
        geometry::CollisionMesh a_collision_mesh =
            buildCollisionMesh(a_he_mesh, a_pos, a_rot, a_scale);

        constexpr Vector3 base_normal = { 0, 0, 1 };
        Vector3 plane_normal = b_rot.rotateVec(base_normal);

        geometry::Plane plane = { b_pos, plane_normal };

        Manifold manifold = doSATPlane(plane, a_collision_mesh);

        if (manifold.numContactPoints > 0) {
            solver.addContacts({{
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

}

namespace solver {

static inline Vector3 multDiag(Vector3 diag, Vector3 v)
{
    return Vector3 {
        diag.x * v.x,
        diag.y * v.y,
        diag.z * v.z,
    };
}

[[maybe_unused]] static inline std::tuple<float, float, float> computeEnergy(
    float inv_m, Vector3 inv_I, Vector3 v, Vector3 omega, Quat q)
{
    if (inv_m == 0.f || inv_I.x == 0.f || inv_I.y == 0.f || inv_I.z == 0.f) {
        return {0.f, 0.f, 0.f};
    }

    float m = 1.f / inv_m;
    Vector3 I {
        1.f / inv_I.x,
        1.f / inv_I.y,
        1.f / inv_I.z,
    };

    float linear = m * v.length2();

    Vector3 omega_local = q.inv().rotateVec(omega);
    float rotational = dot(omega_local, multDiag(I, omega_local));

    return {
        0.5f * (linear + rotational),
        0.5f * linear,
        0.5f * rotational,
    };
}

inline void substepRigidBodies(Context &ctx,
                               Position &pos,
                               Rotation &rot,
                               const Velocity &vel,
                               const ObjectID &obj_id,
                               SubstepPrevState &prev_state,
                               PreSolvePositional &presolve_pos,
                               PreSolveVelocity &presolve_vel)
{
    const auto &solver = ctx.getSingleton<SolverData>();
    const ObjectManager &obj_mgr = *ctx.getSingleton<ObjectData>().mgr;
    const RigidBodyMetadata &metadata = obj_mgr.metadata[obj_id.idx];
    Vector3 inv_I = metadata.invInertiaTensor;
    float inv_m = metadata.invMass;

    float h = solver.h;

    Vector3 x = pos;
    Quat q = rot;

    prev_state.prevPosition = x;
    prev_state.prevRotation = q;

    Vector3 v = vel.linear;
    Vector3 omega = vel.angular;

    // FIXME should really implement static objects differently:
    if (inv_m > 0) {
        v += h * solver.g;
    }

    // FIXME: external forces
 
    x += h * v;

    Vector3 I = {
        (inv_I.x == 0) ? 0.0f : 1.0f / inv_I.x,
        (inv_I.y == 0) ? 0.0f : 1.0f / inv_I.y,
        (inv_I.z == 0) ? 0.0f : 1.0f / inv_I.z
    };

    Quat to_local = q.inv();

    // FIXME: skip all this tau_ext stuff if it's 0
    Vector3 tau_ext { 0, 0, 0 };
    Vector3 tau_ext_local = to_local.rotateVec(tau_ext);
    Vector3 omega_local = to_local.rotateVec(omega);

    Vector3 I_omega_local = multDiag(I, omega_local);

    omega_local += h * multDiag(inv_I,
         tau_ext_local - (cross(omega_local, I_omega_local)));

    omega = q.rotateVec(omega_local);

    Quat apply_omega = Quat::fromAngularVec(0.5f * h * omega);

    q += apply_omega * q;
    q = q.normalize();

    pos = x;
    rot = q;

    presolve_pos.x = x;
    presolve_pos.q = q;
    presolve_vel.v = v;
    presolve_vel.omega = omega;
}

static inline float generalizedInverseMass(Vector3 local,
                                           float inv_m,
                                           Vector3 inv_I,
                                           Vector3 n)
{
    Vector3 lxn = cross(local, n);
    return inv_m + dot(multDiag(inv_I, lxn), lxn);
}

template <typename Fn>
static MADRONA_ALWAYS_INLINE inline void applyPositionalUpdate(
    Vector3 &x1, Vector3 &x2,
    Quat &q1, Quat &q2,
    Vector3 r1, Vector3 r2,
    float inv_m1, float inv_m2,
    Vector3 inv_I1, Vector3 inv_I2,
    Vector3 n_world, Vector3 n1, Vector3 n2,
    float c,
    float alpha_tilde,
    float &lambda,
    Fn &&lambda_check)
{
    float w1 = generalizedInverseMass(r1, inv_m1, inv_I1, n1);
    float w2 = generalizedInverseMass(r2, inv_m2, inv_I2, n2);

    float delta_lambda =
        (-c - alpha_tilde * lambda) / (w1 + w2 + alpha_tilde);

    lambda += delta_lambda;

    if (lambda_check(lambda)) return;

    Vector3 p = delta_lambda * n_world;
    Vector3 p_local1 = delta_lambda * n1;
    Vector3 p_local2 = delta_lambda * n2;

    x1 += p * inv_m1;
    x2 -= p * inv_m2;

    Vector3 r1_x_p = cross(r1, p_local1);
    Vector3 r2_x_p = cross(r2, p_local2);

    Vector3 q1_update_angular_local = 0.5f * multDiag(inv_I1, r1_x_p);
    Vector3 q1_update_angular = q1.rotateVec(q1_update_angular_local);

    Vector3 q2_update_angular_local = 0.5f * multDiag(inv_I2, r2_x_p);
    Vector3 q2_update_angular = q2.rotateVec(q2_update_angular_local);

    q1 += Quat::fromAngularVec(q1_update_angular) * q1;
    q2 -= Quat::fromAngularVec(q2_update_angular) * q2;

    // Paper doesn't explicitly call for normalization but we immediately
    // use q1 and q2 for the next constraint
    q1 = q1.normalize();
    q2 = q2.normalize();
}

static MADRONA_ALWAYS_INLINE inline void handleContactConstraint(
    Vector3 &x1, Vector3 &x2,
    Quat &q1, Quat &q2,
    SubstepPrevState prev1, SubstepPrevState prev2,
    float inv_m1, float inv_m2,
    Vector3 inv_I1, Vector3 inv_I2,
    Vector3 r1, Vector3 r2,
    Vector3 n_world,
    float avg_mu_s,
    float &lambda_n,
    float &lambda_t)
{
    Vector3 p1 = q1.rotateVec(r1) + x1;
    Vector3 p2 = q2.rotateVec(r2) + x2;

    float d = dot(p1 - p2, n_world);

    if (d <= 0) {
        return;
    }

    Vector3 x1_prev = prev1.prevPosition;
    Quat q1_prev = prev1.prevRotation;

    Vector3 x2_prev = prev2.prevPosition;
    Quat q2_prev = prev2.prevRotation;

    Vector3 p1_hat = q1_prev.rotateVec(r1) + x1_prev;
    Vector3 p2_hat = q2_prev.rotateVec(r2) + x2_prev;

    Vector3 n_local1 = q1.inv().rotateVec(n_world);
    Vector3 n_local2 = q2.inv().rotateVec(n_world);

    applyPositionalUpdate(x1, x2,
                          q1, q2,
                          r1, r2,
                          inv_m1, inv_m2,
                          inv_I1, inv_I2,
                          n_world, n_local1, n_local2,
                          d, 0,
                          lambda_n, [](float) { return false; });
 
    Vector3 delta_p = (p1 - p1_hat) - (p2 - p2_hat);
    Vector3 delta_p_t = delta_p - dot(delta_p, n_world) * n_world;

    float tangential_magnitude = delta_p_t.length();

    if (tangential_magnitude > 0.f) {
        Vector3 tangent_dir = delta_p_t / tangential_magnitude;
        Vector3 tangent_dir_local1 = q1.inv().rotateVec(tangent_dir);
        Vector3 tangent_dir_local2 = q2.inv().rotateVec(tangent_dir);

        float lambda_threshold = lambda_n * avg_mu_s;

        applyPositionalUpdate(x1, x2,
                              q1, q2,
                              r1, r2,
                              inv_m1, inv_m2,
                              inv_I1, inv_I2,
                              tangent_dir, tangent_dir_local1, tangent_dir_local2,
                              tangential_magnitude,
                              0, lambda_t, [lambda_threshold](float lambda) {
                                  // If true, this check stops static friction
                                  // from being applied.
                                  // Calculation is negated from the paper that
                                  // seems to have gotten the sign wrong
                                  // due to the negated lambdas
                                  return lambda <= lambda_threshold;
                              });
    }
}

static inline MADRONA_ALWAYS_INLINE std::pair<Vector3, Vector3>
getLocalSpaceContacts(const PreSolvePositional &presolve_pos1,
                      const PreSolvePositional &presolve_pos2,
                      const Contact &contact,
                      CountT point_idx)
{
    Vector3 contact1 = contact.points[point_idx].xyz();
    float penetration_depth = contact.points[point_idx].w;

    Vector3 contact2 = 
        contact1 - contact.normal * penetration_depth;

    // Transform the contact points into local space for a & b
    Vector3 r1 = presolve_pos1.q.inv().rotateVec(contact1 - presolve_pos1.x);
    Vector3 r2 = presolve_pos2.q.inv().rotateVec(contact2 - presolve_pos2.x);

    return { r1, r2 };
}

// For now, this function assumes both a & b are dynamic objects.
// FIXME: Need to add dynamic / static variant or handle missing the velocity
// component for static objects.
static inline void handleContact(Context &ctx,
                                 ObjectManager &obj_mgr,
                                 Contact contact,
                                 float *lambdas)
{
    Position *x1_ptr = &ctx.getUnsafe<Position>(contact.ref);
    Rotation *q1_ptr = &ctx.getUnsafe<Rotation>(contact.ref);
    SubstepPrevState prev1 = ctx.getUnsafe<SubstepPrevState>(contact.ref);
    PreSolvePositional presolve_pos1 =
        ctx.getUnsafe<PreSolvePositional>(contact.ref);
    ObjectID obj_id1 = ctx.getUnsafe<ObjectID>(contact.ref);
    RigidBodyMetadata metadata1 = obj_mgr.metadata[obj_id1.idx];

    Position *x2_ptr = &ctx.getUnsafe<Position>(contact.alt);
    Rotation *q2_ptr = &ctx.getUnsafe<Rotation>(contact.alt);
    SubstepPrevState prev2 = ctx.getUnsafe<SubstepPrevState>(contact.alt);
    PreSolvePositional presolve_pos2 =
        ctx.getUnsafe<PreSolvePositional>(contact.alt);
    ObjectID obj_id2 = ctx.getUnsafe<ObjectID>(contact.alt);
    RigidBodyMetadata metadata2 = obj_mgr.metadata[obj_id2.idx];

    Vector3 x1 = *x1_ptr;
    Vector3 x2 = *x2_ptr;

    Quat q1 = *q1_ptr;
    Quat q2 = *q2_ptr;

    float inv_m1 = metadata1.invMass;
    float inv_m2 = metadata2.invMass;

    Vector3 inv_I1 = metadata1.invInertiaTensor;
    Vector3 inv_I2 = metadata2.invInertiaTensor;

    float mu_s1 = metadata1.muS;
    float mu_s2 = metadata2.muS;

    float avg_mu_s = 0.5f * (mu_s1 + mu_s2);

#pragma unroll
    for (CountT i = 0; i < 4; i++) {
        if (i >= contact.numPoints) continue;

        auto [r1, r2] =
            getLocalSpaceContacts(presolve_pos1, presolve_pos2, contact, i);

        float lambda_n = 0.f;
        float lambda_t = 0.f;

        handleContactConstraint(x1, x2,
                                q1, q2,
                                prev1, prev2,
                                inv_m1, inv_m2,
                                inv_I1, inv_I2,
                                r1, r2,
                                contact.normal,
                                avg_mu_s,
                                lambda_n,
                                lambda_t);

        lambdas[i] = lambda_n;
    }

    *x1_ptr = x1;
    *x2_ptr = x2;

    *q1_ptr = q1;
    *q2_ptr = q2;
}

inline void solvePositions(Context &ctx, SolverData &solver)
{
    ObjectManager &obj_mgr = *ctx.getSingleton<ObjectData>().mgr;

    // Push objects in serial based on the contact normal - total BS.
    CountT num_contacts = solver.numContacts.load(std::memory_order_relaxed);

    //printf("Solver # contacts: %d\n", num_contacts);

    for (CountT i = 0; i < num_contacts; i++) {
        Contact contact = solver.contacts[i];
        handleContact(ctx, obj_mgr, contact, solver.contacts[i].lambdaN);
    }
}

inline void setVelocities(Context &ctx,
                          const Position &pos,
                          const Rotation &rot,
                          const SubstepPrevState &prev_state,
                          Velocity &vel)
{
    const auto &solver = ctx.getSingleton<SolverData>();
    float h = solver.h;

    Vector3 x = pos;
    Quat q = rot;

    Vector3 x_prev = prev_state.prevPosition;
    Quat q_prev = prev_state.prevRotation;

    // when cur and prev rotation are equal there should be 0 angular velocity
    // Unfortunately, this computation introduces a small amount of FP error
    // and in some rotations the object winds up with a small delta_q, so
    // we do a check and if all components match, just set delta_q to the
    // identity quaternion manually
    Quat delta_q;
    if (q.w != q_prev.w || q.x != q_prev.x ||
            q.y != q_prev.y || q.z != q_prev.z) {
        delta_q = q * q_prev.inv();
    } else {
        delta_q = { 1, 0, 0, 0 };
    }

    // FIXME: A noticeable amount of energy is lost for bodies that should
    // be under a constant angular velocity with no resistance. The issue
    // seems to be that delta_q here is consistently slightly less than
    // apply_omega in substepRigidBodies, so the angular velocity reduces
    // a little bit each frame. Investigate whether this is an FP
    // precision issue or a consequence of linearized angular -> quaternion
    // formulas and see if it can be mitigated. Dividing delta_q by delta_q.w
    // seems to fix the issue - does that have unintended consequences?
    Vector3 new_omega =
        2.f / h * Vector3 { delta_q.x, delta_q.y, delta_q.z };

    vel.linear = (x - x_prev) / h;
    vel.angular = delta_q.w > 0.f ? new_omega : -new_omega;
}

static inline Vector3 computeRelativeVelocity(
    Vector3 v1, Vector3 v2,
    Vector3 omega1, Vector3 omega2,
    Vector3 dir1, Vector3 dir2)
{
    return (v1 + cross(omega1, dir1)) - (v2 + cross(omega2, dir2));
}

static inline void applyVelocityUpdate(Vector3 &v1, Vector3 &v2,
                                       Vector3 &omega1, Vector3 &omega2,
                                       Quat q1, Quat q2,
                                       Vector3 r1, Vector3 r2,
                                       float inv_m1, float inv_m2,
                                       Vector3 inv_I1, Vector3 inv_I2,
                                       Vector3 delta_v_world,
                                       Vector3 delta_v_l1, Vector3 delta_v_l2,
                                       float delta_v_magnitude)
{
    float w1 = generalizedInverseMass(r1, inv_m1, inv_I1, delta_v_l1);
    float w2 = generalizedInverseMass(r2, inv_m2, inv_I2, delta_v_l2);

    delta_v_magnitude *= 1.f / (w1 + w2);

    v1 += delta_v_world * delta_v_magnitude * inv_m1;
    v2 -= delta_v_world * delta_v_magnitude * inv_m2;

    Vector3 omega1_update_local =
        multDiag(inv_I1, cross(r1, delta_v_l1 * delta_v_magnitude));
    Vector3 omega2_update_local =
        multDiag(inv_I2, cross(r2, delta_v_l2 * delta_v_magnitude));

    omega1 += q1.rotateVec(omega1_update_local);
    omega2 -= q2.rotateVec(omega2_update_local);
}

static inline void updateVelocityFromContact(Context &ctx,
                                             ObjectManager &obj_mgr,
                                             Contact contact,
                                             float h,
                                             float restitution_threshold)
{
    Velocity *v1_out = &ctx.getUnsafe<Velocity>(contact.ref);
    Quat q1 = ctx.getUnsafe<Rotation>(contact.ref);
    PreSolvePositional presolve_pos1 =
        ctx.getUnsafe<PreSolvePositional>(contact.ref);
    PreSolveVelocity presolve_vel1 =
        ctx.getUnsafe<PreSolveVelocity>(contact.ref);
    ObjectID obj_id1 = ctx.getUnsafe<ObjectID>(contact.ref);
    RigidBodyMetadata metadata1 = obj_mgr.metadata[obj_id1.idx];

    Velocity *v2_out = &ctx.getUnsafe<Velocity>(contact.alt);
    Quat q2 = ctx.getUnsafe<Rotation>(contact.alt);
    PreSolvePositional presolve_pos2 =
        ctx.getUnsafe<PreSolvePositional>(contact.alt);
    PreSolveVelocity presolve_vel2 =
        ctx.getUnsafe<PreSolveVelocity>(contact.alt);
    ObjectID obj_id2 = ctx.getUnsafe<ObjectID>(contact.alt);
    RigidBodyMetadata metadata2 = obj_mgr.metadata[obj_id2.idx];

    auto [v1, omega1] = *v1_out;
    auto [v2, omega2] = *v2_out;

    float inv_m1 = metadata1.invMass;
    float inv_m2 = metadata2.invMass;
    Vector3 inv_I1 = metadata1.invInertiaTensor;
    Vector3 inv_I2 = metadata2.invInertiaTensor;
    float mu_d = 0.5f * (metadata1.muD + metadata2.muD);

#pragma unroll
    for (CountT i = 0; i < 4; i++) {
        if (i >= contact.numPoints) continue;

        // FIXME: If this contact point wasn't actually processed just skip it
        if (contact.lambdaN[i] == 0.f) continue;

        // FIXME: reconsider separate lambdas?
        // h * mu_d * |f_n| in paper
        float dynamic_friction_magnitude =
            mu_d * fabsf(contact.lambdaN[i]) / h;

        auto [r1, r2] =
            getLocalSpaceContacts(presolve_pos1, presolve_pos2, contact, i);
        Vector3 r1_world = q1.rotateVec(r1);
        Vector3 r2_world = q2.rotateVec(r2);

        Vector3 v = computeRelativeVelocity(
            v1, v2, omega1, omega2, r1_world, r2_world);

        Vector3 n = contact.normal;
        Vector3 n_local1 = q1.inv().rotateVec(n);
        Vector3 n_local2 = q2.inv().rotateVec(n);

        float vn = dot(n, v);
        Vector3 vt = v - n * vn;

        float vt_len = vt.length();

        if (vt_len != 0 && dynamic_friction_magnitude != 0.f) {
            float corrected_magnitude =
                -fminf(dynamic_friction_magnitude, vt_len);

            Vector3 delta_world = vt / vt_len;

            Vector3 delta_local1 = q1.inv().rotateVec(delta_world);
            Vector3 delta_local2 = q2.inv().rotateVec(delta_world);

            applyVelocityUpdate(
                v1, v2,
                omega1, omega2,
                q1, q2,
                r1, r2,
                inv_m1, inv_m2,
                inv_I1, inv_I2,
                delta_world, delta_local1, delta_local2,
                corrected_magnitude);
        }

        Vector3 r1_presolve = presolve_pos1.q.rotateVec(r1);
        Vector3 r2_presolve = presolve_pos2.q.rotateVec(r2);

        Vector3 v_bar = computeRelativeVelocity(
            presolve_vel1.v, presolve_vel2.v,
            presolve_vel1.omega, presolve_vel2.omega,
            r1_presolve, r2_presolve); // FIXME r1_world or presolve?

        float vn_bar = dot(n, v_bar);

        float e = 0.3f; // FIXME
        if (fabsf(vn_bar) <= restitution_threshold) {
            e = 0.f;
        }
        float restitution_magnitude = fminf(-e * vn_bar, 0) - vn;

        applyVelocityUpdate(
            v1, v2,
            omega1, omega2,
            q1, q2,
            r1, r2,
            inv_m1, inv_m2,
            inv_I1, inv_I2,
            n, n_local1, n_local2, restitution_magnitude);
    }

    *v1_out = Velocity { v1, omega1 };
    *v2_out = Velocity { v2, omega2 };
}

inline void solveVelocities(Context &ctx, SolverData &solver)
{
    ObjectManager &obj_mgr = *ctx.getSingleton<ObjectData>().mgr;

    CountT num_contacts = solver.numContacts.load(std::memory_order_relaxed);

    for (CountT i = 0; i < num_contacts; i++) {
        const Contact &contact = solver.contacts[i];
        updateVelocityFromContact(ctx, obj_mgr, contact, solver.h,
                                  solver.restitutionThreshold);
    }

    solver.numContacts.store(0, std::memory_order_relaxed);
}

}

void RigidBodyPhysicsSystem::init(Context &ctx,
                                  ObjectManager *obj_mgr,
                                  float delta_t,
                                  CountT num_substeps,
                                  math::Vector3 gravity,
                                  CountT max_dynamic_objects,
                                  CountT max_contacts_per_world)
{
    broadphase::BVH &bvh = ctx.getSingleton<broadphase::BVH>();
    new (&bvh) broadphase::BVH(max_dynamic_objects);

    SolverData &solver = ctx.getSingleton<SolverData>();
    new (&solver) SolverData(max_contacts_per_world, delta_t, num_substeps, gravity);

    ObjectData &objs = ctx.getSingleton<ObjectData>();
    new (&objs) ObjectData { obj_mgr };
}

void RigidBodyPhysicsSystem::reset(Context &ctx)
{
    broadphase::BVH &bvh = ctx.getSingleton<broadphase::BVH>();
    bvh.rebuildOnUpdate();
    bvh.clearLeaves();
}

broadphase::LeafID RigidBodyPhysicsSystem::registerEntity(Context &ctx,
                                                          Entity e)
{
    return ctx.getSingleton<broadphase::BVH>().reserveLeaf(e);
}

void RigidBodyPhysicsSystem::registerTypes(ECSRegistry &registry)
{
    registry.registerComponent<broadphase::LeafID>();
    registry.registerSingleton<broadphase::BVH>();

    registry.registerComponent<Velocity>();
    registry.registerComponent<CollisionAABB>();

    registry.registerComponent<solver::SubstepPrevState>();
    registry.registerComponent<solver::PreSolvePositional>();
    registry.registerComponent<solver::PreSolveVelocity>();

    registry.registerComponent<CollisionEvent>();
    registry.registerArchetype<CollisionEventTemporary>();

    registry.registerComponent<CandidateCollision>();
    registry.registerArchetype<CandidateTemporary>();

    registry.registerSingleton<SolverData>();
    registry.registerSingleton<ObjectData>();

}

TaskGraph::NodeID RigidBodyPhysicsSystem::setupTasks(
    TaskGraph::Builder &builder, Span<const TaskGraph::NodeID> deps,
    CountT num_substeps)
{
    auto update_aabbs = builder.addToGraph<ParallelForNode<Context,
        updateCollisionAABB, Position, Rotation, ObjectID, Velocity,
            CollisionAABB>>(deps);

    auto preprocess_leaves = builder.addToGraph<ParallelForNode<Context,
        broadphase::updateLeavesEntry, broadphase::LeafID, 
        CollisionAABB>>({update_aabbs});

    auto bvh_update = builder.addToGraph<ParallelForNode<Context,
        broadphase::updateBVHEntry, broadphase::BVH>>({preprocess_leaves});

    auto find_overlapping = builder.addToGraph<ParallelForNode<Context,
        broadphase::findOverlappingEntry, Entity, CollisionAABB, Velocity>>(
            {bvh_update});
    
    auto cur_node = find_overlapping;
    for (CountT i = 0; i < num_substeps; i++) {
        auto rgb_update = builder.addToGraph<ParallelForNode<Context,
            solver::substepRigidBodies, Position, Rotation, Velocity, ObjectID,
            solver::SubstepPrevState, solver::PreSolvePositional,
            solver::PreSolveVelocity>>({cur_node});

        auto run_narrowphase = builder.addToGraph<ParallelForNode<Context,
            narrowphase::runNarrowphase, CandidateCollision>>(
                {rgb_update});
        auto reset_tmp = builder.addToGraph<ResetTmpAllocNode>(
            {run_narrowphase});

        auto solve_pos = builder.addToGraph<ParallelForNode<Context,
            solver::solvePositions, SolverData>>({reset_tmp});

        auto vel_set = builder.addToGraph<ParallelForNode<Context,
            solver::setVelocities, Position, Rotation,
            solver::SubstepPrevState, Velocity>>({solve_pos});

        auto solve_vel = builder.addToGraph<ParallelForNode<Context,
            solver::solveVelocities, SolverData>>({vel_set});

        cur_node = builder.addToGraph<ResetTmpAllocNode>({solve_vel});
    }

    auto clear_candidates = builder.addToGraph<
        ClearTmpNode<CandidateTemporary>>({cur_node});

    return clear_candidates;
}

TaskGraph::NodeID RigidBodyPhysicsSystem::setupCleanupTasks(
    TaskGraph::Builder &builder, Span<const TaskGraph::NodeID> deps)
{
    return builder.addToGraph<ClearTmpNode<CollisionEventTemporary>>(deps);
}

}
}
