#include <madrona/physics.hpp>
#include <madrona/context.hpp>

#include "physics_impl.hpp"

namespace madrona::phys {

using namespace base;
using namespace math;

SolverData::SolverData(CountT max_contacts_per_step,
                       CountT max_joint_constraints,
                       float delta_t,
                       CountT num_substeps,
                       Vector3 gravity)
    : contacts((Contact *)rawAlloc(
          sizeof(Contact) * max_contacts_per_step)),
      numContacts(0),
      jointConstraints((JointConstraint *)rawAlloc(
          sizeof(JointConstraint) * max_joint_constraints)),
      numJointConstraints(0),
      maxContacts(max_contacts_per_step),
      deltaT(delta_t),
      h(delta_t / (float)num_substeps),
      g(gravity),
      gMagnitude(gravity.length()),
      restitutionThreshold(2.f * gMagnitude * h)
{}

inline void collectConstraintsSystem(Context &ctx,
                                     JointConstraint &constraint)
{
    auto &solver = ctx.singleton<SolverData>();
    solver.jointConstraints[
        solver.numJointConstraints.fetch_add_relaxed(1)] = constraint;
}

namespace solver {

static inline bool hasNaN(Vector3 v)
{
    return isnan(v.x) || isnan(v.y) || isnan(v.z);
}

static inline bool hasNaN(Quat q)
{
    return isnan(q.w) || isnan(q.x) || isnan(q.y) || isnan(q.z);
}

static inline Vector3 multDiag(Vector3 diag, Vector3 v)
{
    return Vector3 {
        diag.x * v.x,
        diag.y * v.y,
        diag.z * v.z,
    };
}

[[maybe_unused]] static inline Vector3 computeEnergy(
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
                               ResponseType response_type,
                               ExternalForce &ext_force,
                               ExternalTorque &ext_torque,
                               SubstepPrevState &prev_state,
                               PreSolvePositional &presolve_pos,
                               PreSolveVelocity &presolve_vel)
{
    Vector3 x = pos;
    Quat q = rot;

    Vector3 v = vel.linear;
    Vector3 omega = vel.angular;

    if (response_type == ResponseType::Static) {
        // FIXME: currently presolve_pos and prev_state need to be set every
        // frame even for static objects. A better solution would be on
        // creation / making a non-static object static, these variables are
        // set once. This would require a dedicated API to control this rather
        // than just setting objects to ResponseType::Static
        prev_state.prevPosition = x;
        prev_state.prevRotation = q;

        presolve_pos.x = x;
        presolve_pos.q = q;
        presolve_vel.v = Vector3::zero();
        presolve_vel.omega = Vector3::zero();

        return;
    }

    prev_state.prevPosition = x;
    prev_state.prevRotation = q;

    const auto &solver = ctx.singleton<SolverData>();
    const ObjectManager &obj_mgr = *ctx.singleton<ObjectData>().mgr;
    const RigidBodyMetadata &metadata = obj_mgr.metadata[obj_id.idx];

    float inv_m = metadata.mass.invMass;
    Vector3 inv_I = metadata.mass.invInertiaTensor;

    float h = solver.h;

    if (response_type == ResponseType::Dynamic) {
        v += h * solver.g;
    }
    
    v += h * inv_m * ext_force;
 
    x += h * v;

    Vector3 I = {
        (inv_I.x == 0) ? 0.0f : 1.0f / inv_I.x,
        (inv_I.y == 0) ? 0.0f : 1.0f / inv_I.y,
        (inv_I.z == 0) ? 0.0f : 1.0f / inv_I.z
    };

    Quat to_local = q.inv();

    Vector3 tau_ext_local = to_local.rotateVec(ext_torque);
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

[[maybe_unused]] inline void checkSubstep(Context &,
                                          Entity,
                                          const Position &pos,
                                          const Rotation &rot,
                                          const Velocity &vel,
                                          const ObjectID &,
                                          const ResponseType &,
                                          const ExternalForce &,
                                          const ExternalTorque &,
                                          const SubstepPrevState &,
                                          const PreSolvePositional &,
                                          const PreSolveVelocity &)
{
    assert(!hasNaN(pos));
    assert(!hasNaN(rot));
    assert(!hasNaN(vel.linear));
    assert(!hasNaN(vel.angular));
}

static inline float generalizedInverseMass(Vector3 torque_axis,
                                           Vector3 rot_axis,
                                           float inv_m)
{
    return inv_m + dot(torque_axis, rot_axis);
}

float computePositionalLambda(
    Vector3 torque_axis1, Vector3 torque_axis2,
    Vector3 rot_axis1, Vector3 rot_axis2,
    float inv_m1, float inv_m2,
    float c, float alpha_tilde)
{
    float w1 = generalizedInverseMass(torque_axis1, rot_axis1, inv_m1);
    float w2 = generalizedInverseMass(torque_axis2, rot_axis2, inv_m2);

    return -c / (w1 + w2 + alpha_tilde);
}

MADRONA_ALWAYS_INLINE static inline void applyPositionalUpdate(
    Vector3 &x1, Vector3 &x2,
    Quat &q1, Quat &q2,
    Vector3 rot_axis_local1, Vector3 rot_axis_local2,
    float inv_m1, float inv_m2,
    Vector3 n,
    float delta_lambda)
{
    x1 += delta_lambda * inv_m1 * n;
    x2 -= delta_lambda * inv_m2 * n;

    float half_lambda = 0.5f * delta_lambda;

    Vector3 q1_update_angular_local = half_lambda * rot_axis_local1;
    Vector3 q1_update_angular = q1.rotateVec(q1_update_angular_local);

    Vector3 q2_update_angular_local = half_lambda * rot_axis_local2;
    Vector3 q2_update_angular = q2.rotateVec(q2_update_angular_local);

    q1 += Quat::fromAngularVec(q1_update_angular) * q1;
    q2 -= Quat::fromAngularVec(q2_update_angular) * q2;

    // Paper doesn't explicitly call for normalization but we immediately
    // use q1 and q2 for the next constraint
    q1 = q1.normalize();
    q2 = q2.normalize();
}

MADRONA_ALWAYS_INLINE static inline float applyPositionalUpdate(
    Vector3 &x1, Vector3 &x2,
    Quat &q1, Quat &q2,
    Vector3 r1, Vector3 r2,
    float inv_m1, float inv_m2,
    Vector3 inv_I1, Vector3 inv_I2,
    Vector3 n_world,
    float c, float alpha_tilde)
{
    Vector3 n_local1 = q1.inv().rotateVec(n_world);
    Vector3 n_local2 = q2.inv().rotateVec(n_world);

    Vector3 torque_axis_local1 = cross(r1, n_local1);
    Vector3 torque_axis_local2 = cross(r2, n_local2);

    Vector3 rot_axis_local1 = multDiag(inv_I1, torque_axis_local1);
    Vector3 rot_axis_local2 = multDiag(inv_I2, torque_axis_local2);

    float lambda = computePositionalLambda(
        torque_axis_local1, torque_axis_local2,
        rot_axis_local1, rot_axis_local2,
        inv_m1, inv_m2,
        c, alpha_tilde);

    applyPositionalUpdate(
        x1, x2,
        q1, q2,
        rot_axis_local1, rot_axis_local2,
        inv_m1, inv_m2,
        n_world, lambda);

    return lambda;
}

MADRONA_ALWAYS_INLINE static inline
std::pair<Quat, Quat> computeAngularUpdate(
    Quat q1, Quat q2,
    Vector3 inv_I1, Vector3 inv_I2,
    Vector3 n1, Vector3 n2,
    float theta,
    float alpha_tilde)
{
    Vector3 local_rot_axis1 = multDiag(inv_I1, n1);
    Vector3 local_rot_axis2 = multDiag(inv_I2, n2);

    float w1 = dot(n1, local_rot_axis1);
    float w2 = dot(n2, local_rot_axis2);

    float delta_lambda = -theta / (w1 + w2 + alpha_tilde);

    float half_lambda = 0.5f * delta_lambda;
    Vector3 q1_update_angular_local = half_lambda * local_rot_axis1;
    Vector3 q2_update_angular_local = half_lambda * local_rot_axis2;

    return {
        Quat::fromAngularVec(q1.rotateVec(q1_update_angular_local)),
        Quat::fromAngularVec(q2.rotateVec(q2_update_angular_local)),
    };
}

void applyAngularUpdate(
    Quat &q1, Quat &q2,
    Quat q1_update, Quat q2_update)
{
    q1 = (q1 + q1_update * q1).normalize();
    q2 = (q2 - q2_update * q2).normalize();
}

MADRONA_ALWAYS_INLINE static inline void handleContactConstraint(
    Vector3 &x1, Vector3 &x2,
    Quat &q1, Quat &q2,
    SubstepPrevState prev1, SubstepPrevState prev2,
    float inv_m1, float inv_m2,
    Vector3 inv_I1, Vector3 inv_I2,
    Vector3 r1, Vector3 r2,
    Vector3 n_world,
    float avg_mu_s,
    float *lambda_n_out,
    float *lambda_t_out)
{
    Vector3 p1 = q1.rotateVec(r1) + x1;
    Vector3 p2 = q2.rotateVec(r2) + x2;

    float d = dot(p1 - p2, n_world);

    if (d <= 0) {
        return;
    }

    float lambda_n = applyPositionalUpdate(
        x1, x2,
        q1, q2,
        r1, r2,
        inv_m1, inv_m2,
        inv_I1, inv_I2,
        n_world,
        d, 0);
    *lambda_n_out = lambda_n;
     
    Vector3 x1_prev = prev1.prevPosition;
    Quat q1_prev = prev1.prevRotation;

    Vector3 x2_prev = prev2.prevPosition;
    Quat q2_prev = prev2.prevRotation;

    Vector3 p1_hat = q1_prev.rotateVec(r1) + x1_prev;
    Vector3 p2_hat = q2_prev.rotateVec(r2) + x2_prev;

    // Update p1 and p2 so static friction covers any drift
    // as a result of the positional correction along the normal
    p1 = q1.rotateVec(r1) + x1;
    p2 = q2.rotateVec(r2) + x2;

    Vector3 delta_p = (p1 - p1_hat) - (p2 - p2_hat);
    Vector3 delta_p_t = delta_p - dot(delta_p, n_world) * n_world;

    float tangential_magnitude = delta_p_t.length();

    if (tangential_magnitude > 0.f) {
        Vector3 t_world = delta_p_t / tangential_magnitude;
        Vector3 t_local1 = q1.inv().rotateVec(t_world);
        Vector3 t_local2 = q2.inv().rotateVec(t_world);

        Vector3 friction_torque_axis_local1 = cross(r1, t_local1);
        Vector3 friction_torque_axis_local2 = cross(r2, t_local2);

        Vector3 friction_rot_axis_local1 =
            multDiag(inv_I1, friction_torque_axis_local1);

        Vector3 friction_rot_axis_local2 =
            multDiag(inv_I2, friction_torque_axis_local2);

        float lambda_t = computePositionalLambda(
            friction_torque_axis_local1, friction_torque_axis_local2,
            friction_rot_axis_local1, friction_rot_axis_local2,
            inv_m1, inv_m2,
            tangential_magnitude, 0);
        float lambda_threshold = lambda_n * avg_mu_s;

        if (lambda_t > lambda_threshold) {
            *lambda_t_out = lambda_t;
            
            applyPositionalUpdate(
                x1, x2,
                q1, q2,
                friction_rot_axis_local1, friction_rot_axis_local2,
                inv_m1, inv_m2,
                t_world, lambda_t);
        }
    }
}

MADRONA_ALWAYS_INLINE static inline std::pair<Vector3, Vector3>
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
    Position *x1_ptr = &ctx.getDirect<Position>(Cols::Position, contact.ref);
    Position *x2_ptr = &ctx.getDirect<Position>(Cols::Position, contact.alt);

    Rotation *q1_ptr = &ctx.getDirect<Rotation>(Cols::Rotation, contact.ref);
    Rotation *q2_ptr = &ctx.getDirect<Rotation>(Cols::Rotation, contact.alt);

    SubstepPrevState prev1 = ctx.getDirect<SubstepPrevState>(
        Cols::SubstepPrevState, contact.ref);
    SubstepPrevState prev2 = ctx.getDirect<SubstepPrevState>(
        Cols::SubstepPrevState, contact.alt);

    PreSolvePositional presolve_pos1 = ctx.getDirect<PreSolvePositional>(
        Cols::PreSolvePositional, contact.ref);
    PreSolvePositional presolve_pos2 = ctx.getDirect<PreSolvePositional>(
        Cols::PreSolvePositional, contact.alt);

    ObjectID obj_id1 = ctx.getDirect<ObjectID>(
        Cols::ObjectID, contact.ref);
    ObjectID obj_id2 = ctx.getDirect<ObjectID>(Cols::ObjectID, contact.alt);

    ResponseType resp_type1 = ctx.getDirect<ResponseType>(
        Cols::ResponseType, contact.ref);
    ResponseType resp_type2 = ctx.getDirect<ResponseType>(
        Cols::ResponseType, contact.alt);

    RigidBodyMetadata metadata1 = obj_mgr.metadata[obj_id1.idx];
    RigidBodyMetadata metadata2 = obj_mgr.metadata[obj_id2.idx];

    Vector3 x1 = *x1_ptr;
    Vector3 x2 = *x2_ptr;

    Quat q1 = *q1_ptr;
    Quat q2 = *q2_ptr;

    float inv_m1 = metadata1.mass.invMass;
    float inv_m2 = metadata2.mass.invMass;

    Vector3 inv_I1 = metadata1.mass.invInertiaTensor;
    Vector3 inv_I2 = metadata2.mass.invInertiaTensor;

    if (resp_type1 == ResponseType::Static) {
        inv_m1 = 0.f;
        inv_I1 = Vector3::zero();
    }

    if (resp_type2 == ResponseType::Static) {
        inv_m2 = 0.f;
        inv_I2 = Vector3::zero();
    }

    float mu_s1 = metadata1.friction.muS;
    float mu_s2 = metadata2.friction.muS;

    float avg_mu_s = 0.5f * (mu_s1 + mu_s2);

#pragma unroll
    for (CountT i = 4 - 1; i >= 0; i--) {
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
                                &lambda_n,
                                &lambda_t);

        lambdas[i] = lambda_n;
    }

    *x1_ptr = x1;
    *x2_ptr = x2;

    *q1_ptr = q1;
    *q2_ptr = q2;
}

static void applyJointOrientationConstraint(
    Quat &q1, Quat &q2,
    Quat attach_q1, Quat attach_q2,
    Vector3 inv_I1, Vector3 inv_I2)
{
    Quat orientation1 = (q1 * attach_q1).normalize();
    Quat orientation2 = (q2 * attach_q2).normalize();

    Quat diff = orientation1 * orientation2.inv();

    Vector3 delta_q =
        2.f * math::Vector3 { diff.x, diff.y, diff.z };
    float delta_q_magnitude = delta_q.length();

    if (delta_q_magnitude > 0) {
        delta_q /= delta_q_magnitude;
        Vector3 delta_q_local1 = q1.inv().rotateVec(delta_q);
        Vector3 delta_q_local2 = q2.inv().rotateVec(delta_q);

        auto [update_q1, update_q2] = computeAngularUpdate(
            q1, q2,
            inv_I1, inv_I2,
            delta_q_local1, delta_q_local2,
            delta_q_magnitude, 0);

        applyAngularUpdate(q1, q2, update_q1, update_q2);
    }
}

static void applyJointAxisConstraint(
    Quat &q1, Quat &q2,
    Vector3 axis1_local, Vector3 axis2_local,
    Vector3 inv_I1, Vector3 inv_I2)
{
    Vector3 axis1 = q1.rotateVec(axis1_local);
    Vector3 axis2 = q2.rotateVec(axis2_local);

    Vector3 delta_q = cross(axis1, axis2);
    float delta_q_magnitude = delta_q.length();
    
    if (delta_q_magnitude > 0) {
        delta_q /= delta_q_magnitude;
        Vector3 delta_q_local1 = q1.inv().rotateVec(delta_q);
        Vector3 delta_q_local2 = q2.inv().rotateVec(delta_q);

        auto [update_q1, update_q2] = computeAngularUpdate(
            q1, q2,
            inv_I1, inv_I2,
            delta_q_local1, delta_q_local2,
            delta_q_magnitude, 0);
        
        applyAngularUpdate(q1, q2, update_q1, update_q2);
    }
}

inline void handleJointConstraint(Context &ctx,
                                  JointConstraint joint)
{
    Loc l1 = ctx.loc(joint.e1);
    Loc l2 = ctx.loc(joint.e2);

    Vector3 *x1_ptr = &ctx.getDirect<Position>(Cols::Position, l1);
    Vector3 *x2_ptr = &ctx.getDirect<Position>(Cols::Position, l2);
    Quat *q1_ptr = &ctx.getDirect<Rotation>(Cols::Rotation, l1);
    Quat *q2_ptr = &ctx.getDirect<Rotation>(Cols::Rotation, l2);
    Vector3 x1 = *x1_ptr;
    Vector3 x2 = *x2_ptr;
    Quat q1 = *q1_ptr;
    Quat q2 = *q2_ptr;
    ResponseType resp_type1 = ctx.getDirect<ResponseType>(
        Cols::ResponseType, l1);
    ResponseType resp_type2 = ctx.getDirect<ResponseType>(
        Cols::ResponseType, l2);
    ObjectID obj_id1 = ctx.getDirect<ObjectID>(Cols::ObjectID, l1);
    ObjectID obj_id2 = ctx.getDirect<ObjectID>(Cols::ObjectID, l2);

    ObjectManager &obj_mgr = *ctx.singleton<ObjectData>().mgr;
    RigidBodyMetadata metadata1 = obj_mgr.metadata[obj_id1.idx];
    RigidBodyMetadata metadata2 = obj_mgr.metadata[obj_id2.idx];

    float inv_m1 = metadata1.mass.invMass;
    Vector3 inv_I1 = metadata1.mass.invInertiaTensor;

    if (resp_type1 == ResponseType::Static) {
        inv_m1 = 0.f;
        inv_I1 = Vector3::zero();
    }

    float inv_m2 = metadata2.mass.invMass;
    Vector3 inv_I2 = metadata2.mass.invInertiaTensor;

    if (resp_type2 == ResponseType::Static) {
        inv_m2 = 0.f;
        inv_I2 = Vector3::zero();
    }

    Vector3 pos_correction;
    switch (joint.type) {
    case JointConstraint::Type::Fixed: {
        JointConstraint::Fixed fixed_data = joint.fixed;

        applyJointOrientationConstraint(
            q1, q2,
            fixed_data.attachRot1, fixed_data.attachRot2,
            inv_I1, inv_I2);

        Vector3 r1_world = q1.rotateVec(joint.r1) + x1;
        Vector3 r2_world = q2.rotateVec(joint.r2) + x2;

        Vector3 delta_r = r2_world - r1_world;

        Quat axes_rot = (q1 * fixed_data.attachRot1).normalize();

        Vector3 a1 = axes_rot.rotateVec(math::fwd);
        Vector3 b1 = axes_rot.rotateVec(math::right);
        Vector3 c1 = cross(a1, b1);

        // This implements a fixed distance qlong the a1 axis and no distance
        // along the other axes

        pos_correction = Vector3::zero();
        {
            // Unlike paper, subtract from pos_correction because
            // applyPositionalUpdate applies the negative magnitude to object 1
            float a_separation = dot(delta_r, a1);
            pos_correction -= (a_separation - fixed_data.separation) * a1;

            float b_separation = dot(delta_r, b1);
            pos_correction -= b_separation * b1;

            float c_separation = dot(delta_r, c1);
            pos_correction -= c_separation * c1;
        }
    } break;
    case JointConstraint::Type::Hinge: {
        JointConstraint::Hinge hinge_data = joint.hinge;

        applyJointAxisConstraint(q1, q2,
                                 hinge_data.a1Local, hinge_data.a2Local,
                                 inv_I1, inv_I2);

        Vector3 r1_world = q1.rotateVec(joint.r1) + x1;
        Vector3 r2_world = q2.rotateVec(joint.r2) + x2;

        pos_correction = r2_world - r1_world;
    } break;
    default: MADRONA_UNREACHABLE();
    }

    float pos_correction_magnitude = pos_correction.length();
    if (pos_correction_magnitude > 0.f) {
        pos_correction /= pos_correction_magnitude;

        applyPositionalUpdate(
            x1, x2,
            q1, q2,
            joint.r1, joint.r2,
            inv_m1, inv_m2,
            inv_I1, inv_I2,
            pos_correction, pos_correction_magnitude, 0);
    }

    *x1_ptr = x1;
    *x2_ptr = x2;
    *q1_ptr = q1;
    *q2_ptr = q2;
}

inline void solvePositions(Context &ctx, SolverData &solver)
{
    ObjectManager &obj_mgr = *ctx.singleton<ObjectData>().mgr;

    CountT num_contacts = solver.numContacts.load_relaxed();

    for (CountT i = 0; i < num_contacts; i++) {
        Contact contact = solver.contacts[i];
        handleContact(ctx, obj_mgr, contact, solver.contacts[i].lambdaN);
    }

    CountT num_joint_constraints = solver.numJointConstraints.load_relaxed();

    for (CountT i = 0; i < num_joint_constraints; i++) {
        JointConstraint joint_constraint = solver.jointConstraints[i];
        handleJointConstraint(ctx, joint_constraint);
    }

    solver.numJointConstraints.store_relaxed(0);
}

inline void setVelocities(Context &ctx,
                          const Position &pos,
                          const Rotation &rot,
                          const SubstepPrevState &prev_state,
                          Velocity &vel)
{
    const auto &solver = ctx.singleton<SolverData>();
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

static inline void applyVelocityUpdate(
    Vector3 &v1, Vector3 &v2,
    Vector3 &omega1, Vector3 &omega2,
    Quat q1, Quat q2,
    Vector3 torque_axis1, Vector3 torque_axis2,
    float inv_m1, float inv_m2,
    Vector3 inv_I1, Vector3 inv_I2,
    Vector3 delta_v,
    float delta_v_magnitude)
{
    Vector3 rot_axis1 = multDiag(inv_I1, torque_axis1);
    Vector3 rot_axis2 = multDiag(inv_I2, torque_axis2);

    float w1 = generalizedInverseMass(torque_axis1, rot_axis1, inv_m1);
    float w2 = generalizedInverseMass(torque_axis2, rot_axis2, inv_m2);

    delta_v_magnitude *= 1.f / (w1 + w2);

    v1 += delta_v_magnitude * inv_m1 * delta_v;
    v2 -= delta_v_magnitude * inv_m2 * delta_v;

    Vector3 omega1_update_local = delta_v_magnitude * rot_axis1;
    Vector3 omega2_update_local = delta_v_magnitude * rot_axis2;

    omega1 += q1.rotateVec(omega1_update_local);
    omega2 -= q2.rotateVec(omega2_update_local);
}

static inline void applyFrictionVelocityUpdate(
    Vector3 &v1, Vector3 &v2,
    Vector3 &omega1, Vector3 &omega2,
    Quat q1, Quat q2,
    float inv_m1, float inv_m2,
    Vector3 inv_I1, Vector3 inv_I2,
    Vector3 n,
    float mu_d, float h,
    Vector3 r1_locals[4], Vector3 r2_locals[4],
    Vector3 r1_worlds[4], Vector3 r2_worlds[4],
    float lambdas[4],
    CountT num_contacts)
{
#pragma unroll
    for (CountT i = 0; i < 4; i++) {
        if (i >= num_contacts) continue;

        Vector3 r1_local = r1_locals[i];
        Vector3 r2_local = r2_locals[i];

        Vector3 r1_world = r1_worlds[i];
        Vector3 r2_world = r2_worlds[i];

        Vector3 v = computeRelativeVelocity(
            v1, v2, omega1, omega2, r1_world, r2_world);

        // h * mu_d * |f_n| in paper
        float dynamic_friction_magnitude =
            mu_d * fabsf(lambdas[i]) / h;

        float vn = dot(n, v);
        Vector3 vt = v - n * vn;

        float vt_len = vt.length();

        if (vt_len != 0 && dynamic_friction_magnitude != 0.f) {
            float corrected_magnitude =
                -fminf(dynamic_friction_magnitude, vt_len);

            Vector3 delta_world = vt / vt_len;

            Vector3 delta1_local = q1.inv().rotateVec(delta_world);
            Vector3 delta2_local = q2.inv().rotateVec(delta_world);

            Vector3 friction_torque_axis_local1 =
                cross(r1_local, delta1_local);
            Vector3 friction_torque_axis_local2 =
                cross(r2_local, delta2_local);

            applyVelocityUpdate(
                v1, v2,
                omega1, omega2,
                q1, q2,
                friction_torque_axis_local1, friction_torque_axis_local2,
                inv_m1, inv_m2,
                inv_I1, inv_I2,
                delta_world, corrected_magnitude);
        }
    }
}

static inline void applyRestitutionVelocityUpdate(
    Vector3 &v1, Vector3 &v2,
    Vector3 &omega1, Vector3 &omega2,
    Quat q1, Quat q2,
    float inv_m1, float inv_m2,
    Vector3 inv_I1, Vector3 inv_I2,
    Vector3 n,
    float restitution_threshold,
    Vector3 r1_worlds[4],
    Vector3 r2_worlds[4],
    Vector3 restitution_torques1_local[4],
    Vector3 restitution_torques2_local[4],
    float vn_bars[4],
    CountT num_contacts)
{
#pragma unroll
    for (CountT i = 0; i < 4; i++) {
        if (i >= num_contacts) continue;

        Vector3 r1_world = r1_worlds[i];
        Vector3 r2_world = r2_worlds[i];

        Vector3 v = computeRelativeVelocity(
            v1, v2, omega1, omega2, r1_world, r2_world);

        float vn = dot(n, v);

        float vn_bar = vn_bars[i];

        float e = 0.3f; // FIXME
        if (fabsf(vn_bar) <= restitution_threshold) {
            e = 0.f;
        }

        float restitution_magnitude = fminf(-e * vn_bar, 0) - vn;

        Vector3 restitution_torque_axis_local1 =
            restitution_torques1_local[i];
        Vector3 restitution_torque_axis_local2 =
            restitution_torques2_local[i];

        applyVelocityUpdate(
            v1, v2,
            omega1, omega2,
            q1, q2,
            restitution_torque_axis_local1, restitution_torque_axis_local2,
            inv_m1, inv_m2,
            inv_I1, inv_I2,
            n, restitution_magnitude);
    }
}

static inline void solveVelocitiesForContact(Context &ctx,
                                             ObjectManager &obj_mgr,
                                             Contact contact,
                                             float h,
                                             float restitution_threshold)
{
    Velocity *v1_out = &ctx.getDirect<Velocity>(Cols::Velocity, contact.ref);
    Velocity *v2_out = &ctx.getDirect<Velocity>(Cols::Velocity, contact.alt);

    Quat q1 = ctx.getDirect<Rotation>(Cols::Rotation, contact.ref);
    Quat q2 = ctx.getDirect<Rotation>(Cols::Rotation, contact.alt);

    PreSolvePositional presolve_pos1 = ctx.getDirect<PreSolvePositional>(
        Cols::PreSolvePositional, contact.ref);
    PreSolvePositional presolve_pos2 =  ctx.getDirect<PreSolvePositional>(
        Cols::PreSolvePositional, contact.alt);

    PreSolveVelocity presolve_vel1 =
        ctx.getDirect<PreSolveVelocity>(Cols::PreSolveVelocity, contact.ref);
    PreSolveVelocity presolve_vel2 =
        ctx.getDirect<PreSolveVelocity>(Cols::PreSolveVelocity, contact.alt);

    ObjectID obj_id1 = ctx.getDirect<ObjectID>(Cols::ObjectID, contact.ref);
    ObjectID obj_id2 = ctx.getDirect<ObjectID>(Cols::ObjectID, contact.alt);

    ResponseType resp_type1 = ctx.getDirect<ResponseType>(
        Cols::ResponseType, contact.ref);
    ResponseType resp_type2 = ctx.getDirect<ResponseType>(
        Cols::ResponseType, contact.alt);

    RigidBodyMetadata metadata1 = obj_mgr.metadata[obj_id1.idx];
    RigidBodyMetadata metadata2 = obj_mgr.metadata[obj_id2.idx];

    auto [v1, omega1] = *v1_out;
    auto [v2, omega2] = *v2_out;

    float inv_m1 = metadata1.mass.invMass;
    float inv_m2 = metadata2.mass.invMass;
    Vector3 inv_I1 = metadata1.mass.invInertiaTensor;
    Vector3 inv_I2 = metadata2.mass.invInertiaTensor;

    if (resp_type1 == ResponseType::Static) {
        inv_m1 = 0.f;
        inv_I1 = Vector3::zero();
    }

    if (resp_type2 == ResponseType::Static) {
        inv_m2 = 0.f;
        inv_I2 = Vector3::zero();
    }

    float mu_d = 0.5f * (metadata1.friction.muD + metadata2.friction.muD);

    Vector3 r1_locals[4];
    Vector3 r2_locals[4];
    Vector3 r1_worlds[4];
    Vector3 r2_worlds[4];
    Vector3 restitution_torque1_locals[4];
    Vector3 restitution_torque2_locals[4];

    float vn_bars[4];

#pragma unroll
    for (CountT i = 0; i < 4; i++) {
        if (i >= contact.numPoints) {
            continue;
        }

        auto [r1, r2] = getLocalSpaceContacts(
            presolve_pos1, presolve_pos2, contact, i);
    
        Vector3 r1_presolve = presolve_pos1.q.rotateVec(r1);
        Vector3 r2_presolve = presolve_pos2.q.rotateVec(r2);

        Vector3 v_bar = computeRelativeVelocity(
            presolve_vel1.v, presolve_vel2.v,
            presolve_vel1.omega, presolve_vel2.omega,
            r1_presolve, r2_presolve); // FIXME r1_world or presolve?

        float vn_bar = dot(contact.normal, v_bar);

        Vector3 r1_world = q1.rotateVec(r1);
        Vector3 r2_world = q2.rotateVec(r2);

        Vector3 restitution_torque_axis_local1 =
            cross(r1, q1.inv().rotateVec(contact.normal));
        Vector3 restitution_torque_axis_local2 =
            cross(r2, q2.inv().rotateVec(contact.normal));

        r1_locals[i] = r1;
        r2_locals[i] = r2;
        r1_worlds[i] = r1_world;
        r2_worlds[i] = r2_world;
        restitution_torque1_locals[i] = restitution_torque_axis_local1;
        restitution_torque2_locals[i] = restitution_torque_axis_local2;

        vn_bars[i] = vn_bar;
    }

    for (CountT restitution_iters = 0; restitution_iters < 2;
         restitution_iters++) {
        applyRestitutionVelocityUpdate(
            v1, v2,
            omega1, omega2,
            q1, q2,
            inv_m1, inv_m2,
            inv_I1, inv_I2,
            contact.normal,
            restitution_threshold,
            r1_worlds, r2_worlds,
            restitution_torque1_locals, restitution_torque2_locals,
            vn_bars, contact.numPoints);
    }

    applyFrictionVelocityUpdate(
        v1, v2,
        omega1, omega2,
        q1, q2,
        inv_m1, inv_m2,
        inv_I1, inv_I2,
        contact.normal,
        mu_d, h,
        r1_locals, r2_locals,
        r1_worlds, r2_worlds,
        contact.lambdaN, contact.numPoints);

    *v1_out = Velocity { v1, omega1 };
    *v2_out = Velocity { v2, omega2 };
}

inline void solveVelocities(Context &ctx, SolverData &solver)
{
    ObjectManager &obj_mgr = *ctx.singleton<ObjectData>().mgr;

    CountT num_contacts = solver.numContacts.load_relaxed();

    for (CountT i = 0; i < num_contacts; i++) {
        const Contact &contact = solver.contacts[i];
        solveVelocitiesForContact(ctx, obj_mgr, contact, solver.h,
                                  solver.restitutionThreshold);
    }

    solver.numContacts.store_relaxed(0);
}

}

void RigidBodyPhysicsSystem::init(Context &ctx,
                                  ObjectManager *obj_mgr,
                                  float delta_t,
                                  CountT num_substeps,
                                  math::Vector3 gravity,
                                  CountT max_dynamic_objects,
                                  CountT max_contacts_per_world,
                                  CountT max_joint_constraints_per_world)
{
    broadphase::BVH &bvh = ctx.singleton<broadphase::BVH>();

    // expansion factor is 2 * delta_t to give room
    // for acceleration within the timestep
    constexpr float max_inst_accel = 100.f;
    new (&bvh) broadphase::BVH(
        obj_mgr, max_dynamic_objects, 2.f * delta_t,
        max_inst_accel * delta_t * delta_t);

    SolverData &solver = ctx.singleton<SolverData>();
    new (&solver) SolverData(max_contacts_per_world, 
                             max_joint_constraints_per_world,
                             delta_t, num_substeps, gravity);

    ObjectData &objs = ctx.singleton<ObjectData>();
    new (&objs) ObjectData { obj_mgr };
}

void RigidBodyPhysicsSystem::reset(Context &ctx)
{
    broadphase::BVH &bvh = ctx.singleton<broadphase::BVH>();
    bvh.rebuildOnUpdate();
    bvh.clearLeaves();
}

broadphase::LeafID RigidBodyPhysicsSystem::registerEntity(Context &ctx,
                                                          Entity e,
                                                          ObjectID obj_id)
{
    return ctx.singleton<broadphase::BVH>().reserveLeaf(e, obj_id);
}

bool RigidBodyPhysicsSystem::checkEntityAABBOverlap(
    Context &ctx, math::AABB aabb, Entity e)
{
    const ObjectManager &obj_mgr = *ctx.singleton<ObjectData>().mgr;

    ObjectID e_obj_id = ctx.get<ObjectID>(e);
    Position e_pos = ctx.get<Position>(e);
    Rotation e_rot = ctx.get<Rotation>(e);
    Scale e_scale = ctx.get<Scale>(e);

    uint32_t num_prims = obj_mgr.rigidBodyPrimitiveCounts[e_obj_id.idx];
    uint32_t base_prim_offset = obj_mgr.rigidBodyPrimitiveOffsets[e_obj_id.idx];

    bool overlap = false;
    for (uint32_t prim_offset = 0; prim_offset < num_prims; prim_offset++) {
        uint32_t prim_idx = base_prim_offset + prim_offset;

        const CollisionPrimitive &prim = obj_mgr.collisionPrimitives[prim_idx];
        if (prim.type != CollisionPrimitive::Type::Hull) {
            continue;
        }

        AABB prim_aabb = obj_mgr.primitiveAABBs[prim_idx];
        AABB txfmed_aabb = prim_aabb.applyTRS(e_pos, e_rot, e_scale);

        if (!txfmed_aabb.overlaps(aabb)) {
            continue;
        }
        
        const Vector3 *vertices = prim.hull.halfEdgeMesh.vertices;
        CountT num_verts = (CountT)prim.hull.halfEdgeMesh.numVertices;

        const std::array axes {
            right,
            fwd,
            up,
        };

        std::array<float, 3> min_hull_projs {
            FLT_MAX,
            FLT_MAX,
            FLT_MAX,
        };
        std::array<float, 3> max_hull_projs {
            -FLT_MAX,
            -FLT_MAX,
            -FLT_MAX,
        };

        for (CountT vert_idx = 0; vert_idx < num_verts; vert_idx++) {
            Vector3 v =
                e_rot.rotateVec(e_scale * vertices[vert_idx]) + e_pos;

#pragma unroll
            for (CountT i = 0; i < 3; i++) {
                Vector3 axis = axes[i];

                float proj = dot(v, axis);
                if (proj < min_hull_projs[i]) {
                    min_hull_projs[i] = proj;
                }

                if (proj > max_hull_projs[i]) {
                    max_hull_projs[i] = proj;
                }
            }
        }

        bool axes_overlap = true;

#pragma unroll
        for (CountT i = 0; i < 3; i++) {
            float min_aabb_proj = aabb.pMin[i];
            float max_aabb_proj = aabb.pMax[i];

            float min_hull_proj = min_hull_projs[i];
            float max_hull_proj = max_hull_projs[i];

            bool proj_overlap = max_hull_proj > min_aabb_proj && 
                max_aabb_proj > min_hull_proj;

            if (!proj_overlap) {
                axes_overlap = false;
            }
        }

        if (axes_overlap) {
            overlap = true;
            break;
        }
    }

    return overlap;
}


void RigidBodyPhysicsSystem::registerTypes(ECSRegistry &registry)
{
    registry.registerComponent<broadphase::LeafID>();
    registry.registerSingleton<broadphase::BVH>();

    registry.registerComponent<ExternalForce>();
    registry.registerComponent<ExternalTorque>();
    registry.registerComponent<ResponseType>();
    registry.registerComponent<Velocity>();

    registry.registerComponent<solver::SubstepPrevState>();
    registry.registerComponent<solver::PreSolvePositional>();
    registry.registerComponent<solver::PreSolveVelocity>();

    registry.registerComponent<CollisionEvent>();
    registry.registerArchetype<CollisionEventTemporary>();

    registry.registerComponent<CandidateCollision>();
    registry.registerArchetype<CandidateTemporary>();

    registry.registerComponent<JointConstraint>();
    registry.registerArchetype<ConstraintData>();

    registry.registerSingleton<SolverData>();
    registry.registerSingleton<ObjectData>();

}

#ifdef MADRONA_GPU_MODE
//#define COUNT_GPU_CLOCKS
#endif

#ifdef COUNT_GPU_CLOCKS
extern "C" {
extern AtomicU64 narrowphaseAllClocks;
extern AtomicU64 narrowphaseFetchWorldClocks;
extern AtomicU64 narrowphaseSetupClocks;
extern AtomicU64 narrowphasePrepClocks;
extern AtomicU64 narrowphaseSwitchClocks;
extern AtomicU64 narrowphaseSATFaceClocks;
extern AtomicU64 narrowphaseSATEdgeClocks;
extern AtomicU64 narrowphaseSATPlaneClocks;
extern AtomicU64 narrowphaseSATContactClocks;
extern AtomicU64 narrowphaseSATPlaneContactClocks;
extern AtomicU64 narrowphaseSaveContactsClocks;
extern AtomicU64 narrowphaseTxfmHullCtrs;
}

inline void reportNarrowphaseClocks(Engine &ctx,
                                    SolverData &)
{
    if (ctx.worldID().idx != 0) {
        return;
    }

    if (threadIdx.x == 0 && ctx.worldID().idx == 0) {
        printf("[%lu, %lu, %lu, %lu, %lu, %lu, %lu, %lu, %lu, %lu, %lu, %lu]\n",
                narrowphaseAllClocks.load<sync::relaxed>(),
                narrowphaseFetchWorldClocks.load<sync::relaxed>(),
                narrowphaseSetupClocks.load<sync::relaxed>(),
                narrowphasePrepClocks.load<sync::relaxed>(),
                narrowphaseSwitchClocks.load<sync::relaxed>(),
                narrowphaseSATFaceClocks.load<sync::relaxed>(),
                narrowphaseSATEdgeClocks.load<sync::relaxed>(),
                narrowphaseSATPlaneClocks.load<sync::relaxed>(),
                narrowphaseSATContactClocks.load<sync::relaxed>(),
                narrowphaseSATPlaneContactClocks.load<sync::relaxed>(),
                narrowphaseSaveContactsClocks.load<sync::relaxed>(),
                narrowphaseTxfmHullCtrs.load<sync::relaxed>()
               );

        narrowphaseAllClocks.store<sync::relaxed>(0);
        narrowphaseFetchWorldClocks.store<sync::relaxed>(0);
        narrowphaseSetupClocks.store<sync::relaxed>(0),
        narrowphasePrepClocks.store<sync::relaxed>(0);
        narrowphaseSwitchClocks.store<sync::relaxed>(0);
        narrowphaseSATFaceClocks.store<sync::relaxed>(0);
        narrowphaseSATEdgeClocks.store<sync::relaxed>(0);
        narrowphaseSATPlaneClocks.store<sync::relaxed>(0);
        narrowphaseSATContactClocks.store<sync::relaxed>(0);
        narrowphaseSATPlaneContactClocks.store<sync::relaxed>(0);
        narrowphaseSaveContactsClocks.store<sync::relaxed>(0);
        narrowphaseTxfmHullCtrs.store<sync::relaxed>(0);
    }
}
#endif

TaskGraphNodeID RigidBodyPhysicsSystem::setupBroadphaseTasks(
    TaskGraphBuilder &builder,
    Span<const TaskGraphNodeID> deps)
{
    return broadphase::setupBVHTasks(builder, deps);
}

TaskGraphNodeID RigidBodyPhysicsSystem::setupSubstepTasks(
    TaskGraphBuilder &builder,
    Span<const TaskGraphNodeID> deps,
    CountT num_substeps)
{
    auto broadphase_pre =
        broadphase::setupPreIntegrationTasks(builder, deps);

    auto cur_node = broadphase_pre;
    for (CountT i = 0; i < num_substeps; i++) {
        // FIXME, Should only need to collect these once per frame
        auto collect_constraints = builder.addToGraph<ParallelForNode<Context,
            collectConstraintsSystem, JointConstraint>>({cur_node});

        auto rgb_update = builder.addToGraph<ParallelForNode<Context,
            solver::substepRigidBodies, Position, Rotation, Velocity, ObjectID,
            ResponseType, ExternalForce, ExternalTorque,
            solver::SubstepPrevState, solver::PreSolvePositional,
            solver::PreSolveVelocity>>({cur_node});

        auto run_narrowphase = narrowphase::setupTasks(builder, {rgb_update});

        auto solve_pos = builder.addToGraph<ParallelForNode<Context,
            solver::solvePositions, SolverData>>(
                {run_narrowphase, collect_constraints});

        auto vel_set = builder.addToGraph<ParallelForNode<Context,
            solver::setVelocities, Position, Rotation,
            solver::SubstepPrevState, Velocity>>({solve_pos});

        auto solve_vel = builder.addToGraph<ParallelForNode<Context,
            solver::solveVelocities, SolverData>>({vel_set});

        cur_node = builder.addToGraph<ResetTmpAllocNode>({solve_vel});

#if 0
        cur_node = builder.addToGraph<ParallelForNode<Context,
            solver::checkSubstep, Entity, Position, Rotation, Velocity,
            ObjectID, ResponseType, ExternalForce, ExternalTorque,
            solver::SubstepPrevState, solver::PreSolvePositional,
            solver::PreSolveVelocity>>({cur_node});
#endif
    }

    auto clear_candidates = builder.addToGraph<
        ClearTmpNode<CandidateTemporary>>({cur_node});

    auto broadphase_post =
        broadphase::setupPostIntegrationTasks(builder, {clear_candidates});

    auto physics_done = broadphase_post;

#ifdef COUNT_GPU_CLOCKS
    physics_done = builder.addToGraph<ParallelForNode<Context,
        reportNarrowphaseClocks, SolverData>>({physics_done});
#endif

    return physics_done;
}

TaskGraphNodeID RigidBodyPhysicsSystem::setupCleanupTasks(
    TaskGraphBuilder &builder, Span<const TaskGraphNodeID> deps)
{
    return builder.addToGraph<ClearTmpNode<CollisionEventTemporary>>(deps);
}

}
