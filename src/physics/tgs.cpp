#include <madrona/physics.hpp>

#include "physics_impl.hpp"
#include "tgs.hpp"

// This implementation is inspired by the TGS-Soft solver from Solver2D:
// https://github.com/erincatto/solver2d/src/solve_tgs_soft.c
// Solver2D is MIT licensed, Copyright 2024 Erin Catto

namespace madrona::phys::tgs {

using namespace base;
using namespace math;

void registerTypes(ECSRegistry &registry)
{
    (void)registry;
}

static inline void solveJoints(Context &ctx,
                               SolverData &solver,
                               bool use_bias)
{
    (void)ctx;
    (void)solver;
    (void)use_bias;
}

static inline void solveContacts(Context &ctx,
                                 SolverData &solver,
                                 bool use_bias)
{
    (void)ctx;
    (void)solver;
    (void)use_bias;
}

inline void prepareContacts(Context &ctx,
                            ContactConstraint constraint)
{
    (void)ctx;
    (void)constraint;
}

inline void prepareJoints(Context &ctx,
                          JointConstraint constraint)
{
    (void)ctx;
    (void)constraint;
}

inline void integrateVelocities(Context &ctx,
                                Rotation q,
                                ResponseType response_type,
                                ExternalForce ext_force,
                                ExternalTorque ext_torque,
                                ObjectID obj_id,
                                Velocity &vel)
{
    if (response_type == ResponseType::Static) {
        return;
    }

    Vector3 v = vel.linear;
    Vector3 omega = vel.angular;

    const auto &solver = ctx.singleton<SolverData>();
    const ObjectManager &obj_mgr = *ctx.singleton<ObjectData>().mgr;
    const RigidBodyMetadata &metadata = obj_mgr.metadata[obj_id.idx];

    float inv_m = metadata.mass.invMass;
    Diag3x3 inv_I = Diag3x3::fromVec(metadata.mass.invInertiaTensor);

    float h = solver.h;

    if (response_type == ResponseType::Dynamic) {
        v += h * solver.g;
    }

    v += h * inv_m * ext_force;

    Diag3x3 I = {
        (inv_I.d0 == 0) ? 0.0f : 1.0f / inv_I.d0,
        (inv_I.d1 == 0) ? 0.0f : 1.0f / inv_I.d1,
        (inv_I.d2 == 0) ? 0.0f : 1.0f / inv_I.d2,
    };

    Quat to_local = q.inv();

    Vector3 tau_ext_local = to_local.rotateVec(ext_torque);
    Vector3 omega_local = to_local.rotateVec(omega);

    // Integrate omega in local space
    omega_local +=
        h * inv_I * (tau_ext_local - (cross(omega_local, I * omega)));

    omega = q.rotateVec(omega_local);

    // FIXME damping

    vel.linear = v;
    vel.angular = omega;
}

inline void warmStartContacts(Context &ctx,
                              ContactConstraint constraint)
{
    (void)ctx;
    (void)constraint;
}

inline void warmStartJoints(Context &ctx,
                            JointConstraint constraint)
{
    (void)ctx;
    (void)constraint;
}

inline void solveJointsBiased(Context &ctx,
                              SolverData &solver)
{
    solveJoints(ctx, solver, true);
}

inline void solveContactsBiased(Context &ctx,
                                SolverData &solver)
{
    solveContacts(ctx, solver, true);
}

inline void integratePositions(Context &ctx,
                               Position &pos,
                               Rotation &rot,
                               Velocity vel)
{
    Vector3 x = pos;
    Quat q = rot;

    Vector3 v = vel.linear;
    Vector3 omega = vel.angular;

    const auto &solver = ctx.singleton<SolverData>();
    float h = solver.h;

    x += h * v;

    Quat apply_omega = Quat::fromAngularVec(0.5f * h * omega);

    q += apply_omega * q;
    q = q.normalize();

    pos = x;
    rot = q;
}

inline void solveJointsUnbiased(Context &ctx,
                                SolverData &solver)
{
    solveJoints(ctx, solver, false);
}

inline void solveContactsUnbiased(Context &ctx,
                                  SolverData &solver)
{
    solveContacts(ctx, solver, false);
}

TaskGraphNodeID setupTGSSolverTasks(
    TaskGraphBuilder &builder,
    TaskGraphNodeID broadphase,
    CountT num_substeps)
{
    auto cur_node = narrowphase::setupTasks(builder, {broadphase});

    cur_node = builder.addToGraph<ParallelForNode<Context,
        prepareContacts,
            ContactConstraint
        >>({cur_node});

    cur_node = builder.addToGraph<ParallelForNode<Context,
        prepareJoints,
            JointConstraint
        >>({cur_node});
            
    for (CountT i = 0; i < num_substeps; i++) {
        cur_node = builder.addToGraph<ParallelForNode<Context,
            integrateVelocities,
                Rotation,
                ResponseType,
                ExternalForce,
                ExternalTorque,
                ObjectID,
                Velocity
            >>({cur_node});

        cur_node = builder.addToGraph<ParallelForNode<Context,
            warmStartContacts,
                ContactConstraint
            >>({cur_node});

        cur_node = builder.addToGraph<ParallelForNode<Context,
            warmStartContacts,
                ContactConstraint
            >>({cur_node});
         
        cur_node = builder.addToGraph<ParallelForNode<Context,
            solveJointsBiased,
                SolverData
            >>({cur_node});

        cur_node = builder.addToGraph<ParallelForNode<Context,
            solveContactsBiased,
                SolverData
            >>({cur_node});

        cur_node = builder.addToGraph<ParallelForNode<Context,
            integratePositions,
                Position,
                Rotation,
                Velocity
            >>({cur_node});

        cur_node = builder.addToGraph<ParallelForNode<Context,
            solveJointsUnbiased,
                SolverData
            >>({cur_node});

        cur_node = builder.addToGraph<ParallelForNode<Context,
            solveContactsUnbiased,
                SolverData
            >>({cur_node});
    }

    return cur_node;
}

}
