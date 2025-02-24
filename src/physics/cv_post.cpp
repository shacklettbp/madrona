#include "cv.hpp"
#include "physics_impl.hpp"
#include <madrona/cv_physics.hpp>

namespace madrona::phys::cv {

using namespace math;
using namespace base;
    
namespace tasks {
inline void integrationStep(Context &ctx,
                            DofObjectGroup grp_info)
{
    float h = ctx.singleton<PhysicsSystemState>().h;

    BodyGroupMemory m = ctx.get<BodyGroupMemory>(grp_info.bodyGroup);
    BodyGroupProperties p = ctx.get<BodyGroupProperties>(grp_info.bodyGroup);

    BodyOffsets offset = m.offsets(p)[grp_info.idx];

    float *q = m.q(p) + offset.posOffset;
    float *qv = m.qv(p) + offset.velOffset;
    float *dqv = m.dqv(p) + offset.velOffset;

    if (offset.dofType == DofType::FreeBody) {
        // Symplectic Euler
        for (int i = 0; i < 6; ++i) {
            qv[i] += h * dqv[i];
        }
        for (int i = 0; i < 3; ++i) {
            q[i] += h * qv[i];
        }

        // From angular velocity to quaternion [Q_w, Q_x, Q_y, Q_z]
        Vector3 omega = { qv[3], qv[4], qv[5] };
        Quat rot_quat = { q[3], q[4], q[5], q[6] };
        Quat new_rot = {rot_quat.w, rot_quat.x, rot_quat.y, rot_quat.z};
        new_rot.w += 0.5f * h * (-rot_quat.x * omega.x - 
                                 rot_quat.y * omega.y - 
                                 rot_quat.z * omega.z);

        new_rot.x += 0.5f * h * (rot_quat.w * omega.x + 
                                 rot_quat.z * omega.y -
                                 rot_quat.y * omega.z);
        
        new_rot.y += 0.5f * h * (-rot_quat.z * omega.x + 
                                 rot_quat.w * omega.y +
                                 rot_quat.x * omega.z);

        new_rot.z += 0.5f * h * (rot_quat.y * omega.x - 
                                 rot_quat.x * omega.y +
                                 rot_quat.w * omega.z);

        new_rot = new_rot.normalize();

        q[3] = new_rot.w;
        q[4] = new_rot.x;
        q[5] = new_rot.y;
        q[6] = new_rot.z;
    }
    else if (offset.dofType == DofType::Slider) {
        qv[0] += h * dqv[0];
        q[0] += h * qv[0];
    }
    else if (offset.dofType == DofType::Hinge) {
        qv[0] += h * dqv[0];
        q[0] += h * qv[0];
    }
    else if (offset.dofType == DofType::FixedBody) {
        // Do nothing
    }
    else if (offset.dofType == DofType::Ball) {
        // Symplectic Euler
        for (int i = 0; i < 3; ++i) {
            qv[i] += h * dqv[i];
        }

        // From angular velocity to quaternion [Q_w, Q_x, Q_y, Q_z]
        Vector3 omega = { qv[0], qv[1], qv[2] };
        Quat rot_quat = { q[0], q[1], q[2], q[3] };
        Quat new_rot = { rot_quat.w, rot_quat.x, rot_quat.y, rot_quat.z };
        new_rot.w += 0.5f * h * (-rot_quat.x * omega.x -
                                 rot_quat.y * omega.y -
                                 rot_quat.z * omega.z);

        new_rot.x += 0.5f * h * (rot_quat.w * omega.x +
                                 rot_quat.z * omega.y -
                                 rot_quat.y * omega.z);

        new_rot.y += 0.5f * h * (-rot_quat.z * omega.x + 
                                 rot_quat.w * omega.y +
                                 rot_quat.x * omega.z);

        new_rot.z += 0.5f * h * (rot_quat.y * omega.x -
                                 rot_quat.x * omega.y +
                                 rot_quat.w * omega.z);

        new_rot = new_rot.normalize();

        q[0] = new_rot.w;
        q[1] = new_rot.x;
        q[2] = new_rot.y;
        q[3] = new_rot.z;
    }
    else {
        MADRONA_UNREACHABLE();
    }
}

inline void convertPostSolve(
        Context &ctx,
        Position &position,
        Rotation &rotation,
        Scale &scale,
        LinkParentDofObject &link)
{
    BodyGroupMemory m = ctx.get<BodyGroupMemory>(link.bodyGroup);
    BodyGroupProperties p = ctx.get<BodyGroupProperties>(link.bodyGroup);

    BodyOffsets offset = m.offsets(p)[link.bodyIdx];

    BodyTransform transforms = m.bodyTransforms(p)[link.bodyIdx];
    BodyObjectData obj_data = m.objectData(p)[link.objDataIdx];

    scale = obj_data.scale * p.globalScale;

    if (offset.dofType == DofType::FreeBody) {
        position = transforms.com +
                   p.globalScale * 
                        transforms.composedRot.rotateVec(obj_data.offset);
        rotation = transforms.composedRot * obj_data.rotation;
    }
    else if (offset.dofType == DofType::Hinge) {
        position = transforms.com +
                   p.globalScale * 
                        transforms.composedRot.rotateVec(obj_data.offset);
        rotation = transforms.composedRot * obj_data.rotation;
    }
    else if (offset.dofType == DofType::Slider) {
        position = transforms.com +
                   p.globalScale * 
                        transforms.composedRot.rotateVec(obj_data.offset);
        rotation = transforms.composedRot *
                   obj_data.rotation;
    }
    else if (offset.dofType == DofType::FixedBody) {
        // For this, we need to look at the first parent who isn't
        // fixed body and apply its transform.

        position = transforms.com +
                   p.globalScale * 
                        transforms.composedRot.rotateVec(obj_data.offset);
        rotation = transforms.composedRot * obj_data.rotation;

        // Do nothing
    }
    else if (offset.dofType == DofType::Ball) {
        position = transforms.com +
                   p.globalScale * 
                        transforms.composedRot.rotateVec(obj_data.offset);
        rotation = transforms.composedRot *
                   obj_data.rotation;
    }
    else {
        MADRONA_UNREACHABLE();
    }
}
}

TaskGraphNodeID setupPostTasks(TaskGraphBuilder &builder,
                               TaskGraphNodeID solve)
{
    auto cur_node = builder.addToGraph<ParallelForNode<Context,
         tasks::integrationStep,
            DofObjectGroup
        >>({solve});

    cur_node = builder.addToGraph<ParallelForNode<Context,
         tasks::forwardKinematics,
            BodyGroupMemory,
            BodyGroupProperties
        >>({cur_node});

    cur_node =
        builder.addToGraph<ParallelForNode<Context, tasks::convertPostSolve,
            Position,
            Rotation,
            Scale,
            LinkParentDofObject
        >>({cur_node});

    return cur_node;
}
    
}
