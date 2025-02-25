#include "cv.hpp"
#include <madrona/state.hpp>
#include <madrona/cv_physics.hpp>
#include <madrona/taskgraph.hpp>

using namespace madrona::math;
using namespace madrona::base;

namespace madrona::phys::cv {

namespace tasks {

void forwardKinematics(Context &,
                       BodyGroupMemory m,
                       BodyGroupProperties p)
{
    float *all_q = m.q(p);
    BodyTransform *all_transforms = m.bodyTransforms(p);
    BodyHierarchy *all_hiers = m.hierarchies(p);
    BodyPhi *all_phi = m.bodyPhi(p);
    BodyOffsets *all_offsets = m.offsets(p);

    { // Set the parent's state (we require that the root is fixed or free body
        Vector3 com = { all_q[0], all_q[1], all_q[2] };

        all_transforms[0] = {
            .com = com,
            .composedRot = { all_q[3], all_q[4], all_q[5], all_q[6] },
        };

        // omega remains unchanged, and v only depends on the COM position
        all_phi[0].phi[0] = com[0];
        all_phi[0].phi[1] = com[1];
        all_phi[0].phi[2] = com[2];
    }

    // Forward pass from parent to children
    for (int i = 1; i < (int)p.numBodies; ++i) {
        BodyOffsets offsets = all_offsets[i];

        const float *q = all_q + all_offsets[i].posOffset;

        BodyTransform *curr_transform = all_transforms + i;
        BodyPhi *curr_phi = all_phi + i;

        BodyHierarchy hier = all_hiers[i];
        BodyTransform parent_transform = all_transforms[offsets.parent];

        float s = p.globalScale;

        // We can calculate our stuff.
        switch (offsets.dofType) {
        case DofType::Hinge: {
            // Find the hinge axis orientation in world space
            Vector3 rotated_hinge_axis =
                parent_transform.composedRot.rotateVec(
                        hier.parentToChildRot.rotateVec(hier.axis));

            // Calculate the composed rotation applied to the child entity.
            curr_transform->composedRot = parent_transform.composedRot *
                                    hier.parentToChildRot *
                                    Quat::angleAxis(q[0], hier.axis);

            // Calculate the composed COM position of the child
            //  (parent COM + R_{parent} * (rel_pos_parent + R_{hinge} * rel_pos_local))
            curr_transform->com = parent_transform.com +
                s * parent_transform.composedRot.rotateVec(
                        hier.relPositionParent +
                        hier.parentToChildRot.rotateVec(
                            Quat::angleAxis(q[0], hier.axis).
                                rotateVec(hier.relPositionLocal))
                );

            // All we are getting here is the position of the hinge point
            // which is relative to the parent's COM.
            Vector3 anchor_pos = parent_transform.com +
                s * parent_transform.composedRot.rotateVec(
                        hier.relPositionParent);

            // Phi only depends on the hinge axis and the hinge point
            curr_phi->phi[0] = rotated_hinge_axis[0];
            curr_phi->phi[1] = rotated_hinge_axis[1];
            curr_phi->phi[2] = rotated_hinge_axis[2];
            curr_phi->phi[3] = anchor_pos[0];
            curr_phi->phi[4] = anchor_pos[1];
            curr_phi->phi[5] = anchor_pos[2];
        } break;

        case DofType::Slider: {
            Vector3 rotated_axis =
                parent_transform.composedRot.rotateVec(
                        hier.parentToChildRot.rotateVec(hier.axis));

            // The composed rotation for this body is the same as the parent's
            curr_transform->composedRot = parent_transform.composedRot *
                                    hier.parentToChildRot;

            curr_transform->com = parent_transform.com +
                s * parent_transform.composedRot.rotateVec(
                        hier.relPositionParent +
                        hier.parentToChildRot.rotateVec(
                            hier.relPositionLocal +
                            q[0] * hier.axis)
                );

            // This is the same as the comPos I guess?
            Vector3 axis = rotated_axis.normalize();

            curr_phi->phi[0] = axis[0];
            curr_phi->phi[1] = axis[1];
            curr_phi->phi[2] = axis[2];
        } break;

        case DofType::Ball: {
            Quat joint_rot = Quat{
                q[0], q[1], q[2], q[3]
            };

            // Calculate the composed rotation applied to the child entity.
            curr_transform->composedRot = parent_transform.composedRot *
                                    hier.parentToChildRot *
                                    joint_rot;

            // Calculate the composed COM position of the child
            //  (parent COM + R_{parent} * (rel_pos_parent + R_{ball} * rel_pos_local))
            curr_transform->com = parent_transform.com +
                s * parent_transform.composedRot.rotateVec(
                        hier.relPositionParent +
                        hier.parentToChildRot.rotateVec(
                            joint_rot.rotateVec(hier.relPositionLocal))
                );

            // All we are getting here is the position of the ball point
            // which is relative to the parent's COM.
            Vector3 anchor_pos = parent_transform.com +
                s * parent_transform.composedRot.rotateVec(
                        hier.relPositionParent);

            // Phi only depends on the hinge point and parent rotation
            curr_phi->phi[0] = anchor_pos[0];
            curr_phi->phi[1] = anchor_pos[1];
            curr_phi->phi[2] = anchor_pos[2];
            curr_phi->phi[3] = parent_transform.composedRot.w;
            curr_phi->phi[4] = parent_transform.composedRot.x;
            curr_phi->phi[5] = parent_transform.composedRot.y;
            curr_phi->phi[6] = parent_transform.composedRot.z;
        } break;

        case DofType::FixedBody: {
            curr_transform->composedRot = parent_transform.composedRot;

            // This is the origin of the body
            curr_transform->com =
                parent_transform.com +
                s * parent_transform.composedRot.rotateVec(
                        hier.relPositionParent +
                        hier.parentToChildRot.rotateVec(
                            hier.relPositionLocal)
                );

            // omega remains unchanged, and v only depends on the COM position
            curr_phi->phi[0] = curr_transform->com[0];
            curr_phi->phi[1] = curr_transform->com[1];
            curr_phi->phi[2] = curr_transform->com[2];
        } break;

        default: {
            // Only hinges have parents
            assert(false);
        } break;
        }
    }
}
}

}
