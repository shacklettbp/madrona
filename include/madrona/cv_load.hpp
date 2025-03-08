#pragma once

#include <madrona/cv_physics.hpp>

namespace madrona::phys::cv {

struct JointConnection {
    // These are body indices
    uint32_t parentIdx;
    uint32_t childIdx;

    // This determines which Joint struct to use
    DofType type;

    union {
        JointHinge hinge;
        JointBall ball;
        JointSlider slider;
        JointFixed fixed;
        // ...
    };
};

struct JointLimit {
    uint32_t bodyIdx;

    DofType type;

    union {
        HingeLimit hinge;
        SliderLimit slider;
    };
};

// Either link or joint name hashes
struct NameHash {
    uint32_t bodyIdx;
    uint32_t hash;
};

// For loading pre-configured models
struct ModelConfig {
    // Assume that the first one is the root
    uint32_t numBodies;
    uint32_t bodiesOffset;

    uint32_t numConnections;
    uint32_t connectionsOffset;

    uint32_t numColliders;
    uint32_t collidersOffset;

    uint32_t numVisuals;
    uint32_t visualsOffset;

    uint32_t numCollisionDisables;
    uint32_t collisionDisableOffset;

    uint32_t numJointLimits;
    uint32_t jointLimitOffset;

    uint32_t numHashes;
    uint32_t hashOffset;
};

struct CollisionDisable {
    uint32_t aBody;
    uint32_t bBody;
};

// This is the data for all models in that could possibly be loaded.
struct ModelData {
    uint32_t numBodies;
    BodyDesc *bodies;

    uint32_t numConnections;
    JointConnection *connections;

    uint32_t numColliders;
    CollisionDesc *colliders;

    uint32_t numVisuals;
    VisualDesc *visuals;

    uint32_t numCollisionDisables;
    CollisionDisable *collisionDisables;

    uint32_t numJointLimits;
    JointLimit *jointLimits;

    uint32_t numNameHashes;
    NameHash *nameHashes;
};

// This returns the body group entity
Entity loadModel(Context &ctx,
                 ModelConfig cfg,
                 ModelData model_data,
                 math::Vector3 initial_pos,
                 math::Quat initial_rot,
                 float global_scale);

// Fills in things like mass and inertia
BodyDesc makeCapsuleBodyDesc(
        DofType type,
        math::Vector3 initial_pos,
        math::Quat initial_rot,
        ResponseType response_type,
        uint32_t num_limits,
        float mu_s,
        float mass, // Total mass
        float radius,
        float cylinder_height);

// Because of the weird scaling properties of the capsule, we handle the
// capsule differently for rendering and physics.
void attachCapsuleObjects(
        Context &ctx,
        Entity body_grp,
        Entity body,
        float radius,
        float cylinder_height,
        uint32_t capsule_collider_obj_idx,
        uint32_t sphere_render_obj_idx,
        uint32_t cylinder_render_obj_idx);

}
