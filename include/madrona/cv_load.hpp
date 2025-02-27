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
};

// This returns the body group entity
Entity loadModel(Context &ctx,
                 ModelConfig cfg,
                 ModelData model_data,
                 math::Vector3 initial_pos,
                 math::Quat initial_rot,
                 float global_scale);

}
