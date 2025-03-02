#include "cv.hpp"
#include <madrona/cv_load.hpp>
#include <madrona/cv_physics.hpp>

namespace madrona::phys::cv {

using namespace math;

Entity loadModel(Context &ctx,
                 ModelConfig cfg,
                 ModelData model_data,
                 Vector3 initial_pos,
                 Quat initial_rot,
                 float global_scale)
{
    Entity grp = makeBodyGroup(ctx, cfg.numBodies, global_scale);
    Entity *bodies_tmp =
        (Entity *)ctx.tmpAlloc(sizeof(Entity) * cfg.numBodies);

    ctx.get<BodyGroupProperties>(grp).numHashes = cfg.numHashes;

    { // Make the root
        BodyDesc desc = model_data.bodies[cfg.bodiesOffset];

        desc.initialPos = initial_pos;
        desc.initialRot = initial_rot;

        bodies_tmp[0] = makeBody(
                ctx,
                grp,
                desc);
    }

    // Create the bodies (links)
    for (uint32_t i = 1; i < cfg.numBodies; ++i) {
        bodies_tmp[i] = makeBody(
                ctx,
                grp,
                model_data.bodies[cfg.bodiesOffset + i]);
    }

    { // Copy name hashes
        BodyGroupMemory m = ctx.get<BodyGroupMemory>(grp);
        BodyGroupProperties p = ctx.get<BodyGroupProperties>(grp);
        BodyNameHash *hashes = m.nameHashes(p);

        for (uint32_t i = 0; i < cfg.numHashes; ++i) {
            NameHash src = model_data.nameHashes[cfg.hashOffset + i];

            hashes[i] = {
                .bodyIdx = src.bodyIdx,
                .hash = src.hash
            };
        }
    }

    // Attach the colliders
    for (uint32_t i = 0; i < cfg.numColliders; ++i) {
        CollisionDesc desc = model_data.colliders[cfg.collidersOffset + i];

        attachCollision(
                ctx,
                grp,
                bodies_tmp[desc.linkIdx],
                desc.subIndex,
                desc);
    }

    // Attach the visuals
    for (uint32_t i = 0; i < cfg.numVisuals; ++i) {
        VisualDesc desc = model_data.visuals[cfg.visualsOffset + i];

        attachVisual(
                ctx,
                grp,
                bodies_tmp[desc.linkIdx],
                desc.subIndex,
                desc);
    }

    { // Create the hierarchy
        setRoot(ctx, grp, bodies_tmp[0]);

        for (uint32_t i = 0; i < cfg.numConnections; ++i) {
            JointConnection conn =
                model_data.connections[cfg.connectionsOffset + i];

            switch (conn.type) {
            case DofType::Hinge: {
                joinBodies(
                        ctx,
                        grp,
                        bodies_tmp[conn.parentIdx],
                        bodies_tmp[conn.childIdx],
                        conn.hinge);
            } break;

            case DofType::Ball: {
                joinBodies(
                        ctx,
                        grp,
                        bodies_tmp[conn.parentIdx],
                        bodies_tmp[conn.childIdx],
                        conn.ball);
            } break;

            case DofType::Slider: {
                joinBodies(
                        ctx,
                        grp,
                        bodies_tmp[conn.parentIdx],
                        bodies_tmp[conn.childIdx],
                        conn.slider);
            } break;

            case DofType::FixedBody: {
                joinBodies(
                        ctx,
                        grp,
                        bodies_tmp[conn.parentIdx],
                        bodies_tmp[conn.childIdx],
                        conn.fixed);
            } break;

            default: {
                // Not supported yet
                assert(false);
            } break;

            }
        }
    }

    { // Disable collisions:
        // Make sure to disable collisions between colliders in same joints
        BodyGroupMemory &m = ctx.get<BodyGroupMemory>(grp);
        BodyGroupProperties &p = ctx.get<BodyGroupProperties>(grp);

        BodyObjectData *col_data = m.objectData(p);

        for (uint32_t i = 1; i < cfg.numBodies; ++i) {
            Entity body = m.entities(p)[i];

            DofObjectProxies proxies = ctx.get<DofObjectProxies>(body);

            for (uint32_t b0 = 0; b0 < proxies.numColliders; ++b0) {
                for (uint32_t b1 = b0+1; b1 < proxies.numColliders; ++b1) {
                    Entity a_rb = col_data[b0 + proxies.colliderOffset].proxy;
                    Entity b_rb = col_data[b1 + proxies.colliderOffset].proxy;

                    PhysicsSystem::disableCollision(ctx, a_rb, b_rb);
                }
            }
        }

        for (uint32_t i = 0; i < cfg.numCollisionDisables; ++i) {
            CollisionDisable disable =
                model_data.collisionDisables[i + cfg.collisionDisableOffset];

            disableJointCollisions(
                    ctx, 
                    grp,
                    bodies_tmp[disable.aBody],
                    bodies_tmp[disable.bBody]);
        }
    }

    { // Attach joint limits
        for (uint32_t i = 0; i < cfg.numJointLimits; ++i) {
            JointLimit limit = model_data.jointLimits[i + cfg.jointLimitOffset];

            switch (limit.type) {
            case DofType::Hinge: {
                attachLimit(
                        ctx, 
                        grp,
                        bodies_tmp[limit.bodyIdx],
                        limit.hinge);
            } break;

            case DofType::Slider: {
                attachLimit(
                        ctx, 
                        grp,
                        bodies_tmp[limit.bodyIdx],
                        limit.slider);
            } break;

            default: {
                assert(false);
            } break;
            }
        }
    }

    return grp;
}

}
