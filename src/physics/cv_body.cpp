#include "cv.hpp"
#include <madrona/cv_physics.hpp>

namespace madrona::phys::cv {

using namespace math;
using namespace base;
    
Entity makeBodyGroup(Context &ctx,
                     uint32_t num_bodies,
                     float global_scale)
{
    Entity g = ctx.makeEntity<BodyGroupArchetype>();

    { // Initialize properties
        BodyGroupProperties &p = ctx.get<BodyGroupProperties>(g);
        p.globalScale = global_scale;

        // At initialization, set everything we don't know to 0
        p.qDim = 0;
        p.qvDim = 0;
        p.numBodies = num_bodies;
        p.numEq = 0;
        p.numObjData = 0;
        p.tmp.bodyCounter = 0;
    }

    { // Initialize memory
        BodyGroupMemory &m = ctx.get<BodyGroupMemory>(g);
        m.tmpPtr = ctx.tmpAlloc(sizeof(BodyDesc) * num_bodies +
                                sizeof(Entity) * num_bodies);
    }

    Loc init = ctx.makeTemporary<InitBodyGroupArchetype>();
    ctx.get<InitBodyGroup>(init).bodyGroup = g;

    return g;
}

static void initBodyGroupMemory(
        Context &ctx,
        BodyGroupProperties &p,
        BodyGroupMemory &m)
{
    BodyDesc *body_descs = (BodyDesc *)m.tmpPtr;

    for (uint32_t i = 0; i < p.numBodies; ++i) {
        BodyDesc &bd = body_descs[i];
        p.qDim += BodyOffsets::getDofTypeDim(bd.type, true);
        p.qvDim += BodyOffsets::getDofTypeDim(bd.type);
        p.numEq += bd.numLimits;
        p.numObjData += bd.numCollisionObjs + bd.numVisualObjs;
    }

    { // Allocate frame persistent memory
        uint32_t num_bytes = BodyGroupMemory::qVectorsNumBytes(p);
        uint32_t num_elems = (num_bytes + sizeof(MRElement128b) - 1) /
            sizeof(MRElement128b);
        m.qVectors = ctx.allocMemoryRange<MRElement128b>(num_elems);
        m.qVectorsPtr = ctx.memoryRangePointer<MRElement128b>(m.qVectors);

        // Set everything to zero initially
        memset(m.qVectorsPtr, 0, num_bytes);
    }

    { // Use all the BodyDescs to initialize q vectors
        // These are all the attributes that we can initialize at this time
        BodyOffsets *offsets = m.offsets(p);
        float *q = m.q(p);
        float *mus = m.mus(p);
        BodyInertial *inertials = m.inertials(p);
        Entity *entities = m.entities(p);
        BodyNameHash *hashes = m.nameHashes(p);

        uint8_t *max_ptr = (uint8_t *)m.qVectorsPtr +
                           BodyGroupMemory::qVectorsNumBytes(p);

        uint32_t q_dof_offset = 0;
        uint32_t fixed_body_q_offset = 0;
        uint32_t qv_dof_offset = 0;
        uint32_t eq_offset = 0;

        uint32_t num_real_q = 0;

        for (uint32_t i = 0; i < p.numBodies; ++i) {
            BodyDesc &bd = body_descs[i];

            switch (bd.type) {
            case DofType::FreeBody: {
                num_real_q += 7;
            } break;

            case DofType::Hinge: {
                num_real_q += 1;
            } break;

            case DofType::Slider: {
                num_real_q += 1;
            } break;

            case DofType::Ball: {
                num_real_q += 4;
            } break;

            case DofType::FixedBody: {
                // Do nothing
            } break;

            case DofType::None: {
                assert(false);
            } break;
            }
        }

        p.numFixedQ = p.qDim - num_real_q;

        for (uint32_t i = 0; i < p.numBodies; ++i) {
            BodyDesc &bd = body_descs[i];

            uint32_t curr_q_offset = q_dof_offset;
            if (bd.type == DofType::FixedBody) {
                curr_q_offset = num_real_q + fixed_body_q_offset;
                fixed_body_q_offset += 7;
            } else {
                q_dof_offset += BodyOffsets::getDofTypeDim(bd.type, true);
            }

            // First thing to fill in is the offsets
            offsets[i] = BodyOffsets {
                .posOffset = (uint8_t)curr_q_offset,
                .velOffset = (uint8_t)qv_dof_offset,
                .parent = 0xFF,
                .dofType = body_descs[i].type,
                .numDofs = (uint8_t)BodyOffsets::getDofTypeDim(body_descs[i].type),
                .eqOffset = (uint8_t)eq_offset,
                .numEqs = (uint8_t)bd.numLimits,
            };

            assert((uint8_t *)(offsets + i) < max_ptr);

            float *curr_q = q + curr_q_offset;
            assert((uint8_t *)curr_q < max_ptr);

            qv_dof_offset += BodyOffsets::getDofTypeDim(bd.type);
            eq_offset += bd.numLimits;

            // Then, fill in initial values for q, etc...
            switch(bd.type) {
            case DofType::FreeBody: {
                curr_q[0] = bd.initialPos.x;
                curr_q[1] = bd.initialPos.y;
                curr_q[2] = bd.initialPos.z;
                curr_q[3] = bd.initialRot.w;
                curr_q[4] = bd.initialRot.x;
                curr_q[5] = bd.initialRot.y;
                curr_q[6] = bd.initialRot.z;
            } break;

            case DofType::Hinge: {
                curr_q[0] = 0.f;
            } break;

            case DofType::Slider: {
                curr_q[0] = 0.f;
            } break;

            case DofType::Ball: {
                curr_q[0] = bd.initialRot.w;
                curr_q[1] = bd.initialRot.x;
                curr_q[2] = bd.initialRot.y;
                curr_q[3] = bd.initialRot.z;
            } break;

            case DofType::FixedBody: {
                curr_q[0] = bd.initialPos.x;
                curr_q[1] = bd.initialPos.y;
                curr_q[2] = bd.initialPos.z;
                curr_q[3] = bd.initialRot.w;
                curr_q[4] = bd.initialRot.x;
                curr_q[5] = bd.initialRot.y;
                curr_q[6] = bd.initialRot.z;
            } break;

            case DofType::None: {
                assert(false);
            } break;
            }

            // Fill in mus
            mus[i] = bd.muS;

            assert((uint8_t *)(mus + i) < max_ptr);

            // Fill in inertia
            inertials[i] = BodyInertial {
                .mass = bd.mass,
                .inertia = bd.inertia
            };

            assert((uint8_t *)(inertials + i) < max_ptr);

            entities[i] = ((Entity *)((BodyDesc *)m.tmpPtr + p.numBodies))[i];
            assert((uint8_t *)(entities + i) < max_ptr);
        }
    }

    { // Allocate frame volatile memory
        uint32_t num_bytes = BodyGroupMemory::tmpNumBytes(p);
        uint32_t num_elems = (num_bytes + sizeof(MRElement128b) - 1) /
            sizeof(MRElement128b);
        m.tmp = ctx.allocMemoryRange<MRElement128b>(num_elems);
        m.tmpPtr = ctx.memoryRangePointer<MRElement128b>(m.tmp);
    }
}

Entity makeBody(Context &ctx, Entity body_grp, BodyDesc desc)
{
    BodyGroupProperties &p = ctx.get<BodyGroupProperties>(body_grp);
    BodyGroupMemory &m = ctx.get<BodyGroupMemory>(body_grp);

    Entity b = ctx.makeEntity<DofObjectArchetype>();

    ctx.get<DofObjectGroup>(b) = {
        .bodyGroup = body_grp,
        .idx = p.tmp.bodyCounter,
    };

    ctx.get<DofObjectProxies>(b) = {
        .responseType = desc.responseType,

        .visualOffset = p.numObjData,
        .numVisuals = desc.numVisualObjs,

        .colliderOffset = p.numObjData + desc.numVisualObjs,
        .numColliders = desc.numCollisionObjs,
    };

    p.numObjData += desc.numVisualObjs + desc.numCollisionObjs;

    BodyDesc *body_descs = (BodyDesc *)m.tmpPtr;

    // Record entity b
    ((Entity *)(body_descs + p.numBodies))[p.tmp.bodyCounter] = b;
    body_descs[p.tmp.bodyCounter++] = desc;

    if (p.tmp.bodyCounter == p.numBodies) {
        initBodyGroupMemory(
            ctx, 
            ctx.get<BodyGroupProperties>(body_grp),
            ctx.get<BodyGroupMemory>(body_grp));
    }

    return b;
}

void attachCollision(
        Context &ctx,
        Entity body_grp,
        Entity body,
        uint32_t idx,
        CollisionDesc desc)
{
    BodyGroupMemory m = ctx.get<BodyGroupMemory>(body_grp);
    BodyGroupProperties p = ctx.get<BodyGroupProperties>(body_grp);

    DofObjectProxies proxies = ctx.get<DofObjectProxies>(body);
    DofObjectGroup body_grpinfo = ctx.get<DofObjectGroup>(body);

    BodyObjectData *obj_data = m.objectData(p);

    // Optionally attach a render object
    Entity render_entity = Entity::none();
    if (desc.renderObjID != -1) {
        Entity viz_obj = ctx.makeEntity<LinkVisual>();

        ctx.get<ObjectID>(viz_obj) = { (int32_t)desc.renderObjID };

        // Make this entity renderable
        render::RenderingSystem::makeEntityRenderable(ctx, viz_obj);

        ctx.get<LinkParentDofObject>(viz_obj) = {
            .bodyGroup = body_grp,
            .bodyIdx = body_grpinfo.idx,
            .objDataIdx = proxies.colliderOffset + idx,
            .type = LinkParentDofObject::Type::RenderCollider,
        };
    }

    Entity col_obj = ctx.makeEntity<LinkCollider>();

    ctx.get<DisabledColliders>(col_obj).numDisabled = 0;

    ctx.get<broadphase::LeafID>(col_obj) =
        PhysicsSystem::registerEntity(ctx, col_obj, { (int32_t)desc.objID });
    ctx.get<ResponseType>(col_obj) = proxies.responseType;
    ctx.get<ObjectID>(col_obj) = { (int32_t)desc.objID };

    ctx.get<Velocity>(col_obj) = {
        Vector3::zero(),
        Vector3::zero(),
    };

    ctx.get<ExternalForce>(col_obj) = Vector3::zero();
    ctx.get<ExternalTorque>(col_obj) = Vector3::zero();

    ctx.get<LinkParentDofObject>(col_obj) = {
        .bodyGroup = body_grp,
        .bodyIdx = body_grpinfo.idx,
        .objDataIdx = proxies.colliderOffset + idx,
        .type = LinkParentDofObject::Type::Collider,
    };

    obj_data[proxies.colliderOffset + idx] = {
        col_obj,
        desc.offset,
        desc.rotation,
        desc.scale,
        render_entity,
    };
}

void attachVisual(
        Context &ctx,
        Entity body_grp,
        Entity body,
        uint32_t idx,
        VisualDesc desc)
{
    BodyGroupMemory m = ctx.get<BodyGroupMemory>(body_grp);
    BodyGroupProperties p = ctx.get<BodyGroupProperties>(body_grp);
    DofObjectGroup body_grpinfo = ctx.get<DofObjectGroup>(body);
    DofObjectProxies proxies = ctx.get<DofObjectProxies>(body);
    BodyObjectData *obj_data = m.objectData(p);

    Entity viz_obj = ctx.makeEntity<LinkVisual>();

    ctx.get<ObjectID>(viz_obj) = { (int32_t)desc.objID };

    render::RenderingSystem::makeEntityRenderable(ctx, viz_obj);

    ctx.get<LinkParentDofObject>(viz_obj) = {
        .bodyGroup = body_grp,
        .bodyIdx = body_grpinfo.idx,
        .objDataIdx = proxies.visualOffset + idx,
        .type = LinkParentDofObject::Type::Render,
    };

    obj_data[proxies.visualOffset + idx] = {
        viz_obj,
        desc.offset,
        desc.rotation,
        desc.scale,
        Entity::none(),
    };
}

void setRoot(
        Context &ctx,
        Entity body_grp,
        Entity body)
{
    BodyGroupMemory m = ctx.get<BodyGroupMemory>(body_grp);
    BodyGroupProperties p = ctx.get<BodyGroupProperties>(body_grp);
    DofObjectGroup body_info = ctx.get<DofObjectGroup>(body);

    BodyOffsets *offsets = m.offsets(p);

    // We require that the root have index 0
    offsets[0].parent = 0xFF;
    assert(body_info.idx == 0);
}

void disableJointCollisions(
        Context &ctx,
        Entity grp,
        Entity joint_a,
        Entity joint_b)
{
    BodyGroupMemory m = ctx.get<BodyGroupMemory>(grp);
    BodyGroupProperties p = ctx.get<BodyGroupProperties>(grp);
    
    DofObjectProxies a_proxies = ctx.get<DofObjectProxies>(joint_a);
    DofObjectProxies b_proxies = ctx.get<DofObjectProxies>(joint_b);

    BodyObjectData *obj_datas = m.objectData(p);

    for (uint32_t a_idx = 0; a_idx < a_proxies.numColliders; ++a_idx) {
        for (uint32_t b_idx = 0; b_idx < b_proxies.numColliders; ++b_idx) {
            Entity a_obj = obj_datas[a_idx + a_proxies.colliderOffset].proxy;
            Entity b_obj = obj_datas[b_idx + b_proxies.colliderOffset].proxy;
            PhysicsSystem::disableCollision(ctx, a_obj, b_obj);
        }
    }
}

static inline void joinBodiesGeneral(
        Context &ctx,
        Entity body_grp,
        Entity parent_physics_entity,
        Entity child_physics_entity,
        Vector3 rel_position_parent,
        Vector3 rel_position_child,
        Quat rel_parent_rotation,
        Vector3 axis = Vector3 { 0.f, 0.f, 0.f })
{
    DofObjectGroup parent_info = ctx.get<DofObjectGroup>(parent_physics_entity);
    DofObjectGroup child_info = ctx.get<DofObjectGroup>(child_physics_entity);

    BodyGroupMemory m = ctx.get<BodyGroupMemory>(body_grp);
    BodyGroupProperties p = ctx.get<BodyGroupProperties>(body_grp);

    BodyHierarchy *hiers = m.hierarchies(p);
    BodyOffsets *offsets = m.offsets(p);

    assert(parent_info.idx < 0xFF);

    offsets[child_info.idx].parent = (uint8_t)parent_info.idx;

    hiers[child_info.idx] = BodyHierarchy {
        .axis = axis,
        .relPositionParent = rel_position_parent,
        .relPositionLocal = rel_position_child,
        .parentToChildRot = rel_parent_rotation,
    };
    
    // You need to disable all the colliders between these two
    disableJointCollisions(
            ctx,
            body_grp,
            parent_physics_entity,
            child_physics_entity);
}

void joinBodies(
        Context &ctx,
        Entity body_grp,
        Entity parent_physics_entity,
        Entity child_physics_entity,
        JointHinge hinge_info)
{
    joinBodiesGeneral(ctx, 
                      body_grp,
                      parent_physics_entity,
                      child_physics_entity,
                      hinge_info.relPositionParent,
                      hinge_info.relPositionChild,
                      hinge_info.relParentRotation,
                      hinge_info.hingeAxis);
}

void joinBodies(
        Context &ctx,
        Entity body_grp,
        Entity parent_physics_entity,
        Entity child_physics_entity,
        JointBall ball_info)
{
    joinBodiesGeneral(ctx, 
                      body_grp,
                      parent_physics_entity,
                      child_physics_entity,
                      ball_info.relPositionParent,
                      ball_info.relPositionChild,
                      ball_info.relParentRotation);
}

void joinBodies(
        Context &ctx,
        Entity body_grp,
        Entity parent_physics_entity,
        Entity child_physics_entity,
        JointSlider slider_info)
{
    joinBodiesGeneral(ctx, 
                      body_grp,
                      parent_physics_entity,
                      child_physics_entity,
                      slider_info.relPositionParent,
                      slider_info.relPositionChild,
                      slider_info.relParentRotation,
                      slider_info.slideVector);
}

void joinBodies(
        Context &ctx,
        Entity body_grp,
        Entity parent_physics_entity,
        Entity child_physics_entity,
        JointFixed fixed_info)
{
    joinBodiesGeneral(ctx, 
                      body_grp,
                      parent_physics_entity,
                      child_physics_entity,
                      fixed_info.relPositionParent,
                      fixed_info.relPositionChild,
                      fixed_info.relParentRotation);
}

void attachLimit(
        Context &ctx,
        Entity body_grp,
        Entity body,
        HingeLimit hinge_limit)
{
    DofObjectGroup body_info = ctx.get<DofObjectGroup>(body);

    BodyGroupMemory m = ctx.get<BodyGroupMemory>(body_grp);
    BodyGroupProperties p = ctx.get<BodyGroupProperties>(body_grp);

    BodyOffsets *offsets = m.offsets(p);
    BodyLimitConstraint *limits = m.limits(p);

    assert(offsets[body_info.idx].numEqs > 0);

    auto &l = limits[offsets[body_info.idx].eqOffset];
    l.type = BodyLimitConstraint::Type::Hinge;
    l.bodyIdx = body_info.idx;
    l.hinge = hinge_limit;

    // Set the joint value to be in the limits
    m.q(p)[offsets[body_info.idx].posOffset] = (hinge_limit.lower + hinge_limit.upper) / 2.f;
}

void attachLimit(
        Context &ctx,
        Entity body_grp,
        Entity body,
        SliderLimit slider_limit)
{
    DofObjectGroup body_info = ctx.get<DofObjectGroup>(body);

    BodyGroupMemory m = ctx.get<BodyGroupMemory>(body_grp);
    BodyGroupProperties p = ctx.get<BodyGroupProperties>(body_grp);

    BodyOffsets *offsets = m.offsets(p);
    BodyLimitConstraint *limits = m.limits(p);

    assert(offsets[body_info.idx].numEqs > 0);

    auto &l = limits[offsets[body_info.idx].eqOffset];
    l.type = BodyLimitConstraint::Type::Slider;
    l.bodyIdx = body_info.idx;
    l.slider = slider_limit;

    m.q(p)[offsets[body_info.idx].posOffset] = (slider_limit.lower + slider_limit.upper) / 2.f;
}

// External forces:
void addHingeExternalForce(
        Context &ctx, Entity hinge_joint, float newtons)
{
    DofObjectGroup joint_info = ctx.get<DofObjectGroup>(hinge_joint);
    
    BodyGroupMemory m = ctx.get<BodyGroupMemory>(joint_info.bodyGroup);
    BodyGroupProperties p = ctx.get<BodyGroupProperties>(joint_info.bodyGroup);

    BodyOffsets *offsets = m.offsets(p);
    float *f = m.f(p);

    f[offsets[joint_info.idx].velOffset] = newtons;
}

float * getBodyGroupDofPos(Context &ctx, Entity body_grp) {
    BodyGroupMemory &m = ctx.get<BodyGroupMemory>(body_grp);
    BodyGroupProperties &p = ctx.get<BodyGroupProperties>(body_grp);
    return m.q(p);
}

float * getBodyGroupDofVel(Context &ctx, Entity body_grp) {
    BodyGroupMemory &m = ctx.get<BodyGroupMemory>(body_grp);
    BodyGroupProperties &p = ctx.get<BodyGroupProperties>(body_grp);
    return m.qv(p);
}

float * getBodyGroupDofAcc(Context &ctx, Entity body_grp) {
    BodyGroupMemory &m = ctx.get<BodyGroupMemory>(body_grp);
    BodyGroupProperties &p = ctx.get<BodyGroupProperties>(body_grp);
    return m.dqv(p);
}

float * getBodyGroupForces(Context &ctx, Entity body_grp)
{
    BodyGroupMemory &m = ctx.get<BodyGroupMemory>(body_grp);
    BodyGroupProperties &p = ctx.get<BodyGroupProperties>(body_grp);
    return m.f(p);
}

uint8_t getBodyNumDofs(Context &ctx, Entity body_grp, uint32_t body_idx)
{
    BodyGroupMemory &m = ctx.get<BodyGroupMemory>(body_grp);
    BodyGroupProperties &p = ctx.get<BodyGroupProperties>(body_grp);
    BodyOffsets *offsets = m.offsets(p);
    return offsets[body_idx].numDofs;
}

float * getBodyDofPos(Context &ctx, Entity body_grp, uint32_t body_idx) {
    BodyGroupMemory &m = ctx.get<BodyGroupMemory>(body_grp);
    BodyGroupProperties &p = ctx.get<BodyGroupProperties>(body_grp);
    BodyOffsets *offsets = m.offsets(p);
    return m.q(p) + offsets[body_idx].posOffset;
}

float * getBodyDofVel(Context &ctx, Entity body_grp, uint32_t body_idx) {
    BodyGroupMemory &m = ctx.get<BodyGroupMemory>(body_grp);
    BodyGroupProperties &p = ctx.get<BodyGroupProperties>(body_grp);
    BodyOffsets *offsets = m.offsets(p);
    return m.qv(p) + offsets[body_idx].velOffset;
}

float * getBodyDofAcc(Context &ctx, Entity body_grp, uint32_t body_idx) {
    BodyGroupMemory &m = ctx.get<BodyGroupMemory>(body_grp);
    BodyGroupProperties &p = ctx.get<BodyGroupProperties>(body_grp);
    BodyOffsets *offsets = m.offsets(p);
    return m.dqv(p) + offsets[body_idx].velOffset;
}

BodyTransform getBodyWorldPos(Context &ctx, Entity body_grp, uint32_t body_idx)
{
    BodyGroupMemory &m = ctx.get<BodyGroupMemory>(body_grp);
    BodyGroupProperties &p = ctx.get<BodyGroupProperties>(body_grp);
    return m.bodyTransforms(p)[body_idx];
}

float * getBodyForces(Context &ctx, Entity body_grp, uint32_t body_idx)
{
    BodyGroupMemory &m = ctx.get<BodyGroupMemory>(body_grp);
    BodyGroupProperties &p = ctx.get<BodyGroupProperties>(body_grp);
    BodyOffsets *offsets = m.offsets(p);
    return m.f(p) + offsets[body_idx].velOffset;
}

static uint32_t getBodyIndex(
        BodyGroupMemory &m,
        BodyGroupProperties &p,
        StringID string_id)
{
    BodyNameHash *hashes = m.nameHashes(p);
    for (uint32_t i = 0; i < p.numHashes; ++i) {
        if (string_id.hash == hashes[i].hash) {
            return hashes[i].bodyIdx;
        }
    }

    assert(false);
    return 0xFFFF'FFFF;
}

uint8_t getBodyNumDofs(Context &ctx, Entity body_grp, StringID string_id)
{
    BodyGroupMemory &m = ctx.get<BodyGroupMemory>(body_grp);
    BodyGroupProperties &p = ctx.get<BodyGroupProperties>(body_grp);
    BodyOffsets *offsets = m.offsets(p);
    return offsets[getBodyIndex(m, p, string_id)].numDofs;
}

float * getBodyDofPos(Context &ctx, Entity body_grp, StringID string_id)
{
    BodyGroupMemory &m = ctx.get<BodyGroupMemory>(body_grp);
    BodyGroupProperties &p = ctx.get<BodyGroupProperties>(body_grp);
    BodyOffsets *offsets = m.offsets(p);
    return m.q(p) + offsets[getBodyIndex(m, p, string_id)].posOffset;
}

float * getBodyDofVel(Context &ctx, Entity body_grp, StringID string_id)
{
    BodyGroupMemory &m = ctx.get<BodyGroupMemory>(body_grp);
    BodyGroupProperties &p = ctx.get<BodyGroupProperties>(body_grp);
    BodyOffsets *offsets = m.offsets(p);
    return m.qv(p) + offsets[getBodyIndex(m, p, string_id)].velOffset;
}

float * getBodyDofAcc(Context &ctx, Entity body_grp, StringID string_id)
{
    BodyGroupMemory &m = ctx.get<BodyGroupMemory>(body_grp);
    BodyGroupProperties &p = ctx.get<BodyGroupProperties>(body_grp);
    BodyOffsets *offsets = m.offsets(p);
    return m.dqv(p) + offsets[getBodyIndex(m, p, string_id)].velOffset;
}

float * getBodyForces(Context &ctx, Entity body_grp, StringID string_id)
{
    BodyGroupMemory &m = ctx.get<BodyGroupMemory>(body_grp);
    BodyGroupProperties &p = ctx.get<BodyGroupProperties>(body_grp);
    BodyOffsets *offsets = m.offsets(p);
    return m.f(p) + offsets[getBodyIndex(m, p, string_id)].velOffset;
}

BodyTransform getBodyWorldPos(Context &ctx, Entity body_grp, StringID string_id)
{
    BodyGroupMemory &m = ctx.get<BodyGroupMemory>(body_grp);
    BodyGroupProperties &p = ctx.get<BodyGroupProperties>(body_grp);
    return m.bodyTransforms(p)[getBodyIndex(m, p, string_id)];
}

BodyInertial & getBodyInertial(Context &ctx, Entity body_grp, uint32_t body_idx)
{
    BodyGroupMemory &m = ctx.get<BodyGroupMemory>(body_grp);
    BodyGroupProperties &p = ctx.get<BodyGroupProperties>(body_grp);
    return m.inertials(p)[body_idx];
}

BodyInertial & getBodyInertial(Context &ctx, Entity body_grp, StringID string_id)
{
    BodyGroupMemory &m = ctx.get<BodyGroupMemory>(body_grp);
    BodyGroupProperties &p = ctx.get<BodyGroupProperties>(body_grp);
    return m.inertials(p)[getBodyIndex(m, p, string_id)];
}

}
