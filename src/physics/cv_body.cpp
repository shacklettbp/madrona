#include <madrona/cv_physics.hpp>

namespace madrona::phys::cv {
    
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
        m.tmpPtr = ctx.tmpAlloc(sizeof(BodyDesc) * num_bodies);
    }

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
        p.qDim += BodyOffsets:getDofTypeDim(bd.type, true);
        p.qvDim += BodyOffsets:getDofTypeDim(bd.type);
        q.numDofs += BodyOffsets::getDofTypedim(bd.type);
        p.numEq += bd.numLimits;
        p.numObjData += bd.numCollisionObjs + bd.numVisualObjs;
    }

    {// Allocate frame persistent memory
        uint32_t num_bytes = BodyGroupMemory::qVectorsNumBytes(p);
        uint32_t num_elems = (num_bytes + sizeof(MRElement128b) - 1) /
            sizeof(MRElement128b);
        m.qVectors = ctx.allocMemoryRange<MRElement128b>(num_elems);
        m.qVectorsPtr = ctx.memoryRangePointer(m.qVectors);
    }

    { // Allocate frame volatile memory

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

    ctx.get<DofObjectProxy>(b) = {
        .numVisuals = desc.numVisualObjs,
        .numColliders = desc.numCollisionObjs,
    };

    BodyDesc *body_descs = (BodyDesc *)m.tmpPtr;
    body_descs[p.tmp.bodyCounter++] = desc;

    if (p.tmp.bodyCounter == p.numBodies) {
        initBodyGroupMemory(ctx, body_grp);
    }
}

}
