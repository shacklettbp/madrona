#include <madrona/physics.hpp>
#include <madrona/context.hpp>

#include "physics_impl.hpp"
#include "xpbd.hpp"
#include "tgs.hpp"

namespace madrona::phys {

using namespace base;
using namespace math;

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


static SolverData initSolverState(Context &ctx,
                                  float delta_t,
                                  CountT num_substeps,
                                  Vector3 gravity)
{
    float h = delta_t / (float)num_substeps;
    float g_mag = gravity.length();

    return SolverData {
        .deltaT = delta_t,
        .h = h,
        .g = gravity,
        .gMagnitude = g_mag,
        .restitutionThreshold = 2.f * g_mag * h,
        .jointQuery = ctx.query<JointConstraint>(),
        .contactQuery = ctx.query<ContactConstraint>(),
    };
}

namespace PhysicsSystem {

void init(Context &ctx,
          ObjectManager *obj_mgr,
          float delta_t,
          CountT num_substeps,
          math::Vector3 gravity,
          CountT max_dynamic_objects)
{
    broadphase::BVH &bvh = ctx.singleton<broadphase::BVH>();

    // expansion factor is 2 * delta_t to give room
    // for acceleration within the timestep
    constexpr float max_inst_accel = 100.f;
    new (&bvh) broadphase::BVH(
        obj_mgr, max_dynamic_objects, 2.f * delta_t,
        max_inst_accel * delta_t * delta_t);

    ctx.singleton<SolverData>() = initSolverState(
        ctx, delta_t, num_substeps, gravity);

    ObjectData &objs = ctx.singleton<ObjectData>();
    new (&objs) ObjectData { obj_mgr };
}

void reset(Context &ctx)
{
    broadphase::BVH &bvh = ctx.singleton<broadphase::BVH>();
    bvh.rebuildOnUpdate();
    bvh.clearLeaves();
}

broadphase::LeafID registerEntity(Context &ctx,
                                  Entity e,
                                  ObjectID obj_id)
{
    return ctx.singleton<broadphase::BVH>().reserveLeaf(e, obj_id);
}

bool checkEntityAABBOverlap(
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


Entity makeFixedJoint(
    Context &ctx,
    Entity e1, Entity e2,
    math::Quat attach_rot1, math::Quat attach_rot2,
    math::Vector3 r1, math::Vector3 r2,
    float separation)
{
    Entity e = ctx.makeEntity<Joint>();

    ctx.get<JointConstraint>(e) = {
        .e1 = e1,
        .e2 = e2,
        .type = JointConstraint::Type::Fixed,
        .fixed = {
            .attachRot1 = attach_rot1,
            .attachRot2 = attach_rot2,
            .separation = separation,
        },
        .r1 = r1,
        .r2 = r2,
    };

    return e;
}

Entity makeHingeJoint(
    Context &ctx,
    Entity e1, Entity e2,
    math::Vector3 a1_local, math::Vector3 a2_local,
    math::Vector3 b1_local, math::Vector3 b2_local,
    math::Vector3 r1, math::Vector3 r2)
{
    Entity e = ctx.makeEntity<Joint>();

    ctx.get<JointConstraint>(e) = {
        .e1 = e1,
        .e2 = e2,
        .type = JointConstraint::Type::Hinge,
        .hinge = {
            .a1Local = a1_local,
            .a2Local = a2_local,
            .b1Local = b1_local,
            .b2Local = b2_local,
        },
        .r1 = r1,
        .r2 = r2,
    };

    return e;
}

void registerTypes(ECSRegistry &registry)
{
    registry.registerComponent<broadphase::LeafID>();
    registry.registerSingleton<broadphase::BVH>();

    registry.registerComponent<ExternalForce>();
    registry.registerComponent<ExternalTorque>();
    registry.registerComponent<ResponseType>();
    registry.registerComponent<Velocity>();

    registry.registerComponent<CollisionEvent>();
    registry.registerArchetype<CollisionEventTemporary>();

    registry.registerComponent<CandidateCollision>();
    registry.registerArchetype<CandidateTemporary>();

    registry.registerComponent<JointConstraint>();
    registry.registerArchetype<Joint>();

    registry.registerComponent<ContactConstraint>();
    registry.registerArchetype<Contact>();

    registry.registerSingleton<SolverData>();
    registry.registerSingleton<ObjectData>();

    xpbd::registerTypes(registry);
}

TaskGraphNodeID setupBroadphaseTasks(
    TaskGraphBuilder &builder,
    Span<const TaskGraphNodeID> deps)
{
    return broadphase::setupBVHTasks(builder, deps);
}

TaskGraphNodeID setupBroadphaseOverlapTasks(
    TaskGraphBuilder &builder,
    Span<const TaskGraphNodeID> deps)
{
    return broadphase::setupPreIntegrationTasks(builder, deps);
}

TaskGraphNodeID setupSubstepTasks(
    TaskGraphBuilder &builder,
    Span<const TaskGraphNodeID> deps,
    CountT num_substeps,
    bool use_tgs)
{
    auto broadphase_pre =
        broadphase::setupPreIntegrationTasks(builder, deps);

    auto cur_node = broadphase_pre;

#ifdef MADRONA_GPU_MODE
    cur_node = 
        builder.addToGraph<SortArchetypeNode<Joint, WorldID>>({cur_node});
    cur_node = builder.addToGraph<ResetTmpAllocNode>({cur_node});
#endif

    if (use_tgs) {
        cur_node = tgs::setupTGSSolverTasks(builder, cur_node, num_substeps);
    } else {
        cur_node = xpbd::setupXPBDSolverTasks(builder, cur_node, num_substeps);
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

TaskGraphNodeID setupCleanupTasks(
    TaskGraphBuilder &builder, Span<const TaskGraphNodeID> deps)
{
    return builder.addToGraph<ClearTmpNode<CollisionEventTemporary>>(deps);
}

}

}
