#include <madrona/physics.hpp>
#include <madrona/context.hpp>

namespace madrona {

using namespace base;
using namespace math;

namespace phys {

struct SolverData {
    Contact *contacts;
    std::atomic<CountT> numContacts;
    CountT maxContacts;
    float deltaT;
    float h;

    inline SolverData(CountT max_contacts_per_step,
                      float delta_t,
                      CountT num_substeps)
        : contacts((Contact *)rawAlloc(sizeof(Contact) * max_contacts_per_step)),
          numContacts(0),
          maxContacts(max_contacts_per_step),
          deltaT(delta_t),
          h(delta_t / (float)num_substeps)
    {}

    inline void addContacts(Span<const Contact> added_contacts)
    {
        int32_t contact_idx = numContacts.fetch_add(added_contacts.size(),
                                                    std::memory_order_relaxed);
        assert(contact_idx < maxContacts);

        for (CountT i = 0; i < added_contacts.size(); i++) {
            contacts[contact_idx + i] = added_contacts[i];
        }
    }
};

struct ObjectData {
    ObjectManager *mgr;
};

inline void updateCollisionAABB(Context &ctx,
                                const Position &pos,
                                const Rotation &rot,
                                const ObjectID &obj_id,
                                const Velocity &vel,
                                CollisionAABB &out_aabb)
{
    // FIXME: this could all be more efficient with a center + width
    // AABB representation
    ObjectManager &obj_mgr = *ctx.getSingleton<ObjectData>().mgr;

    Mat3x3 rot_mat = Mat3x3::fromQuat(rot);
    AABB obj_aabb = obj_mgr.aabbs[obj_id.idx];

    AABB world_aabb;

    // RTCD page 86
#pragma unroll
    for (CountT i = 0; i < 3; i++) {
        world_aabb.pMin[i] = world_aabb.pMax[i] = pos[i];

#pragma unroll
        for (CountT j = 0; j < 3; j++) {
            float e = rot_mat[i][j] * obj_aabb.pMin[j];
            float f = rot_mat[i][j] * obj_aabb.pMax[j];

            if (e < f) {
                world_aabb.pMin[i] += e;
                world_aabb.pMax[i] += f;
            } else {
                world_aabb.pMin[i] += f;
                world_aabb.pMax[i] += e;
            }
        }
    }

    constexpr float expansion_factor = 2.f;
    constexpr float max_accel = 100.f;

    float delta_t = ctx.getSingleton<SolverData>().deltaT;
    float min_pos_change = max_accel * delta_t * delta_t;

    Vector3 linear_velocity = vel.linear;

#pragma unroll
    for (int32_t i = 0; i < 3; i++) {
        float pos_delta = expansion_factor * linear_velocity[i] * delta_t;

        float min_delta = pos_delta - min_pos_change;
        float max_delta = pos_delta + min_pos_change;

        if (min_delta < 0.f) {
            world_aabb.pMin[i] += min_delta;
        }
        if (max_delta > 0.f) {
            world_aabb.pMax[i] += max_delta;
        }
    }

    out_aabb = world_aabb;
}

namespace narrowphase {

enum class NarrowphaseTest : uint32_t {
    SphereSphere = 1,
    HullHull = 2,
    SphereHull = 3,
    PlanePlane = 4,
    SpherePlane = 5,
    HullPlane = 6,
};

inline void runNarrowphase(
    Context &ctx,
    const CandidateCollision &candidate_collision)
{
    ObjectID a_obj = ctx.getUnsafe<ObjectID>(candidate_collision.a);
    ObjectID b_obj = ctx.getUnsafe<ObjectID>(candidate_collision.b);

    SolverData &solver = ctx.getSingleton<SolverData>();
    const ObjectManager &obj_mgr = *ctx.getSingleton<ObjectData>().mgr;

    const CollisionPrimitive *a_prim = &obj_mgr.primitives[a_obj.idx];
    const CollisionPrimitive *b_prim = &obj_mgr.primitives[b_obj.idx];

    uint32_t raw_type_a = static_cast<uint32_t>(a_prim->type);
    uint32_t raw_type_b = static_cast<uint32_t>(b_prim->type);

    Entity a_entity = candidate_collision.a;
    Entity b_entity = candidate_collision.b;

    if (raw_type_a > raw_type_b) {
        std::swap(raw_type_a, raw_type_b);
        std::swap(a_entity, b_entity);
        std::swap(a_prim, b_prim);
    }

    NarrowphaseTest test_type {raw_type_a | raw_type_b};

    Position a_pos = ctx.getUnsafe<Position>(a_entity);
    Position b_pos = ctx.getUnsafe<Position>(b_entity);

    switch (test_type) {
    case NarrowphaseTest::SphereSphere: {
        float a_radius = a_prim->sphere.radius;
        float b_radius = b_prim->sphere.radius;

        Vector3 to_b = b_pos - a_pos;
        float dist = to_b.length();

        if (dist > 0 && dist < a_radius + b_radius) {
            Vector3 mid = to_b / 2.f;

            Vector3 to_b_normal = to_b / dist;
            solver.addContacts({{
                a_entity,
                b_entity,
                { 
                    makeVector4(a_pos + mid, dist / 2.f),
                    {}, {}, {}
                },
                1,
                to_b_normal,
            }});


            Loc loc = ctx.makeTemporary<CollisionEventTemporary>();
            ctx.getUnsafe<CollisionEvent>(loc) = CollisionEvent {
                candidate_collision.a,
                candidate_collision.b,
            };
        }
    } break;
    case NarrowphaseTest::PlanePlane: {
        // Do nothing, planes must be static.
        // Should rework this entire setup so static objects
        // aren't checked against the BVH
    } break;
    case NarrowphaseTest::SpherePlane: {
        auto sphere = a_prim->sphere;
        Rotation b_rot = ctx.getUnsafe<Rotation>(b_entity);

        constexpr Vector3 base_normal = { 0, 0, 1 };
        Vector3 plane_normal = b_rot.rotateDir(base_normal);

        float d = plane_normal.dot(b_pos);
        float t = plane_normal.dot(a_pos) - d;

        if (t < sphere.radius) {
            solver.addContacts({{
                a_entity,
                Entity::none(),
                {
                    makeVector4(a_pos + plane_normal * sphere.radius, t),
                    {}, {}, {}
                },
                1,
                plane_normal,
            }});
        }
    } break;
    case NarrowphaseTest::HullHull: {
        // Get half edge mesh for hull A and hull B
        const auto &hEdgeA = a_prim->hull.halfEdgeMesh;
        const auto &hEdgeB = a_prim->hull.halfEdgeMesh;

        auto transformVertex = [&ctx] (math::Vector3 v, Entity &e) {
            Scale e_scale = ctx.getUnsafe<Scale>(e);
            Rotation e_rotation = ctx.getUnsafe<Rotation>(e);
            Position e_position = ctx.getUnsafe<Position>(e);

            return e_position + e_rotation.rotateDir((math::Vector3)e_scale * v);
        };

        geometry::CollisionMesh collisionMeshA;
        collisionMeshA.halfEdgeMesh = &hEdgeA;
        collisionMeshA.vertexCount = hEdgeA.getVertexCount();
        collisionMeshA.vertices = (math::Vector3 *)TmpAllocator::get().alloc(sizeof(math::Vector3) * collisionMeshA.vertexCount);
        collisionMeshA.center = a_pos;
        for (int v = 0; v < collisionMeshA.vertexCount; ++v) {
            collisionMeshA.vertices[v] = transformVertex(hEdgeA.vertex(v), a_entity);
        }

        geometry::CollisionMesh collisionMeshB;
        collisionMeshB.halfEdgeMesh = &hEdgeB;
        collisionMeshB.vertexCount = hEdgeB.getVertexCount();
        collisionMeshB.vertices = (math::Vector3 *)TmpAllocator::get().alloc(sizeof(math::Vector3) * collisionMeshB.vertexCount);
        collisionMeshB.center = b_pos;
        for (int v = 0; v < collisionMeshB.vertexCount; ++v) {
            collisionMeshB.vertices[v] = transformVertex(hEdgeB.vertex(v), b_entity);
        }

        Manifold manifold = doSAT(collisionMeshA, collisionMeshB);

        if (manifold.numContactPoints > 0) {
            solver.addContacts({{
                manifold.aIsReference ? a_entity : b_entity,
                manifold.aIsReference ? b_entity : a_entity,
                {
                    manifold.contactPoints[0],
                    manifold.contactPoints[1],
                    manifold.contactPoints[2],
                    manifold.contactPoints[3],
                },
                manifold.numContactPoints,
                manifold.normal,
            }});
        }
    } break;
    case NarrowphaseTest::SphereHull: {
        assert(false);
    } break;
    case NarrowphaseTest::HullPlane: {
        // Get half edge mesh for hull A and hull B
        const auto &hEdgeA = a_prim->hull.halfEdgeMesh;

        auto transformVertex = [&ctx] (math::Vector3 v, Entity &e) {
            Scale e_scale = ctx.getUnsafe<Scale>(e);
            Rotation e_rotation = ctx.getUnsafe<Rotation>(e);
            Position e_position = ctx.getUnsafe<Position>(e);

            return e_position + e_rotation.rotateDir((math::Vector3)e_scale * v);
        };

        geometry::CollisionMesh collisionMeshA;
        collisionMeshA.halfEdgeMesh = &hEdgeA;
        collisionMeshA.vertexCount = hEdgeA.getVertexCount();
        collisionMeshA.vertices = (math::Vector3 *)TmpAllocator::get().alloc(sizeof(math::Vector3) * collisionMeshA.vertexCount);
        collisionMeshA.center = a_pos;
        for (int v = 0; v < collisionMeshA.vertexCount; ++v) {
            collisionMeshA.vertices[v] = transformVertex(hEdgeA.vertex(v), a_entity);
        }

        Rotation b_rot = ctx.getUnsafe<Rotation>(b_entity);
        constexpr Vector3 base_normal = { 0, 0, 1 };
        Vector3 plane_normal = b_rot.rotateDir(base_normal);

        geometry::Plane plane = { b_pos, plane_normal };

        Manifold manifold = doSATPlane(plane, collisionMeshA);

        if (manifold.numContactPoints > 0) {
            solver.addContacts({{
                a_entity,
                Entity::none(),
                {
                    makeVector3(manifold.contactPoints[0]),
                    makeVector3(manifold.contactPoints[1]),
                    makeVector3(manifold.contactPoints[2]),
                    makeVector3(manifold.contactPoints[3]),
                },
                manifold.numContactPoints,
                manifold.normal,
            }});
        }
    } break;
    default: __builtin_unreachable();
    }
}

}

namespace solver {

inline void updatePositions(Context &ctx,
                            Position &pos,
                            Rotation &rot,
                            Velocity &vel,
                            const ObjectID &obj_id,
                            InstanceState &inst_state)
{
    (void)rot;
    (void)obj_id;

    const auto &solver = ctx.getSingleton<SolverData>();
    float h = solver.h;

    inst_state.prevPosition = pos;
    inst_state.prevRotation = rot;

    Vector3 linear_velocity = vel.linear;
    //cur_velocity += h * gravity;

    pos += h * linear_velocity;
}

inline void updateVelocities(Context &ctx,
                             const Position &pos,
                             const InstanceState &inst_state,
                             Velocity &vel)
{
    const auto &solver = ctx.getSingleton<SolverData>();
    float h = solver.h;

    vel.linear = (pos - inst_state.prevPosition) / h;
}

static inline void handleContactConstraint(Position &pos_ref,
                                           Rotation &rot_ref,
                                           InstanceState inst_state,
                                           Vector3 contact_point)
{
    Vector3 cur_translate = pos_ref;
    Quat cur_rotation = rot_ref;
}

static inline void handleContact(Context &ctx, Contact &contact)
{
    Position &a_pos = ctx.getUnsafe<Position>(contact.a);
    Rotation &a_rot = ctx.getUnsafe<Rotation>(contact.a);
    //Velocity &a_vel = ctx.getUnsafe<Velocity>(contact.a);
    InstanceState a_state = ctx.getUnsafe<InstanceState>(contact.a);

#pragma unroll
    for (CountT i = 0; i < 4; i++) {
        if (i >= contact.numPoints) continue;

        Vector3 contact_point = contact.points[i].xyz();

        handleContactConstraint(a_pos, a_rot, a_state, contact_point);

        a_pos += contact.normal;
    }

    if (contact.b != Entity::none()) {
        Position &b_pos = ctx.getUnsafe<Position>(contact.b);
        b_pos -= contact.normal;
    }
}

inline void solverEntry(Context &ctx, SolverData &solver)
{
    // Push objects in serial based on the contact normal - total BS.
    CountT num_contacts = solver.numContacts.load(std::memory_order_relaxed);

    //printf("Solver # contacts: %d\n", num_contacts);

    for (CountT i = 0; i < num_contacts; i++) {
        Contact &contact = solver.contacts[i];

        handleContact(ctx, contact);
    }

    solver.numContacts.store(0, std::memory_order_relaxed);
}

}

void RigidBodyPhysicsSystem::init(Context &ctx,
                                  ObjectManager *obj_mgr,
                                  float delta_t,
                                  CountT num_substeps,
                                  CountT max_dynamic_objects,
                                  CountT max_contacts_per_world)
{
    broadphase::BVH &bvh = ctx.getSingleton<broadphase::BVH>();
    new (&bvh) broadphase::BVH(max_dynamic_objects);

    SolverData &solver = ctx.getSingleton<SolverData>();
    new (&solver) SolverData(max_contacts_per_world, delta_t, num_substeps);

    ObjectData &objs = ctx.getSingleton<ObjectData>();
    new (&objs) ObjectData { obj_mgr };
}

void RigidBodyPhysicsSystem::reset(Context &ctx)
{
    broadphase::BVH &bvh = ctx.getSingleton<broadphase::BVH>();
    bvh.rebuildOnUpdate();
    bvh.clearLeaves();
}

broadphase::LeafID RigidBodyPhysicsSystem::registerEntity(Context &ctx,
                                                          Entity e)
{
    return ctx.getSingleton<broadphase::BVH>().reserveLeaf(e);
}

void RigidBodyPhysicsSystem::registerTypes(ECSRegistry &registry)
{
    registry.registerComponent<broadphase::LeafID>();
    registry.registerSingleton<broadphase::BVH>();

    registry.registerComponent<Velocity>();
    registry.registerComponent<CollisionAABB>();

    registry.registerComponent<solver::InstanceState>();

    registry.registerComponent<CollisionEvent>();
    registry.registerArchetype<CollisionEventTemporary>();

    registry.registerComponent<CandidateCollision>();
    registry.registerArchetype<CandidateTemporary>();

    registry.registerSingleton<SolverData>();
    registry.registerSingleton<ObjectData>();

}

TaskGraph::NodeID RigidBodyPhysicsSystem::setupTasks(
    TaskGraph::Builder &builder, Span<const TaskGraph::NodeID> deps,
    CountT num_substeps)
{
    auto update_aabbs = builder.addToGraph<ParallelForNode<Context,
        updateCollisionAABB, Position, Rotation, ObjectID, Velocity,
            CollisionAABB>>(deps);

    auto preprocess_leaves = builder.addToGraph<ParallelForNode<Context,
        broadphase::updateLeavesEntry, broadphase::LeafID, 
        CollisionAABB>>({update_aabbs});

    auto bvh_update = builder.addToGraph<ParallelForNode<Context,
        broadphase::updateBVHEntry, broadphase::BVH>>({preprocess_leaves});

    auto find_overlapping = builder.addToGraph<ParallelForNode<Context,
        broadphase::findOverlappingEntry, Entity, CollisionAABB, Velocity>>(
            {bvh_update});
    
    auto cur_node = find_overlapping;
    for (CountT i = 0; i < num_substeps; i++) {
        auto update_positions = builder.addToGraph<ParallelForNode<Context,
            solver::updatePositions, Position, Rotation, Velocity, ObjectID,
            solver::InstanceState>>({cur_node});

        auto run_narrowphase = builder.addToGraph<ParallelForNode<Context,
            narrowphase::runNarrowphase, CandidateCollision>>(
                {update_positions});

        auto solver = builder.addToGraph<ParallelForNode<Context,
            solver::solverEntry, SolverData>>({run_narrowphase});

        auto vel_update = builder.addToGraph<ParallelForNode<Context,
            solver::updateVelocities, Position,
            solver::InstanceState, Velocity>>({solver});

        cur_node = builder.addToGraph<ResetTmpAllocNode>({vel_update});
    }

    auto clear_candidates = builder.addToGraph<
        ClearTmpNode<CandidateTemporary>>({cur_node});

    return clear_candidates;
}

TaskGraph::NodeID RigidBodyPhysicsSystem::setupCleanupTasks(
    TaskGraph::Builder &builder, Span<const TaskGraph::NodeID> deps)
{
    return builder.addToGraph<ClearTmpNode<CollisionEventTemporary>>(deps);
}

}
}

