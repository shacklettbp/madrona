#include <madrona/physics.hpp>
#include <madrona/context.hpp>

namespace madrona {
using namespace base;
using namespace math;

namespace phys {

struct CandidateCollision {
    Entity a;
    Entity b;
};

struct CandidateTemporary : Archetype<CandidateCollision> {};

struct Contact {
    Entity a;
    Entity b;
    Vector3 normal;
};

struct CollisionEventTemporary : Archetype<CollisionEvent> {};

struct SolverData {
    Contact *contacts;
    std::atomic<CountT> numContacts;
    CountT maxContacts;
    float h;

    inline SolverData(CountT max_contacts_per_step,
                      float delta_t,
                      CountT num_substeps)
        : contacts((Contact *)rawAlloc(sizeof(Contact) * max_contacts_per_step)),
          numContacts(0),
          maxContacts(max_contacts_per_step),
          h(delta_t / (float)num_substeps)
    {}

    inline void addContact(Contact c)
    {
        int32_t contact_idx =
            numContacts.fetch_add(1, std::memory_order_relaxed);
        assert(contact_idx < maxContacts);

        contacts[contact_idx] = c;
    }
};

static inline AABB computeAABBFromMesh(
    const base::Position &pos,
    const base::Rotation &rot)
{
    Mat3x4 model_mat = Mat3x4::fromTRS(pos, rot);

    // No actual mesh, just hardcode a fake 2 *unit cube centered around
    Vector3 cube[8] = {
        model_mat.txfmPoint(Vector3 {-1.f, -1.f, -1.f}),
        model_mat.txfmPoint(Vector3 { 1.f, -1.f, -1.f}),
        model_mat.txfmPoint(Vector3 { 1.f,  1.f, -1.f}),
        model_mat.txfmPoint(Vector3 {-1.f,  1.f, -1.f}),
        model_mat.txfmPoint(Vector3 {-1.f, -1.f,  1.f}),
        model_mat.txfmPoint(Vector3 { 1.f, -1.f,  1.f}),
        model_mat.txfmPoint(Vector3 { 1.f,  1.f,  1.f}),
        model_mat.txfmPoint(Vector3 {-1.f,  1.f,  1.f}),
    };

    AABB aabb = AABB::point(cube[0]);
    for (int i = 1; i < 8; i++) {
        aabb.expand(cube[i]);
    }

    return aabb;
}

inline void updateCollisionAABB(Context &,
                                const Position &pos,
                                const Rotation &rot,
                                CollisionAABB &out_aabb)
{
    out_aabb = computeAABBFromMesh(pos, rot);
}

namespace broadphase {

BVH::BVH(CountT max_leaves)
    : nodes_((Node *)rawAlloc(sizeof(Node) *
                            numInternalNodes(max_leaves))),
      num_nodes_(0),
      num_allocated_nodes_(numInternalNodes(max_leaves)),
      leaf_aabbs_((AABB *)rawAlloc(sizeof(AABB) * max_leaves)),
      leaf_centers_((Vector3 *)rawAlloc(sizeof(Vector3) * max_leaves)),
      leaf_parents_((uint32_t *)rawAlloc(sizeof(uint32_t) * max_leaves)),
      leaf_entities_((Entity *)rawAlloc(sizeof(Entity) * max_leaves)),
      sorted_leaves_((int32_t *)rawAlloc(sizeof(int32_t) * max_leaves)),
      num_leaves_(0),
      num_allocated_leaves_(max_leaves),
      force_rebuild_(false)
{}

CountT BVH::numInternalNodes(CountT num_leaves) const
{
    return std::max(utils::divideRoundUp(num_leaves - 1, CountT(3)), CountT(1)) +
        num_leaves; // + num_leaves should not be necessary but the current
                    // top down build has an issue where leaves get
                    // unnecessarily split amongst internal nodes with only
                    // 1 or 2 children
}

void BVH::rebuild()
{
    int32_t num_internal_nodes =
        numInternalNodes(num_leaves_.load(std::memory_order_relaxed));
    num_nodes_ = num_internal_nodes;
    assert(num_nodes_ <= num_allocated_nodes_);

    struct StackEntry {
        int32_t nodeID;
        int32_t parentID;
        int32_t offset;
        int32_t numObjs;
    };

    StackEntry stack[128];
    stack[0] = StackEntry {
        sentinel_,
        sentinel_,
        0,
        int32_t(num_leaves_),
    };

    int32_t cur_node_offset = 0;
    CountT stack_size = 1;

    while (stack_size > 0) {
        StackEntry &entry = stack[stack_size - 1];
        int32_t node_id;
        if (entry.numObjs <= 4) {
            node_id = cur_node_offset++;

            Node &node = nodes_[node_id];
            node.parentID = entry.parentID;

            for (int i = 0; i < 4; i++) {
                if (i < entry.numObjs) {
                    int32_t leaf_id = sorted_leaves_[entry.offset + i];

                    const auto &aabb = leaf_aabbs_[leaf_id];
                    leaf_parents_[leaf_id] = ((uint32_t)node_id << 2) | (uint32_t)i;

                    node.setLeaf(i, leaf_id);
                    node.minX[i] = aabb.pMin.x;
                    node.minY[i] = aabb.pMin.y;
                    node.minZ[i] = aabb.pMin.z;
                    node.maxX[i] = aabb.pMax.x;
                    node.maxY[i] = aabb.pMax.y;
                    node.maxZ[i] = aabb.pMax.z;
                } else {
                    node.clearChild(i);
                    node.minX[i] = FLT_MAX;
                    node.minY[i] = FLT_MAX;
                    node.minZ[i] = FLT_MAX;
                    node.maxX[i] = FLT_MIN;
                    node.maxY[i] = FLT_MIN;
                    node.maxZ[i] = FLT_MIN;
                }
            }
        } else if (entry.nodeID == sentinel_) {
            node_id = cur_node_offset++;
            // Record the node id in the stack entry for when this entry
            // is reprocessed
            entry.nodeID = node_id;

            Node &node = nodes_[node_id];
            for (CountT i = 0; i < 4; i++) {
                node.clearChild(i);
            }
            node.parentID = entry.parentID;

            // midpoint sort items
            auto midpoint_split = [this](
                    int32_t base, int32_t num_elems) {

                auto get_center = [this, base](int32_t offset) {
                    return leaf_centers_[sorted_leaves_[base + offset]];
                };

                Vector3 center_min {
                    FLT_MAX,
                    FLT_MAX,
                    FLT_MAX,
                };

                Vector3 center_max {
                    FLT_MIN,
                    FLT_MIN,
                    FLT_MIN,
                };

                for (int i = 0; i < num_elems; i++) {
                    const Vector3 &center = get_center(i);
                    center_min = Vector3::min(center_min, center);
                    center_max = Vector3::max(center_max, center);
                }

                auto split = [&](auto get_component) {
                    float split_val = 0.5f * (get_component(center_min) +
                                              get_component(center_max));

                    int start = 0;
                    int end = num_elems;

                    while (start < end) {
                        while (start < end &&
                               get_component(get_center(start)) < split_val) {
                            ++start;
                        }

                        while (start < end && get_component(
                                get_center(end - 1)) >= split_val) {
                            --end;
                        }

                        if (start < end) {
                            std::swap(sorted_leaves_[base + start],
                                      sorted_leaves_[base + end - 1]);
                            ++start;
                            --end;
                        }
                    }

                    if (start > 0 && start < num_elems) {
                        return start;
                    } else {
                        return num_elems / 2;
                    }
                };

                Vector3 center_diff = center_max - center_min;
                if (center_diff.x > center_diff.y &&
                    center_diff.x > center_diff.z) {
                    return split([](Vector3 v) {
                        return v.x;
                    });
                } else if (center_diff.y > center_diff.x &&
                           center_diff.y > center_diff.z) {
                    return split([](Vector3 v) {
                        return v.y;
                    });
                } else {
                    return split([](Vector3 v) {
                        return v.z;
                    });
                }
            };

            int32_t second_split = midpoint_split(entry.offset, entry.numObjs);
            int32_t num_h1 = second_split;
            int32_t num_h2 = entry.numObjs - second_split;

            int32_t first_split = midpoint_split(entry.offset, num_h1);
            int32_t third_split =
                midpoint_split(entry.offset + second_split, num_h2);

            // Setup stack to recurse into fourths. Put fourths on stack in
            // reverse order to preserve left-right depth first ordering

            stack[stack_size++] = {
                -1,
                entry.nodeID,
                entry.offset + num_h1 + third_split,
                num_h2 - third_split,
            };

            stack[stack_size++] = {
                -1,
                entry.nodeID,
                entry.offset + num_h1,
                third_split,
            };

            stack[stack_size++] = {
                -1,
                entry.nodeID,
                entry.offset + first_split,
                num_h1 - first_split,
            };

            stack[stack_size++] = {
                -1,
                entry.nodeID,
                entry.offset,
                first_split,
            };

            // Don't finish processing this node until children are processed
            continue;
        } else {
            // Revisiting this node after having processed children
            node_id = entry.nodeID;
        }

        // At this point, remove the current entry from the stack
        stack_size -= 1;

        Node &node = nodes_[node_id];
        if (node.parentID == -1) {
            continue;
        }

        AABB combined_aabb = AABB::invalid(); 
        for (CountT i = 0; i < 4; i++) {
            if (!node.hasChild(i)) {
                break;
            }

            combined_aabb = AABB::merge(combined_aabb, AABB {
                /* .pMin = */ {
                    node.minX[i],
                    node.minY[i],
                    node.minZ[i],
                },
                /* .pMax = */ {
                    node.maxX[i],
                    node.maxY[i],
                    node.maxZ[i],
                },
            });
        }

        Node &parent = nodes_[node.parentID];
        CountT child_offset;
        for (child_offset = 0; ; child_offset++) {
            if (parent.children[child_offset] == sentinel_) {
                break;
            }
        }

        parent.setInternal(child_offset, node_id);
        parent.minX[child_offset] = combined_aabb.pMin.x;
        parent.minY[child_offset] = combined_aabb.pMin.y;
        parent.minZ[child_offset] = combined_aabb.pMin.z;
        parent.maxX[child_offset] = combined_aabb.pMax.x;
        parent.maxY[child_offset] = combined_aabb.pMax.y;
        parent.maxZ[child_offset] = combined_aabb.pMax.z;
    }
}

void BVH::refit(LeafID *moved_leaf_ids, CountT num_moved)
{
    (void)moved_leaf_ids;
    (void)num_moved;

    int32_t num_moved_hacked = num_leaves_.load(std::memory_order_relaxed);

    for (CountT i = 0; i < num_moved_hacked; i++) {
        int32_t leaf_id = i;
        const AABB &leaf_aabb = leaf_aabbs_[leaf_id];
        uint32_t leaf_parent = leaf_parents_[leaf_id];

        int32_t node_idx = int32_t(leaf_parent >> 2_u32);
        int32_t sub_idx = int32_t(leaf_parent & 3);

        Node &leaf_node = nodes_[node_idx];
        leaf_node.minX[sub_idx] = leaf_aabb.pMin.x;
        leaf_node.minY[sub_idx] = leaf_aabb.pMin.y;
        leaf_node.minZ[sub_idx] = leaf_aabb.pMin.z;
        leaf_node.maxX[sub_idx] = leaf_aabb.pMax.x;
        leaf_node.maxY[sub_idx] = leaf_aabb.pMax.y;
        leaf_node.maxZ[sub_idx] = leaf_aabb.pMax.z;

        int32_t child_idx = node_idx;
        node_idx = leaf_node.parentID;

        while (node_idx != sentinel_) {
            Node &node = nodes_[node_idx];
            int child_offset = -1;
            for (int j = 0; j < 4; j++) {
                if (node.children[j] == child_idx) {
                    child_offset = j;
                    break;
                }
            }
            assert(child_offset != -1);

            bool expanded = false;
            if (leaf_aabb.pMin.x < node.minX[child_offset]) {
                node.minX[child_offset] = leaf_aabb.pMin.x;
                expanded = true;
            }

            if (leaf_aabb.pMin.y < node.minY[child_offset]) {
                node.minY[child_offset] = leaf_aabb.pMin.y;
                expanded = true;
            }

            if (leaf_aabb.pMin.z < node.minZ[child_offset]) {
                node.minZ[child_offset] = leaf_aabb.pMin.z;
                expanded = true;
            }

            if (leaf_aabb.pMax.x > node.maxX[child_offset]) {
                node.maxX[child_offset] = leaf_aabb.pMax.x;
                expanded = true;
            }

            if (leaf_aabb.pMax.y > node.maxY[child_offset]) {
                node.maxY[child_offset] = leaf_aabb.pMax.y;
                expanded = true;
            }

            if (leaf_aabb.pMax.z > node.maxZ[child_offset]) {
                node.maxZ[child_offset] = leaf_aabb.pMax.z;
                expanded = true;
            }

            if (!expanded) {
                break;
            }
            
            child_idx = node_idx;
            node_idx = node.parentID;
        }
    }
}

void BVH::updateLeaf(Entity e,
                     LeafID leaf_id,
                     const CollisionAABB &obj_aabb)
{
    // FIXME, handle difference between potentially inflated leaf AABB and
    // object AABB
    AABB &leaf_aabb = leaf_aabbs_[leaf_id.id];
    leaf_aabb = obj_aabb;

    Vector3 &leaf_center = leaf_centers_[leaf_id.id];
    leaf_center = (leaf_aabb.pMin + leaf_aabb.pMax) / 2;

    leaf_entities_[leaf_id.id] = e;
    sorted_leaves_[leaf_id.id] = leaf_id.id;
}

void BVH::updateTree()
{
    if (force_rebuild_) {
        force_rebuild_ = false;
        rebuild();
    } else {
        refit(nullptr, 0);
    }
}

inline void updateLeavesEntry(
    Context &ctx,
    const Entity &e,
    const LeafID &leaf_id,
    const CollisionAABB &aabb)
{
    BVH &bvh = ctx.getSingleton<BVH>();
    bvh.updateLeaf(e, leaf_id, aabb);
}

inline void updateBVHEntry(
    Context &, BVH &bvh)
{
    bvh.updateTree();
}

inline void findOverlappingEntry(
    Context &ctx,
    const Entity &e,
    const CollisionAABB &obj_aabb)
{
    BVH &bvh = ctx.getSingleton<BVH>();

    bvh.findOverlaps(obj_aabb, [&](Entity overlapping_entity) {
        if (e.id < overlapping_entity.id) {
            Loc candidate_loc = ctx.makeTemporary<CandidateTemporary>();
            CandidateCollision &candidate = ctx.getUnsafe<
                CandidateCollision>(candidate_loc);

            candidate.a = e;
            candidate.b = overlapping_entity;
        }
    });
}

}

namespace narrowphase {

inline void processCandidatesEntry(
    Context &ctx,
    const CandidateCollision &candidate_collision)
{
    Position &a_pos = ctx.getUnsafe<Position>(candidate_collision.a);
    Position &b_pos = ctx.getUnsafe<Position>(candidate_collision.b);

    // FIXME real narrowphase
    constexpr float sphere_radius = 1.f;
    Vector3 to_a = b_pos - a_pos;
    float dist = to_a.length();

    SolverData &solver = ctx.getSingleton<SolverData>();

    if (dist > 0 && dist <= sphere_radius * 2.f) {
        solver.addContact(Contact {
            candidate_collision.a,
            candidate_collision.b,
            to_a / dist,
        });
        
        Loc loc = ctx.makeTemporary<CollisionEventTemporary>();
        ctx.getUnsafe<CollisionEvent>(loc) = CollisionEvent {
            candidate_collision.a,
            candidate_collision.b,
        };
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

    Vector3 cur_velocity = vel;
    //cur_velocity += h * gravity;
    
    pos += h * cur_velocity;
}

inline void updateVelocities(Context &ctx,
                             const Position &pos,
                             const InstanceState &inst_state,
                             Velocity &vel)
{
    const auto &solver = ctx.getSingleton<SolverData>();
    float h = solver.h;

    vel = (pos - inst_state.prevPosition) / h;
}

inline void solverEntry(Context &ctx, SolverData &solver)
{
    // Push objects in serial based on the contact normal - total BS.
    CountT num_contacts = solver.numContacts.load(std::memory_order_relaxed);
    printf("Solver %d\n", num_contacts);

    for (CountT i = 0; i < num_contacts; i++) {
        Contact &contact = solver.contacts[i];

        Position &a_pos = ctx.getUnsafe<Position>(contact.a);
        Position &b_pos = ctx.getUnsafe<Position>(contact.b);

        a_pos -= contact.normal;
        b_pos += contact.normal;
    }

    solver.numContacts.store(0, std::memory_order_relaxed);
}

}

void RigidBodyPhysicsSystem::init(Context &ctx,
                                  float delta_t,
                                  CountT num_substeps,
                                  CountT max_dynamic_objects,
                                  CountT max_contacts_per_world)
{
    broadphase::BVH &bvh = ctx.getSingleton<broadphase::BVH>();
    new (&bvh) broadphase::BVH(max_dynamic_objects);

    SolverData &solver = ctx.getSingleton<SolverData>();
    new (&solver) SolverData(max_contacts_per_world, delta_t, num_substeps);
}

void RigidBodyPhysicsSystem::reset(Context &ctx)
{
    broadphase::BVH &bvh = ctx.getSingleton<broadphase::BVH>();
    bvh.rebuildOnUpdate();
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

}

TaskGraph::NodeID RigidBodyPhysicsSystem::setupTasks(
    TaskGraph::Builder &builder, Span<const TaskGraph::NodeID> deps,
    CountT num_substeps)
{
    auto update_aabbs = builder.parallelForNode<Context, updateCollisionAABB,
        Position, Rotation, CollisionAABB>(deps);

    auto preprocess_leaves = builder.parallelForNode<Context,
        broadphase::updateLeavesEntry, Entity, broadphase::LeafID, 
        CollisionAABB>({update_aabbs});

    auto bvh_update = builder.parallelForNode<Context,
        broadphase::updateBVHEntry, broadphase::BVH>({preprocess_leaves});

    auto find_overlapping = builder.parallelForNode<Context,
        broadphase::findOverlappingEntry, Entity, CollisionAABB>({bvh_update});
    
    auto cur_node = find_overlapping;
    for (CountT i = 0; i < num_substeps; i++) {
        auto update_positions = builder.parallelForNode<Context,
            solver::updatePositions, Position, Rotation, Velocity, ObjectID,
            solver::InstanceState>({cur_node});

        auto run_narrowphase = builder.parallelForNode<Context,
            narrowphase::processCandidatesEntry, CandidateCollision>(
                {update_positions});

        auto solver = builder.parallelForNode<Context,
            solver::solverEntry, SolverData>({run_narrowphase});

        cur_node = builder.parallelForNode<Context,
            solver::updateVelocities, Position,
            solver::InstanceState, Velocity>({solver});
    }

    auto clear_candidates = builder.clearTemporariesNode<CandidateTemporary>(
        {cur_node});

    return clear_candidates;
}

TaskGraph::NodeID RigidBodyPhysicsSystem::setupCleanupTasks(
    TaskGraph::Builder &builder, Span<const TaskGraph::NodeID> deps)
{
    return builder.clearTemporariesNode<CollisionEventTemporary>(deps);
}

}
}
