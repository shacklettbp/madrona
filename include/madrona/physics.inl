#pragma once

namespace madrona::phys {

namespace geometry {

uint32_t HalfEdgeMesh::getVertexCount() const
{
    return mVertexCount;
}

const math::Vector3 &HalfEdgeMesh::vertex(uint32_t id) const
{
    return mVertices[id];
}

const math::Vector3 * HalfEdgeMesh::vertices() const
{
    return mVertices;
}


template <typename Fn>
void HalfEdgeMesh::iteratePolygonIndices(PolygonID poly,
                                         Fn &&fn)
{
    // Half edge of the polygon
    uint32_t hEdge = mPolygons[poly];
    uint32_t start = hEdge;

    fn(halfEdge(hEdge).rootVertex);

    while (halfEdge(hEdge).next != start) {
        fn(halfEdge(halfEdge(hEdge).next).rootVertex);
        hEdge = halfEdge(hEdge).next;
    }
}

}

namespace broadphase {

LeafID BVH::reserveLeaf(Entity e, CollisionPrimitive *prim)
{
    int32_t leaf_idx = num_leaves_.fetch_add_relaxed(1);
    assert(leaf_idx < num_allocated_leaves_);

    leaf_entities_[leaf_idx] = e;
    leaf_primitives_[leaf_idx] = prim;

    return LeafID {
        leaf_idx,
    };
}

math::AABB BVH::getLeafAABB(LeafID leaf_id) const
{
    return leaf_aabbs_[leaf_id.id];
}

template <typename Fn>
void BVH::findOverlaps(const math::AABB &aabb, Fn &&fn) const
{
    int32_t stack[128];
    stack[0] = 0;
    CountT stack_size = 1;

    while (stack_size > 0) {
        int32_t node_idx = stack[--stack_size];
        const Node &node = nodes_[node_idx];
        for (int i = 0; i < 4; i++) {
            if (!node.hasChild(i)) {
                continue; // Technically this could be break?
            };

            madrona::math::AABB child_aabb {
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
            };

            if (aabb.overlaps(child_aabb)) {
                if (node.isLeaf(i)) {
                    Entity e = leaf_entities_[node.leafIDX(i)];
                    fn(e);
                } else {
                    stack[stack_size++] = node.children[i];
                }
            }
        }
    }
}

template <typename Fn>
void BVH::findOverlapsForLeaf(LeafID leaf_id, Fn &&fn) const
{
    math::AABB leaf_aabb = leaf_aabbs_[leaf_id.id];
    findOverlaps(leaf_aabb, std::forward<Fn>(fn));
}

void BVH::rebuildOnUpdate()
{
    force_rebuild_ = true;
}

void BVH::clearLeaves()
{
    num_leaves_.store_relaxed(0);
}

bool BVH::Node::isLeaf(CountT child) const
{
    return children[child] & 0x80000000;
}

int32_t BVH::Node::leafIDX(CountT child) const
{
    return children[child] & ~0x80000000;
}

void BVH::Node::setLeaf(CountT child, int32_t idx)
{
    children[child] = 0x80000000 | idx;
}

void BVH::Node::setInternal(CountT child, int32_t internal_idx)
{
    children[child] = internal_idx;
}

bool BVH::Node::hasChild(CountT child) const
{
    return children[child] != sentinel_;
}

void BVH::Node::clearChild(CountT child)
{
    children[child] = sentinel_;
}

}

}
