#pragma once

namespace madrona {
namespace phys {

template <typename Fn>
void BroadphaseBVH::findOverlaps(madrona::math::AABB &aabb, Fn &&fn)
{
    int32_t stack[128];
    stack[0] = 0;
    int32_t stack_size = 1;

    while (stack_size > 0) {
        int32_t node_idx = stack[--stack_size];
        PhysicsBVHNode &node = nodes[node_idx];
        for (int i = 0; i < 4; i++) {
            int child_idx = node.children[i];
            if (child_idx == sentinel) {
                continue;
            }

            madrona::math::AABB child_aabb {
                .pMin = {
                    node.minX[i],
                    node.minY[i],
                    node.minZ[i],
                },
                .pMax = {
                    node.maxX[i],
                    node.maxY[i],
                    node.maxZ[i],
                },
            };

            if (aabb.overlaps(child_aabb)) {
                if (node.isLeaf(i)) {
                    fn(node.leafRawEntity(i));
                } else {
                    stack[stack_size++] = child_idx;
                }
            }
        }
    }
}

bool BroadphaseBVH::Node::isLeaf(IdxT child) const
{
    return children[child] & 0x80000000;
}

uint32_t BroadphaseBVH::Node::leafRawEntity(IdxT child) const
{
    return uint32_t(children[child] & ~0x80000000);
}

void BroadphaseBVH::Node::setLeaf(IdxT child, int32_t entity_id)
{
    children[child] = 0x80000000 | entity_id;
}

void BroadphaseBVH::Node::setInternal(IdxT child, int32_t internal_idx)
{
    children[child] = internal_idx;
}

void BroadphaseBVH::Node::clearChild(IdxT child)
{
    children[child] = sentinel_;
}

}
}
