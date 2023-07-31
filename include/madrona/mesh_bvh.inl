namespace madrona::phys {

bool MeshBVH::Node::isLeaf(madrona::CountT child) const
{
    return children[child] & 0x80000000;
}

int32_t MeshBVH::Node::leafIDX(madrona::CountT child) const
{
    return children[child] & ~0x80000000;
}

void MeshBVH::Node::setLeaf(madrona::CountT child, int32_t idx)
{
    children[child] = 0x80000000 | idx;
}

void MeshBVH::Node::setInternal(madrona::CountT child, int32_t internal_idx)
{
    children[child] = internal_idx;
}

bool MeshBVH::Node::hasChild(madrona::CountT child) const
{
    return children[child] != sentinel;
}

void MeshBVH::Node::clearChild(madrona::CountT child)
{
    children[child] = sentinel;
}

}
