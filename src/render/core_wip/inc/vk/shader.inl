namespace madrona::render::vk {

void ParamBlockAllocator::addParamBlock(const ParamBlockDesc &desc,
                                        CountT num_blocks)
{
    for (CountT i = 0; i < desc.typeCounts.size(); i++) {
        type_counts_[i] += desc.typeCounts[i] * num_blocks;
    }
}

}
