namespace madrona::render::vk {

void ParamBlockAllocator::addParamBlock(const ParamBlockDesc &desc)
{
}

RasterParamBlock ParamBlockAllocator::makeRasterParamBlock(CountT block_idx)
{
    return makeParamBlock(block_idx);
}

ComputeParamBlock ParamBlockAllocator::makeComputeParamBlock(CountT block_idx)
{
    return makeParamBlock(block_idx);
}

}
