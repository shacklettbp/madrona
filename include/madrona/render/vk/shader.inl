namespace madrona::render::vk {

RasterParamBlock Shader::makeRasterParamBlock(CountT block_idx)
{
    return makeParamBlock(block_idx);
}

ComputeParamBlock Shader::makeComputeParamBlock(CountT block_idx)
{
    return makeParamBlock(block_idx);
}

}
