namespace madrona::render {

Shader::Shader(GPU &gpu, StackAlloc &tmp_alloc,
               void *ir, CountT num_ir_bytes,
               const refl::ShaderInfo &reflection_info)
    : hdl_(gpu.backendDevice(), tmp_alloc, ir, num_ir_bytes, reflection_info)
{}

RasterParamBlock Shader::makeRasterParamBlock(CountT block_idx)
{
    return hdl_.makeRasterParamBlock(block_idx);
}

ComputeParamBlock Shader::makeComputeParamBlock(CountT block_idx)
{
    return hdl_.makeComputeParamBlock(block_idx);
}

void Shader::destroy(GPU &gpu)
{
    return hdl_.destroy(gpu);
}

}
