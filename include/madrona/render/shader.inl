namespace madrona::render {

Shader::Shader(GPU &gpu, Span<char> bytecode,
               const refl::ShaderInfo &reflection_info)
    : hdl_(gpu, bytecode, reflection_info)
{}

ParamBlock Shader::makeParamBlock(CountT block_idx)
{
    return hdl_.makeParamBlock(block_idx);
}

void Shader::destroy(GPU &gpu)
{
    return hdl_.destroy(gpu);
}

}
