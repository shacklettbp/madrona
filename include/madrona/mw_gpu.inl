namespace madrona {

template <EnumType EnumT>
MWCudaLaunchGraph MWCudaExecutor::buildLaunchGraph(EnumT taskgraph_id,
                                                   bool enable_raytracing)
{
    return buildLaunchGraph(static_cast<uint32_t>(taskgraph_id),
                            enable_raytracing);
}

MWCudaLaunchGraph MWCudaExecutor::buildLaunchGraph(uint32_t taskgraph_id,
                                                   bool enable_raytracing)
{
    return buildLaunchGraph(Span<const uint32_t>(&taskgraph_id, 1),
                            enable_raytracing);
}

}
