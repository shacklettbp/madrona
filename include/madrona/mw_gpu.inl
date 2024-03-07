namespace madrona {

template <EnumType EnumT>
MWCudaLaunchGraph MWCudaExecutor::buildLaunchGraph(EnumT taskgraph_id)
{
    return buildLaunchGraph(static_cast<uint32_t>(taskgraph_id));
}

MWCudaLaunchGraph MWCudaExecutor::buildLaunchGraph(uint32_t taskgraph_id)
{
    return buildLaunchGraph(Span<const uint32_t>(&taskgraph_id, 1));
}

}
