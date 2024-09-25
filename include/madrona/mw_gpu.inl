namespace madrona {

template <EnumType EnumT>
MWCudaLaunchGraph MWCudaExecutor::buildLaunchGraph(EnumT taskgraph_id,
                                                   const char *stat_name)
{
    return buildLaunchGraph(static_cast<uint32_t>(taskgraph_id), stat_name);
}

MWCudaLaunchGraph MWCudaExecutor::buildLaunchGraph(uint32_t taskgraph_id,
                                                   const char *stat_name)
{
    return buildLaunchGraph(Span<const uint32_t>(&taskgraph_id, 1), stat_name);
}

}
