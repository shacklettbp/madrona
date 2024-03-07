namespace madrona {

template <EnumType EnumT>
MWCudaLaunchGraph MWCudaExecutor::buildLaunchGraph(
    Span<const EnumT> taskgraph_ids)
{
    buildLaunchGraph(Span<const uint32_t>(
        static_cast<const uint32_t *>(taskgraph_ids.data()),
        taskgraph_ids.size()));
}

}
