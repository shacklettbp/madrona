namespace madrona::render {

template <typename Fn>
void RenderGraphBuilder::addTask(Fn &&fn, TaskArguments args)
{
    CountT num_data_bytes = sizeof(Fn);
}

}
