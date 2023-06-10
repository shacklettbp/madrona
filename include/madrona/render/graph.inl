namespace madrona::render {

template <typename Fn>
void RenderGraphBuilder::addTask(
        Fn &&fn,
        TaskType type,
        Span<TaskResource> read_resources,
        Span<TaskResource> write_resources)
{
    auto *fn_ptr = &RenderGraph::taskEntry<Fn>;

    auto *closure_store = alloc_->alloc<Fn>();
    *closure_store = fn;

    CountT num_data_bytes = sizeof(Fn);

    CountT num_read_resource_bytes =
        sizeof(TaskResource) * read_resources.size();
    CountT num_write_resource_bytes =
        sizeof(TaskResource) * write_resources.size();
    auto *read_resources_dst = (TaskResource *)alloc_->alloc(
        num_read_resource_bytes, alignof(TaskResource));
    auto *write_resources_dst = (TaskResource *)alloc_->alloc(
        num_write_resource_bytes, alignof(TaskResource));

    memcpy(read_resources_dst, read_resources.data(), num_read_resource_bytes);
    memcpy(write_resources_dst, write_resources.data(),
           num_write_resource_bytes);

    TaskDesc *new_task = alloc_->alloc<TaskDesc>();
    new_task->fn = fn_ptr;
    new_task->data = closure_store;
    new_task->numDataBytes = (uint32_t)num_data_bytes;
    new_task->type = type;
    new_task->readResources = read_resources_dst;
    new_task->writeResources = write_resources_dst;
    new_task->next = nullptr;

    task_list_tail_->next = new_task;
    task_list_head_ = new_task;
}

template <typename Fn>
void RenderGraph::taskEntry(void *data, GPU &gpu)
{
    auto &closure = *(Fn *)data;
    closure(gpu);
}

}
