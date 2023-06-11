namespace madrona::render {

struct RenderGraphBuilder::LogicalResource {
    enum class Type {
        Buffer,
        Texture2D,
    } type;

    union {
        Texture2DDesc tex2D;
        BufferDesc buffer;
    };

    bool isLive;

    LogicalResource *next;
};

struct RenderGraphBuilder::TaskDesc {
    void (*fn)(void *, GPU &);
    void *data;
    CountT numDataBytes;

    TaskType type;

    Span<TaskResource> readResources;
    Span<TaskResource> writeResources;

    TaskDesc *next;
};

LogicalResource * RenderGraphBuilder::addTex2D(Texture2DDesc desc)
{
    auto *resource = addResource();
    resource->type = LogicalResource::Type::Texture2D;
    resource->tex2D = desc;

    return TaskResource { resource } ;
}

LogicalResource * RenderGraphBuilder::addBuffer(CountT num_bytes)
{
    auto *resource = addResource();
    resource->type = LogicalResource::Type::Buffer;

    return TaskResource { resource };
}

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

    auto *read_resources_dst =
        alloc_->allocN<TaskResource>(read_resources.size());
    auto *write_resources_dst =
        alloc_->allocN<TaskResource>(write_resources.size());

    utils::copyN<TaskResource>(read_resources_dst, read_resources.data(),
                               read_resources.size());
    utils::copyN<TaskResource>(write_resources_dst, write_resources.data(),
                               write_resources.size());

    TaskDesc *new_task = alloc_->alloc<TaskDesc>();
    new_task->fn = fn_ptr;
    new_task->data = closure_store;
    new_task->numDataBytes = (uint32_t)num_data_bytes;
    new_task->type = type;
    new_task->readResources = Span(read_resources_dst, read_resources.size());
    new_task->writeResources =
        Span(write_resources_dst, write_resources.size());

    new_task->next = nullptr;
    task_list_tail_->next = new_task;
    task_list_tail_ = new_task;
}

LogicalResource * RenderGraphBuilder::addResource()
{
    auto *resource = alloc_->alloc<LogicalResource>();
    resource->isLive = false;

    resource->next = nullptr;
    rsrc_list_tail_->next = resource;
    rsrc_list_tail_ = resource;

    return resource;
}

LogicalResource * RenderGraphBuilder::getResource(TaskResource hdl)
{
    return (LogicalResource *)(hdl.hdl);
}

template <typename Fn>
void RenderGraph::taskEntry(void *data, GPU &gpu)
{
    auto &closure = *(Fn *)data;
    closure(gpu);
}

}
