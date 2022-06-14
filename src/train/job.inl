namespace madrona {

template <typename Fn>
__global__ void jobEntry(JobQueue *job_queue, void *data)
{
    Context ctx {
        .jobQueue = *job_queue,
    };

    auto fn_ptr = (Fn *)data;
    (*fn_ptr)(ctx);
    fn_ptr->~Fn();

    atomicSub(&ctx.jobQueue.numOutstandingJobs, 1u);
}

}
