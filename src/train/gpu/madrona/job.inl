#include <type_traits>

namespace madrona {

template <typename Fn>
__global__ void Context::jobEntry(void *data, uint32_t num_launches,
                                  uint32_t grid_id)
{
    Context ctx {
        .grid_id_ = grid_id,
    };

    if constexpr (std::is_empty_v<Fn>) {
        Fn()(ctx);
    } else {
        auto fn_ptr = (Fn *)data;
        (*fn_ptr)(ctx);
        fn_ptr->~Fn();
    }

    ctx.markJobFinished();
}

template <typename Fn>
void Context::queueJob(Fn &&fn)
{
    auto func_ptr = jobEntry<Fn>;

    Job job;
    job.fn = func_ptr;
    if constexpr (std::is_empty_v<Fn>) {
        job.data = nullptr;
    } else {
    }

    queueJob(job);
}

}
