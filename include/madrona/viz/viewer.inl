namespace madrona::viz {

template <typename StepFn>
void Viewer::loop(StepFn &&step_fn)
{
    void *data = &step_fn;

    loop([](void *data) {
        auto *lambda_ptr = (StepFn *)data;
        (*lambda_ptr)();

    }, data);
}

}
