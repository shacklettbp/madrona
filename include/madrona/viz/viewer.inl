namespace madrona::viz {

bool Viewer::UserInput::keyPressed(KeyboardKey key) const
{
    return keys_state_[(uint32_t)key];
}

bool Viewer::UserInput::keyHit(KeyboardKey key) const
{
    return press_state_[(uint32_t)key];
}

template <typename InputFn, typename StepFn, typename UIFn>
void Viewer::loop(InputFn &&input_fn, StepFn &&step_fn, UIFn &&ui_fn)
{
    void *input_data = &input_fn;
    void *step_data = &step_fn;
    void *ui_data = &ui_fn;

    loop([](void *input_data, CountT world_idx, CountT agent_idx,
            const UserInput &input) {
        auto *lambda_ptr = (InputFn *)input_data;
        (*lambda_ptr)(world_idx, agent_idx, input);
    }, input_data, [](void *step_data) {
        auto *lambda_ptr = (StepFn *)step_data;
        (*lambda_ptr)();
    }, step_data, [](void *ui_data) {
        auto *lambda_ptr = (UIFn *)ui_data;
        (*lambda_ptr)();
    }, ui_data);
}

}
