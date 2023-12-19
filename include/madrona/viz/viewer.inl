namespace madrona::viz {

bool Viewer::UserInput::keyPressed(KeyboardKey key) const
{
    return keys_state_[(uint32_t)key];
}

bool Viewer::UserInput::keyHit(KeyboardKey key) const
{
    return press_state_[(uint32_t)key];
}

template <typename WorldInputFn, typename AgentInputFn,
          typename StepFn, typename UIFn>
void Viewer::loop(WorldInputFn &&world_input_fn, AgentInputFn &&agent_input_fn,
                  StepFn &&step_fn, UIFn &&ui_fn)
{
    void *world_input_data = &world_input_fn;
    void *agent_input_data = &agent_input_fn;
    void *step_data = &step_fn;
    void *ui_data = &ui_fn;

    loop([](
        void *world_input_data,
        CountT world_idx,
        const UserInput &input)
    {
        auto *lambda_ptr = (WorldInputFn *)world_input_data;
        (*lambda_ptr)(world_idx, input);
    }, world_input_data, [](
        void *agent_input_data,
        CountT world_idx,
        CountT agent_idx,
        const UserInput &input)
    {
        auto *lambda_ptr = (AgentInputFn *)agent_input_data;
        (*lambda_ptr)(world_idx, agent_idx, input);
    }, agent_input_data, [](void *step_data)
    {
        auto *lambda_ptr = (StepFn *)step_data;
        (*lambda_ptr)();
    }, step_data, [](void *ui_data)
    {
        auto *lambda_ptr = (UIFn *)ui_data;
        (*lambda_ptr)();
    }, ui_data);
}

}
