namespace madrona {

// Workaround for what seems to be a clang compiler bug:
// https://stackoverflow.com/questions/53408962/try-to-understand-compiler-error-message-default-member-initializer-required-be
WindowManager::Config WindowManager::defaultConfig() { return {}; }

Window * WindowHandle::get()
{
    return win_;
}

WindowHandle::WindowHandle(Window *win, WindowManager *wm)
    : win_(win), wm_(wm)
{
}

WindowHandle::~WindowHandle()
{
    if (win_)
        wm_->destroyWindow(win_);
}

WindowHandle::WindowHandle(WindowHandle &&o)
{
    this->win_ = o.win_;
    this->wm_ = o.wm_;

    o.win_ = nullptr;
    o.wm_ = nullptr;
}

}
