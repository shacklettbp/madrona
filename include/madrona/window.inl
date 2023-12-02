namespace madrona {

// Workaround for what seems to be a clang compiler bug:
// https://stackoverflow.com/questions/53408962/try-to-understand-compiler-error-message-default-member-initializer-required-be
WindowManager::Config WindowManager::defaultConfig() { return {}; }

WindowHandle::WindowHandle(WindowHandle &&o)
{
    this->win_ = o.win_;
    this->wm_ = o.wm_;

    o.win_ = nullptr;
}

WindowHandle::~WindowHandle()
{
    if (win_) {
        wm_->destroyWindow(win_);
    }
}

Window * WindowHandle::get()
{
    return win_;
}


}
