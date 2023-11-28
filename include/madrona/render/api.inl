namespace madrona::render {

GPUHandle::GPUHandle(GPUHandle &&o)
    : backend_(o.backend_),
      dev_(o.dev_)
{
    o.dev_ = nullptr;
}

GPUHandle & GPUHandle::operator=(GPUHandle &&o)
{
    backend_ = o.backend_;
    dev_ = o.dev_;
    o.dev_ = nullptr;

    return *this;
}

GPUDevice * GPUHandle::device() const
{
    return dev_;
}

APILibHandle::APILibHandle(APILibHandle &&o)
    : backend_(o.backend_),
      lib_(o.lib_)
{
    o.lib_ = nullptr;
}

APILibHandle & APILibHandle::operator=(APILibHandle &&o)
{
    backend_ = o.backend_;
    lib_ = o.lib_;
    o.lib_ = nullptr;

    return *this;
}

APILib * APILibHandle::lib() const
{
    return lib_;
}

APIBackend * APIManager::backend() const
{
    return backend_;
}

// Workaround for what seems to be a clang compiler bug:
// https://stackoverflow.com/questions/53408962/try-to-understand-compiler-error-message-default-member-initializer-required-be
APIManager::Config APIManager::defaultConfig() { return {}; }

}
