namespace madrona::render {

GPU::GPU(backend::Backend &backend, const backend::DeviceID &dev_id)
    : dev_(backend.initDevice(dev_id))
{}

backend::Device & GPU::backendDevice() { return dev_; }

}
