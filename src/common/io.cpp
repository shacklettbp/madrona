#include <madrona/io.hpp>

namespace madrona {

IOManager::IOManager(JobManager &job_mgr)
    : job_mgr_(job_mgr)
{}

IOPromise IOManager::makePromise()
{
}

IOPromise IOManager::load(IOPromise promise, const char *path, Job job)
{
}

IOBuffer IOManager::getBuffer(IOPromise promise)
{
}

}
