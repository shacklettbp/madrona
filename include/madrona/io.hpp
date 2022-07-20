#pragma once

#include <madrona/job.hpp>

namespace madrona {

struct IOBuffer {
    void *data;
    uint64_t numBytes;
};

struct IOPromise {
    IOBuffer buffer;
};

class IOManager {
public:
    IOManager(JobManager &job_mgr);

    IOPromise makePromise();

    void load(IOPromise promise, const char *path, Job job);

    IOBuffer getBuffer(IOPromise promise);

private:
    JobManager *job_mgr_;
};

}
