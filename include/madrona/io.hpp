/*
 * Copyright 2021-2022 Brennan Shacklett and contributors
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 */
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
