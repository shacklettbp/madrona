/*
 * Copyright 2021-2022 Brennan Shacklett and contributors
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 */
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
