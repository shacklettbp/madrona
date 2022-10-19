/*
 * Copyright 2021-2022 Brennan Shacklett and contributors
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 */
#include <madrona/context.hpp>

#include "simple.hpp"
#include "init.hpp"

using namespace madrona;

namespace SimpleExample {

static void launch()
{
    StateManager state_mgr;

    EnvInit env_init = generateEnvironmentInitialization(0);
    
    JobManager job_mgr(JobManager::makeEntry<Engine>(
        [&env_init](Engine &ctx) {
            SimpleSim::entry(ctx, env_init);
        }), 0, 0, &state_mgr);
    
    job_mgr.waitForAllFinished();

    free(env_init.objsInit);
}

}

int main(int, char **)
{
    SimpleExample::launch();
}
