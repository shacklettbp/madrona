#include <madrona/context.hpp>

#include <fstream>

#include "collisions.hpp"

using namespace madrona;

namespace CollisionExample {

static void launch()
{
    StateManager state_mgr;
    
    JobManager job_mgr(JobManager::makeEntry<Engine, CollisionSim>(
        [](Engine &ctx) {
            CollisionSim::entry(ctx);
        }), 4, 0, &state_mgr);
    
    job_mgr.waitForAllFinished();
}

}

int main(int argc, char *argv[])
{
    (void)argc;
    (void)argv;
    CollisionExample::launch();
}
