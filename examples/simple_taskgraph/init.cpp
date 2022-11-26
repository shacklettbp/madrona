#include "init.hpp"

#include <random>

using namespace madrona;
using namespace madrona::math;

namespace SimpleTaskgraph {

static std::mt19937 & randGen()
{
    //thread_local std::mt19937 rand_gen { std::random_device {}() };
    thread_local std::mt19937 rand_gen { 0 };

    return rand_gen;
}

static Vector3 randomPosition(const AABB &bounds)
{
    std::uniform_real_distribution<float> x_dist(bounds.pMin.x, bounds.pMax.x);
    std::uniform_real_distribution<float> y_dist(bounds.pMin.y, bounds.pMax.y);
    std::uniform_real_distribution<float> z_dist(bounds.pMin.z, bounds.pMax.z);
    
    return Vector3 {
        x_dist(randGen()),
        y_dist(randGen()),
        z_dist(randGen()),
    };
}

EnvInit generateEnvironmentInitialization()
{
    const int num_init_objs = 100;

    ObjectInit *objs_init = 
        (ObjectInit *)malloc(sizeof(ObjectInit) * num_init_objs);

    AABB world_bounds = {
        .pMin = Vector3 { -10, -10, 0, },
        .pMax = Vector3 { 10, 10, 10, },
    };

    std::uniform_real_distribution<float> angle_dist(0.f, M_PI); 
    for (int64_t i = 0; i < (int64_t)num_init_objs; i++) {
        Position rand_pos = randomPosition(world_bounds);
        Rotation rand_rot = Quat::angleAxis(angle_dist(randGen()),
            Vector3 { 0, 1, 0 });

        objs_init[i].initPosition = rand_pos;
        objs_init[i].initRotation = rand_rot;
    }

    return EnvInit {
        .worldBounds = world_bounds,
        .objsInit = objs_init,
        .numObjs = num_init_objs,
    };
}

}
