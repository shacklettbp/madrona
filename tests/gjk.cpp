/*
 * Copyright 2021-2022 Brennan Shacklett and contributors
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 */
#include <gtest/gtest.h>

#define MADRONA_GJK_DEBUG
#include "../src/physics/gjk.hpp"

#include <array>

using namespace madrona;
using namespace madrona::geo;
using namespace madrona::math;

TEST(GJK, Solve4Simplex1)
{
    Vector3 Y[4] {
        {1.1680119, 2.82056189, 0.699643314},
        {0.189007372, -0.435441136, -0.700356662},
        {-1.34322929, 0.0252667665, 0.699643314},
        {1.1680119, 2.82056189, -0.700356662},
    };

    gjkSolve4Simplex(Y[3], Y[2], Y[1], Y[0]);
    
}

TEST(GJK, Solve4Simplex2)
{
    Vector3 Y[4] {
        {0.114758693, -0.194909215, 0.699410498},
        {0.114758693, -0.194909215, -0.700589478},
        {0.114758693, -0.194909215, 0.699410498},
        {-1.32535255, 2.88504028, -0.700589478},
    };

    gjkSolve4Simplex(Y[0], Y[1], Y[2], Y[3]);
}
