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

TEST(GJK, Solve4SimplexDuplicatePoint)
{
    Vector3 Y[4] {
        {0.814353108, 0.195752025, -0.698764443},
        {-0.784147143, 0.126484752, 0.701235533},
        {-0.784147143, 0.126484752, -0.698764443},
        {-0.784147143, 0.126484752, 0.701235533},
    };

    auto solve3 = gjkSolve3Simplex(Y[0], Y[1], Y[2]);
    auto solve4 = gjkSolve4Simplex(Y[0], Y[1], Y[2], Y[3]);

    EXPECT_LE(solve4.vLen2 - solve3.vLen2, 1e-5f);
}

TEST(GJK, Solve4SimplexAroundOrigin)
{
    Vector3 Y[4] {
        {0.793287277, 2.86326122, -0.700307727},
        {-0.794485092, -0.542466521, 0.699692249},
        {0.80550468, -0.536717057, -0.700307727},
        {-0.794485092, -0.542466521, -0.700307727},
    };

    auto solve_state = gjkSolve4Simplex(Y[0], Y[1], Y[2], Y[3]);
    EXPECT_LT(fabsf(solve_state.v.x), 1e-5f);
    EXPECT_LT(fabsf(solve_state.v.y), 1e-5f);
    EXPECT_LT(fabsf(solve_state.v.z), 1e-5f);
    EXPECT_LT(solve_state.vLen2, 1e-5f);
}
