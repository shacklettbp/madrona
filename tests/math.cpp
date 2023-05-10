/*
 * Copyright 2021-2022 Brennan Shacklett and contributors
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 */
#include <gtest/gtest.h>

#include <madrona/math.hpp>

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

using namespace madrona;
using namespace madrona::math;

static inline void expectQuat(Quat a, Quat b)
{
    EXPECT_NEAR(a.w, b.w, 1e-4f);
    EXPECT_NEAR(a.x, b.x, 1e-4f);
    EXPECT_NEAR(a.y, b.y, 1e-4f);
    EXPECT_NEAR(a.z, b.z, 1e-4f);
}

TEST(Quaternions, Math)
{
    auto q1 = Quat::angleAxis(0, {0, 1, 0});
    {
        SCOPED_TRACE("");
        expectQuat(q1, { 1, 0, 0, 0 });
    }

    auto q2 = Quat::angleAxis(toRadians(45), {0, 1, 0});
    {
        SCOPED_TRACE("");
        expectQuat(q2, { 0.9238795, 0, 0.3826834, 0 });
    }

    auto q3 = Quat::angleAxis(toRadians(45), {1, 0, 0});
    {
        SCOPED_TRACE("");
        expectQuat(q3, { 0.9238795, 0.3826834, 0, 0 });
    }

    auto m1 = q2 * q3;
    {
        SCOPED_TRACE("");
        expectQuat(m1, { 0.853553, 0.353553, 0.353553, -0.146447 });
    }
}

TEST(Mat3x4, Math)
{
}
