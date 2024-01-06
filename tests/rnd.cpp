#include <gtest/gtest.h>

#include <madrona/rnd.hpp>

using namespace madrona;

struct RandomSplitTest : public testing::Test {
    RNG rngA;
    RNG rngB;

    void SetUp() override {
        RandKey init_key(5);

        rngA = RNG(rand::split_i(init_key, 0));
        rngB = RNG(rand::split_i(init_key, 1));
    }
};

TEST_F(RandomSplitTest, Uniform)
{
    for (int32_t i = 0; i < 100; i++) {
        float v_a = rngA.sampleUniform();
        float v_b = rngB.sampleUniform();
        EXPECT_GE(v_a, 0.f);
        EXPECT_LT(v_a, 1.f);
        EXPECT_GE(v_b, 0.f);
        EXPECT_LT(v_b, 1.f);

        // Note this isn't guaranteed but a collision is unlikely
        EXPECT_NE(v_a, v_b);
    }
}

TEST_F(RandomSplitTest, Int)
{
    int32_t num_matches = 0;
    for (int32_t i = 0; i < 100; i++) {
        int32_t v_a = rngA.sampleI32(1, 100);
        int32_t v_b = rngB.sampleI32(1, 100);
        EXPECT_GE(v_a, 1);
        EXPECT_LT(v_a, 100);
        EXPECT_GE(v_b, 1);
        EXPECT_LT(v_b, 100);

        if (v_a == v_b) {
            num_matches++;
        }
    }
    EXPECT_LT(num_matches, 3);
}

static inline void checkLimits(RNG &rng_a, RNG &rng_b, const int32_t low, const int32_t high, const int32_t num_tries)
{
    int32_t num_matches = 0;
    bool low_found_a = false;
    bool low_found_b = false;
    bool high_found_a = false;
    bool high_found_b = false;
    for (int32_t i = 0; i < num_tries; i++) {
        int32_t v_a = rng_a.sampleI32(low, high);
        int32_t v_b = rng_b.sampleI32(low, high);

        EXPECT_GE(v_a, low);
        EXPECT_LT(v_a, high);
        EXPECT_GE(v_b, low);
        EXPECT_LT(v_b, high);

        if (v_a == v_b) {
            num_matches++;
        }

        if (v_a == low) {
            low_found_a = true;
        }

        if (v_b == low) {
            low_found_b = true;
        }

        if (v_a == high) {
            high_found_a = true;
        }

        if (v_b == high) {
            high_found_b = true;
        }
    }

    EXPECT_LT(num_matches, num_tries / 10);

    EXPECT_TRUE(low_found_a);
    EXPECT_TRUE(low_found_b);

    EXPECT_FALSE(high_found_a);
    EXPECT_FALSE(high_found_b);
}

TEST_F(RandomSplitTest, IntPosLimits)
{
    checkLimits(rngA, rngB, 2, 20, 100);
}

TEST_F(RandomSplitTest, IntNegPosLimits)
{
    checkLimits(rngA, rngB, -20, 2, 100);
}

TEST_F(RandomSplitTest, IntNegLimits)
{
    checkLimits(rngA, rngB, -30, -10, 100);
}

TEST_F(RandomSplitTest, IntBigNegLimits)
{
    checkLimits(rngA, rngB, 0x8000'0000_i32, 0x8000'0000_i32 + 20, 100);
}

TEST(RandomInit, DifferentSeeds)
{
    RNG a(5);
    RNG b(10);

    for (int i = 0; i < 100; i++) {
        float v_a = a.sampleUniform();
        float v_b = b.sampleUniform();
        EXPECT_NE(v_a, v_b);
    }
}
