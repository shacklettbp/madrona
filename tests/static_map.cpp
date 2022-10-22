/*
 * Copyright 2021-2022 Brennan Shacklett and contributors
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 */
#include <gtest/gtest.h>

#include <array>
#include <madrona/hashmap.hpp>

using namespace madrona;

class StaticIntegerMapTest : public testing::Test {
protected:
    void SetUp() override {
        for (uint32_t i = 0; i < num_elems_; i++) {
            integer_map_array_[i] = IntegerMapPair{.key = i, .value = i};
        }
        static_integer_map_ = new StaticIntegerMap<max_n_> {
            integer_map_array_.data(), (uint32_t)integer_map_array_.size()};
    }

    void TearDown() override { delete static_integer_map_; }

    // Seems to max out currently at 80 elements (81 can't find perfect hash)
    static constexpr int num_elems_ = 80;
    static constexpr int max_n_ = 128;
    std::array<IntegerMapPair, num_elems_> integer_map_array_;
    StaticIntegerMap<max_n_> *static_integer_map_;
};

TEST_F(StaticIntegerMapTest, SimpleTest) {
    EXPECT_EQ(static_integer_map_->numFree(), max_n_ - 2);
    for (uint32_t i = 0; i < num_elems_; i++) {
        EXPECT_TRUE(static_integer_map_->exists(i));
        EXPECT_EQ(*static_integer_map_->lookup(i), i);
    }
}
