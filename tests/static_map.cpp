/*
 * Copyright 2021-2024 Brennan Shacklett and contributors
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
public:
    // Seems to max out currently at 80 elements (81 can't find perfect hash)
    static constexpr int numElems = 90;
    static constexpr int numBytes = 1024;
    using MapT = StaticIntegerMap<numBytes>;

    void SetUp() override {
        IntegerMapPair inputs[numElems];
        for (uint32_t i = 0; i < numElems; i++) {
            inputs[i] = IntegerMapPair {
                .key = i,
                .value = i,
            };
        }
        testMap = new StaticIntegerMap<numBytes>(inputs, numElems);
    }

    void TearDown() override { delete testMap; }

    StaticIntegerMap<numBytes> *testMap;
};

TEST_F(StaticIntegerMapTest, SimpleTest) {
    EXPECT_EQ(testMap->capacity(), 126);
    static_assert(sizeof(*testMap) == numBytes);

    for (uint32_t i = 0; i < numElems; i++) {
        EXPECT_TRUE(testMap->exists(i));
        EXPECT_EQ(*testMap->lookup(i), i);
    }

    for (uint32_t i = numElems; i < numElems * 2; i++) {
        EXPECT_FALSE(testMap->exists(i));
    }
}
