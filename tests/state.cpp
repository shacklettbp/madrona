/*
 * Copyright 2021-2022 Brennan Shacklett and contributors
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 */
#include <gtest/gtest.h>

#include <madrona/state.hpp>

#include <array>

using namespace madrona;

struct Component1 {
    uint32_t v;
};

struct alignas(16) Component2 {
    uint32_t x;
    uint32_t y;
    uint32_t z;
};

struct Component3 {
    unsigned char v;
};

struct ComponentBig {
    uint32_t x[1000];
};

struct Archetype1 : Archetype<Component1> {};
struct Archetype2 : Archetype<Component1, Component2, Component3> {};
struct Archetype3 : Archetype<ComponentBig> {};

TEST(State, Indexing)
{
    int num_entities = 1'000'000;

    StateManager state;
    StateCache cache;
    state.registerComponent<Component1>();
    state.registerArchetype<Archetype1>();

    Entity init_e = state.makeEntityNow<Archetype1>(cache);

    EXPECT_TRUE(state.get<Component1>(init_e).valid());

    DynArray<Entity> entities(0);

    for (int i = 0; i < num_entities; i++) {
        entities.push_back(state.makeEntityNow<Archetype1>(cache));
    }

    for (int i = 0; i < num_entities; i++) {
        EXPECT_TRUE(state.getLoc(entities[i]).valid());
    }

    for (int i = 0; i < num_entities; i += 10) {
        state.destroyEntityNow(cache, entities[i]);
    }

    for (int i = 0; i < num_entities; i++) {
        EXPECT_EQ(state.getLoc(entities[i]).valid(), i % 10 != 0);
    }

    DynArray<Entity> new_entities(0);
    DynArray<Entity> deleted_entities(0);
    for (int i = 0; i < num_entities; i++) {
        if (i % 10 == 0) {
            deleted_entities.push_back(entities[i]);
        } else {
            new_entities.push_back(entities[i]);
        }
    }

    num_entities *= 2;

    entities = std::move(new_entities);
    while ((int)entities.size() != num_entities) {
        entities.push_back(state.makeEntityNow<Archetype1>(cache));
    }

    for (const auto &e : deleted_entities) {
        EXPECT_FALSE(state.getLoc(e).valid());
    }

    for (int i = 0; i < num_entities; i++) {
        auto idx = state.getLoc(entities[i]);
        EXPECT_TRUE(idx.valid());
        auto res = state.get<Component1>(idx);
        EXPECT_TRUE(res.valid());

        res.value().v = i;
    }

    for (int i = 0; i < num_entities; i++) {
        auto res = state.get<Component1>(entities[i]);
        EXPECT_TRUE(res.valid());
        EXPECT_EQ(res.value().v, i);
    }

    for (int i = 0; i < num_entities; i++) {
        if (i % 5 == 0) {
            auto res = state.get<Component1>(entities[i]);
            EXPECT_TRUE(res.valid());
            EXPECT_EQ(res.value().v, i );

            state.destroyEntityNow(cache, entities[i]);
            auto loc = state.getLoc(entities[i]);
            EXPECT_FALSE(loc.valid());
        } else {
            auto res = state.get<Component1>(entities[i]);
            EXPECT_TRUE(res.valid());
            EXPECT_EQ(res.value().v, i);
        }
    }

    for (int i = 0; i < num_entities; i++) {
        if (i % 5 == 0) { continue; }

        auto res = state.get<Component1>(entities[i]);
        EXPECT_TRUE(res.valid());
        EXPECT_EQ(res.value().v, i);
    }
}

TEST(State, MultiColumn)
{
    StateManager state;
    StateCache cache;
    state.registerComponent<Component1>();
    state.registerComponent<Component2>();
    state.registerComponent<Component3>();
    state.registerArchetype<Archetype2>();

    int num_entities = 1'001;

    DynArray<Entity> entities(num_entities);
    for (int i = 0; i < num_entities; i++) {
        entities.push_back(state.makeEntityNow<Archetype2>(cache));

        Loc loc = state.getLoc(entities[i]);
        EXPECT_TRUE(loc.valid());

        Component1 &first = state.get<Component1>(loc).value();

        first.v = i;

        Component2 &second = state.get<Component2>(loc).value();
        EXPECT_TRUE((uint64_t)&second % std::alignment_of_v<Component2> == 0);

        second.x = i * 2;
        second.y = i * 2 + 1;
        second.z = -i;

        Component3 &third = state.get<Component3>(loc).value();
        third.v = i % 256;
    }

    for (int i = 0; i < num_entities; i += 10) {
        state.destroyEntityNow(cache, entities[i]);
    }

    for (int i = 0; i < num_entities; i += 10) {
        entities[i] = state.makeEntityNow<Archetype2>(cache);
    }

    for (int i = 0; i < num_entities; i++) {
        if (i % 10 == 0) {
            continue;
        }

        Loc loc = state.getLoc(entities[i]);
        EXPECT_TRUE(loc.valid());

        Component1 &first = state.get<Component1>(loc).value();
        Component2 &second = state.get<Component2>(loc).value();
        Component3 &third = state.get<Component3>(loc).value();

        EXPECT_EQ(first.v, i);
        EXPECT_EQ(second.x, i * 2);
        EXPECT_EQ(second.y, i * 2 + 1);
        EXPECT_EQ(second.z, -i);

        EXPECT_EQ(third.v, i % 256);
    }
}

TEST(State, DeleteMany)
{
    StateManager state;
    StateCache cache;
    state.registerComponent<ComponentBig>();
    state.registerArchetype<Archetype3>();

    int num_entities = 1'000'000;

    DynArray<Entity> entities(num_entities);
    for (int i = 0; i < num_entities; i++) {
        entities.push_back(
            state.makeEntityNow<Archetype3>(cache));

        Loc loc = state.getLoc(entities[i]);

        ComponentBig &big = state.get<ComponentBig>(loc).value();
        big.x[0] = i;
    }

    for (int i = 2; i < num_entities - 2; i++) {
        state.destroyEntityNow(cache, entities[i]);
    }

    for (int i = 0; i < 2; i++) {
        ComponentBig &big = state.get<ComponentBig>(entities[i]).value();
        EXPECT_EQ(big.x[0], i);
    }

    for (int i = num_entities - 2; i < num_entities; i++) {
        ComponentBig &big = state.get<ComponentBig>(entities[i]).value();
        EXPECT_EQ(big.x[0], i);
    }
}

TEST(State, Reset)
{
    StateManager state;
    StateCache cache;
    state.registerComponent<Component1>();
    state.registerComponent<Component2>();
    state.registerComponent<Component3>();
    state.registerArchetype<Archetype1>();
    state.registerArchetype<Archetype2>();

    int num_entities = 100'000;

    DynArray<Entity> initial_entities(num_entities);
    for (int i = 0; i < num_entities; i++) {
        initial_entities.push_back(state.makeEntityNow<Archetype1>(cache));
    }

    for (Entity e : initial_entities) {
        EXPECT_TRUE(state.get<Component1>(e).valid());
    }

    state.clear<Archetype1>(cache);

    for (Entity e : initial_entities) {
        EXPECT_FALSE(state.get<Component1>(e).valid());
    }

    DynArray<Entity> new_entities(num_entities);
    for (int i = 0; i < num_entities; i++) {
        new_entities.push_back(state.makeEntityNow<Archetype1>(cache));
    }

    for (Entity e : initial_entities) {
        EXPECT_FALSE(state.get<Component1>(e).valid());
    }

    for (Entity e : new_entities) {
        EXPECT_TRUE(state.get<Component1>(e).valid());
    }
}
