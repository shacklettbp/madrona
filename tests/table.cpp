#include <gtest/gtest.h>

#include <madrona/table.hpp>

#include <array>

using namespace madrona;
using namespace std;

TEST(Table, Indexing)
{
    int num_entities = 1'000'000;

    array<TypeInfo, 1> cols {
        TypeInfo {
            alignof(uint32_t),
            sizeof(uint32_t),
        },
    };
    Table tbl(cols.data(), cols.size(), 0);

    Entity init_e = tbl.addRow();

    EXPECT_TRUE(tbl.getLoc(init_e).valid());

    vector<Entity> entities;

    for (int i = 0; i < num_entities; i++) {
        entities.push_back(tbl.addRow());
    }

    for (int i = 0; i < num_entities; i++) {
        EXPECT_TRUE(tbl.getLoc(entities[i]).valid());
    }

    for (int i = 0; i < num_entities; i += 10) {
        tbl.removeRow(entities[i]);
    }

    for (int i = 0; i < num_entities; i++) {
        EXPECT_EQ(tbl.getLoc(entities[i]).valid(), i % 10 != 0);
    }

    vector<Entity> new_entities;
    vector<Entity> deleted_entities;
    for (int i = 0; i < num_entities; i++) {
        if (i % 10 == 0) {
            deleted_entities.push_back(entities[i]);
        } else {
            new_entities.push_back(entities[i]);
        }
    }

    num_entities *= 2;

    entities = move(new_entities);
    while ((int)entities.size() != num_entities) {
        entities.push_back(tbl.addRow());
    }

    for (const auto &e : deleted_entities) {
        EXPECT_FALSE(tbl.getLoc(e).valid());
    }

    for (int i = 0; i < num_entities; i++) {
        auto idx = tbl.getLoc(entities[i]);
        EXPECT_TRUE(idx.valid());
        auto addr = (uint32_t *)tbl.getValue(1, idx);

        *addr = i;
    }

    for (int i = 0; i < num_entities; i++) {
        auto idx = tbl.getLoc(entities[i]);
        auto addr = (uint32_t *)tbl.getValue(1, idx);
        EXPECT_EQ(*addr, i);
    }

    for (int i = 0; i < num_entities; i++) {
        if (i % 5 == 0) {
            auto idx = tbl.getLoc(entities[i]);
            EXPECT_TRUE(idx.valid());
            auto addr = (uint32_t *)tbl.getValue(1, idx);
            EXPECT_EQ(*addr, i );

            tbl.removeRow(entities[i]);
            idx = tbl.getLoc(entities[i]);
            EXPECT_FALSE(idx.valid());
        } else {
            auto idx = tbl.getLoc(entities[i]);
            EXPECT_TRUE(idx.valid());
            auto addr = (uint32_t *)tbl.getValue(1, idx);
            EXPECT_EQ(*addr, i);
        }
    }

    for (int i = 0; i < num_entities; i++) {
        if (i % 5 == 0) { continue; }

        auto idx = tbl.getLoc(entities[i]);
        EXPECT_TRUE(idx.valid());
        auto addr = (uint32_t *)tbl.getValue(1, idx);
        EXPECT_EQ(*addr, i);
    }
}

TEST(Table, MultiColumn)
{
    struct alignas(16) V {
        uint32_t x;
        uint32_t y;
        uint32_t z;
    };

    int num_entities = 1'001;
    array cols {
        TypeInfo {
            alignof(uint32_t),
            sizeof(uint32_t),
        },
        TypeInfo {
            alignof(V),
            sizeof(V),
        },
        TypeInfo {
            alignof(unsigned char),
            sizeof(unsigned char),
        },
    };
    Table tbl(cols.data(), cols.size(), 0);

    std::vector<Entity> entities;
    for (int i = 0; i < num_entities; i++) {
        entities.push_back(tbl.addRow());

        auto idx = tbl.getLoc(entities[i]);

        auto first = (uint32_t *)tbl.getValue(1, idx);

        *first = i;
        auto second = (V *)tbl.getValue(2, idx);
        EXPECT_TRUE((uint64_t)second % std::alignment_of_v<V> == 0);

        second->x = i * 2;
        second->y = i * 2 + 1;
        second->z = -i;

        auto third = (unsigned char *)tbl.getValue(3, idx);
        
        *third = i % 256;
    }

    for (int i = 0; i < num_entities; i += 10) {
        tbl.removeRow(entities[i]);
    }

    for (int i = 0; i < num_entities; i += 10) {
        entities[i] = tbl.addRow();
    }

    for (int i = 0; i < num_entities; i++) {
        if (i % 10 == 0) {
            continue;
        }

        auto idx = tbl.getLoc(entities[i]);

        auto first = (uint32_t *)tbl.getValue(1, idx);
        auto second = (V *)tbl.getValue(2, idx);
        auto third = (unsigned char *)tbl.getValue(3, idx);

        EXPECT_EQ(*first, i);
        EXPECT_EQ(second->x, i * 2);
        EXPECT_EQ(second->y, i * 2 + 1);
        EXPECT_EQ(second->z, -i);

        EXPECT_EQ(*third, i % 256);
    }
}

TEST(Table, DeleteMany)
{
    struct V {
        uint32_t x[1000];
    };
    array cols {
        TypeInfo {
            alignof(V),
            sizeof(V),
        }
    };
    Table tbl(cols.data(), cols.size(), 0);

    int num_entities = 1'000'000;

    vector<Entity> entities;
    for (int i = 0; i < num_entities; i++) {
        entities.push_back(tbl.addRow());

        auto idx = tbl.getLoc(entities[i]);

        auto ptr = (uint32_t *)tbl.getValue(1, idx);
        *ptr = i;
    }

    for (int i = 2; i < num_entities - 2; i++) {
        tbl.removeRow(entities[i]);
    }

    for (int i = 0; i < 2; i++) {
        auto ptr = (uint32_t *)tbl.getValue(1, tbl.getLoc(entities[i]));
        EXPECT_EQ(*ptr, i);
    }

    for (int i = 0; i < 2; i++) {
        auto ptr = (uint32_t *)tbl.getValue(1, tbl.getLoc(entities[i]));
        EXPECT_EQ(*ptr, i);
    }
    for (int i = num_entities - 2; i < num_entities; i++) {
        auto ptr = (uint32_t *)tbl.getValue(1, tbl.getLoc(entities[i]));
        EXPECT_EQ(*ptr, i);
    }
}
