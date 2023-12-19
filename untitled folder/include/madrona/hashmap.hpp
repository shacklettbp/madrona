/*
 * Copyright 2021-2022 Brennan Shacklett and contributors
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 */
#pragma once

#include <array>
#include <cstdint>
#include <utility>
#include <madrona/optional.hpp>

namespace madrona {

struct IntegerMapPair {
    uint32_t key;
    uint32_t value;
};

struct StaticMapHelper {
    static void buildMap(uint32_t *key_storage, uint32_t *value_storage,
                         bool *scratch_storage, uint32_t storage_size,
                         const IntegerMapPair *inputs,
                         uint32_t num_inputs, uint32_t shift_idx,
                         uint32_t constant_idx);

    static inline uint32_t map(uint32_t key, uint32_t shift,
                               uint32_t constant);

    static constexpr uint32_t invalidKey = ~0u;
};

template <uint32_t maxN>
class StaticIntegerMap {
public:
    StaticIntegerMap(const IntegerMapPair *inputs, uint32_t num_inputs);

    inline bool exists(uint32_t key) const;
    inline uint32_t operator[](uint32_t key) const;

    inline Optional<uint32_t> lookup(uint32_t key) const;

    constexpr static uint32_t numFree();

private:
    static constexpr uint32_t shift_idx_ = 0;
    static constexpr uint32_t constant_idx_ = 1;

    alignas(64) std::array<uint32_t, maxN> keys_;
    alignas(64) std::array<uint32_t, maxN> values_;
};

}

#include "hashmap.inl"
