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
#include <madrona/macros.hpp>
#include <madrona/optional.hpp>

namespace madrona {

struct IntegerMapPair {
    uint32_t key;
    uint32_t value;
};

template <uint32_t num_bytes>
class StaticIntegerMap {
public:
    StaticIntegerMap(const IntegerMapPair *inputs, uint32_t num_inputs);

    inline bool exists(uint32_t key) const;
    inline uint32_t operator[](uint32_t key) const;

    inline Optional<uint32_t> lookup(uint32_t key) const;

    constexpr static inline uint32_t capacity();

private:
    constexpr static inline uint32_t numInts();

    static inline constexpr uint32_t shift_idx_ = numInts() - 2;
    static inline constexpr uint32_t constant_idx_ = numInts() - 1;

    alignas(64) std::array<uint32_t, numInts()> keys_;
    alignas(64) std::array<uint32_t, numInts()> values_;
};

}

#include "hashmap.inl"
