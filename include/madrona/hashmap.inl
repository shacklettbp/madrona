/*
 * Copyright 2021-2022 Brennan Shacklett and contributors
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 */
#pragma once

#include <madrona/utils.hpp>

namespace madrona {

uint32_t StaticMapHelper::map(uint32_t key, uint32_t shift,
                              uint32_t constant)
{
    // Simple multiplicative hashing with a configurable k
    uint32_t hash = key;
    hash ^= hash >> shift;
    uint32_t idx = (hash * constant) >> shift;

    return idx;
}

template <uint32_t maxN>
StaticIntegerMap<maxN>::StaticIntegerMap(const IntegerMapPair *inputs,
                                         uint32_t num_inputs)
    : keys_(),
      values_()
{
    static_assert(maxN > 0 && utils::isPower2(maxN));

    std::array<bool, maxN> scratch_storage;

    StaticMapHelper::buildMap(keys_.data(), values_.data(),
                              scratch_storage.data(), maxN, inputs,
                              num_inputs, shift_idx_, constant_idx_);
}

template <uint32_t maxN>
bool StaticIntegerMap<maxN>::exists(uint32_t key) const
{
    uint32_t idx = StaticMapHelper::map(key, keys_[shift_idx_],
                                        keys_[constant_idx_]);

    if (idx < 2) return false;

    return key == keys_[idx];
}

template <uint32_t maxN>
uint32_t StaticIntegerMap<maxN>::operator[](uint32_t key) const
{
    uint32_t idx = StaticMapHelper::map(key, values_[shift_idx_],
                                        values_[constant_idx_]);
    return values_[idx];
}

template <uint32_t maxN>
Optional<uint32_t> StaticIntegerMap<maxN>::lookup(uint32_t key) const
{
    uint32_t idx = StaticMapHelper::map(key, keys_[shift_idx_],
                                        keys_[constant_idx_]);

    if (idx < 2 || key != keys_[idx]) {
        return Optional<uint32_t>::none();
    }

    return Optional<uint32_t>::make(values_[idx]);
}

template <uint32_t maxN>
constexpr uint32_t StaticIntegerMap<maxN>::numFree()
{
    return maxN - 2;
}

}
