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

uint32_t StaticMapHelper::map(uint32_t key, uint32_t shift,
                              uint32_t constant)
{
    // Simple multiplicative hashing with a configurable k
    uint32_t hash = key;
    hash ^= hash >> shift;
    uint32_t idx = (hash * constant) >> shift;

    return idx;
}

template <uint32_t num_bytes>
StaticIntegerMap<num_bytes>::StaticIntegerMap(const IntegerMapPair *inputs,
                                         uint32_t num_inputs)
    : keys_(),
      values_()
{
    static_assert(num_bytes > 0);
    static_assert(num_bytes % MADRONA_CACHE_LINE == 0);
    static_assert(sizeof(*this) == num_bytes);

    std::array<bool, num_bytes> scratch_storage;

    StaticMapHelper::buildMap(keys_.data(), values_.data(),
                              scratch_storage.data(), num_bytes, inputs,
                              num_inputs, shift_idx_, constant_idx_);
}

template <uint32_t num_bytes>
bool StaticIntegerMap<num_bytes>::exists(uint32_t key) const
{
    uint32_t idx = StaticMapHelper::map(key, keys_[shift_idx_],
                                        keys_[constant_idx_]);

    if (idx >= capacity()) return false;

    return key == keys_[idx];
}

template <uint32_t num_bytes>
uint32_t StaticIntegerMap<num_bytes>::operator[](uint32_t key) const
{
    uint32_t idx = StaticMapHelper::map(key, values_[shift_idx_],
                                        values_[constant_idx_]);
    return values_[idx];
}

template <uint32_t num_bytes>
Optional<uint32_t> StaticIntegerMap<num_bytes>::lookup(uint32_t key) const
{
    uint32_t idx = StaticMapHelper::map(key, keys_[shift_idx_],
                                        keys_[constant_idx_]);

    if (idx >= capacity() || key != keys_[idx]) {
        return Optional<uint32_t>::none();
    }

    return Optional<uint32_t>::make(values_[idx]);
}

template <uint32_t num_bytes>
constexpr uint32_t StaticIntegerMap<num_bytes>::capacity()
{
    return numInts() - 2;
}

template <uint32_t num_bytes>
constexpr uint32_t StaticIntegerMap<num_bytes>::numInts()
{
    return num_bytes / sizeof(uint32_t) / 2;
}

}
