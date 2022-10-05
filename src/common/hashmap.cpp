/*
 * Copyright 2021-2022 Brennan Shacklett and contributors
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 */
#include <madrona/hashmap.hpp>
#include <madrona/utils.hpp>
#include <madrona/crash.hpp>
#include <madrona/heap_array.hpp>

#include <cassert>

namespace madrona {

void StaticMapHelper::buildMap(uint32_t *key_storage, uint32_t *value_storage,
                               uint32_t storage_size,
                               const IntegerMapPair *inputs,
                               uint32_t num_inputs, uint32_t shift_idx,
                               uint32_t constant_idx)
{
    uint32_t capacity = utils::int32NextPow2(num_inputs);
    capacity = std::max(capacity, 16u);
    assert(capacity <= storage_size);

    for (int i = 0; i < (int)capacity; i++) {
        key_storage[i] = invalidKey;
    }

    // This shift is used in map(), multiplicative hashing
    // so w - m where w = 32 and capacity = 2 ** m
    uint32_t capacity_shift = 32u - utils::int32Log2(capacity);
    uint32_t multiplier = 0x7feb352du;

    constexpr int max_attempts = 32;

    HeapArray<bool> used(capacity);
    auto clearUsed = [&]() {
        for (int i = 0; i < (int)capacity; i++) {
            used[i] = false;
        }
    };

    int attempts;
    for (attempts = 0; attempts < max_attempts; attempts++) {
        clearUsed();

        // the hash table needs to not map valid inputs into the metadata
        // slots in the first cacheline
        used[shift_idx] = true;
        used[constant_idx] = true;

        bool success = true;

        for (int i = 0; i < (int)num_inputs; i++) {
            uint32_t idx = map(inputs[i].key, capacity_shift, multiplier);

            if (used[idx]) {
                // Update the constant by rehashing and forcing to be odd
                // Probably a better way to do this
                multiplier = utils::int32Hash(multiplier) | 1;
                success = false;
                break;
            }
             
            assert(inputs[i].key != invalidKey);

            key_storage[idx] = inputs[i].key;
            value_storage[idx] = inputs[i].value;
        }

        if (success == true) {
            break;
        }
    }

    if (attempts == max_attempts) {
        FATAL("StaticIntegerMap: failed to find perfect hash");
    }

    // Store metadata info in appropriate slots. Stored twice to avoid
    // unnecessary cache line loads in cases where key or value is not needed
    key_storage[shift_idx] = capacity_shift;
    key_storage[constant_idx] = multiplier;
    value_storage[shift_idx] = capacity_shift;
    value_storage[constant_idx] = multiplier;

}

}
