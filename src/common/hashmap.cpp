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

#include <algorithm>
#include <cassert>

namespace madrona {

static bool searchForPerfectHash(const IntegerMapPair *inputs,
                                 uint32_t num_inputs,
                                 uint32_t capacity,
                                 uint32_t shift_idx,
                                 uint32_t constant_idx,
                                 uint32_t capacity_shift,
                                 bool *slot_tracker,
                                 uint32_t *multiplier)
{
    auto clearUsed = [&]() {
        for (int i = 0; i < (int)capacity; i++) {
            slot_tracker[i] = false;
        }
    };

    uint32_t cur_multiplier = *multiplier;

    constexpr int max_attempts = 32;
    for (int attempts = 0; attempts < max_attempts; attempts++) {
        clearUsed();

        // the hash table needs to not map valid inputs into the metadata slots 
        slot_tracker[shift_idx] = true;
        slot_tracker[constant_idx] = true;
    
        bool success = true;
    
        for (int i = 0; i < (int)num_inputs; i++) {
            uint32_t idx = StaticMapHelper::map(inputs[i].key, capacity_shift,
                                                cur_multiplier);
            assert(idx < capacity);
    
            if (slot_tracker[idx]) {
                // Update the constant by rehashing and forcing to be odd
                // Probably a better way to do this
                cur_multiplier = utils::int32Hash(cur_multiplier) | 1;
                success = false;
                break;
            }
             
            assert(inputs[i].key != StaticMapHelper::invalidKey);
    
            slot_tracker[idx] = true;
        }
    
        if (success == true) {
            *multiplier = cur_multiplier;
            return true;
        }
    }

    return false;
}

void StaticMapHelper::buildMap(uint32_t *key_storage, uint32_t *value_storage,
                               bool *scratch_storage, uint32_t storage_size,
                               const IntegerMapPair *inputs,
                               uint32_t num_inputs, uint32_t shift_idx,
                               uint32_t constant_idx)
{
    // Capacity needs to be at least num_inputs + 2 for the metadata slots
    uint32_t capacity = utils::int32NextPow2(num_inputs + 2);
    capacity = std::max(capacity, 16u);
    assert(capacity <= storage_size);

    // This shift is used in map(), multiplicative hashing
    // so w - m where w = 32 and capacity = 2 ** m
    uint32_t capacity_shift = 32u - utils::int32Log2(capacity);
    uint32_t multiplier = 0x7feb352du;

    while (!searchForPerfectHash(inputs, num_inputs, capacity, shift_idx,
            constant_idx, capacity_shift, scratch_storage, &multiplier)) {
        capacity *= 2;
        capacity_shift -= 1;

        if (capacity > storage_size) {
            FATAL("StaticIntegerMap: failed to find perfect hash");
        }
    }

    // Given the hash function, actually save the table
    // First, erase the table
    for (uint32_t i = 0; i < capacity; i++) {
        key_storage[i] = invalidKey;
    }

    for (int i = 0; i < (int)num_inputs; i++) {
        const IntegerMapPair &input = inputs[i];
        uint32_t idx = map(input.key, capacity_shift, multiplier);
        key_storage[idx] = input.key;
        value_storage[idx] = input.value;
    }

    // Store metadata info in appropriate slots. Stored twice to avoid
    // unnecessary cache line loads in cases where key or value is not needed
    key_storage[shift_idx] = capacity_shift;
    key_storage[constant_idx] = multiplier;
    value_storage[shift_idx] = capacity_shift;
    value_storage[constant_idx] = multiplier;
}

}
