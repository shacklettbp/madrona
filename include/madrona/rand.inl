#include <madrona/macros.hpp>
#include <madrona/utils.hpp>

namespace madrona {
namespace rand {

constexpr RandKey initKey(uint32_t seed, uint32_t seed_upper)
{
    return split_i(RandKey { seed, seed_upper }, 0);
}

// This implementation is based on JAX's threefry splitting implementation.
// The number of rounds is set to 20, for a safe margin over the currently
// known lower limit, 13. 20 is also the default in the original authors'
// threefry implementation:
// https://github.com/DEShawResearch/random123/blob/main/include/Random123/threefry.h
//
// Original copyright / license:
// Copyright 2019 The JAX Authors.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// 
//     http://www.apache.org/licenses/LICENSE-2.0
// 
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
constexpr RandKey split_i(RandKey src, uint32_t idx, uint32_t idx_upper)
{
    // Rotation distances specified by the Threefry2x32 algorithm.
    uint32_t rotations[8] = {13, 15, 26, 6, 17, 29, 16, 24};
    uint32_t x[2];
    uint32_t ks[3];
    
    // 0x1BD11BDA is a parity constant specified by the ThreeFry2x32 algorithm.
    ks[2] = 0x1BD11BDA;
    
    ks[0] = src.a;
    x[0] = idx;
    ks[2] = ks[2] ^ src.a;
    
    ks[1] = src.b;
    x[1] = idx_upper;
    ks[2] = ks[2] ^ src.b;
    
    auto rotate_left = [](uint32_t v, uint32_t distance) {
        return (v << distance) | (v >> (32 - distance));
    };
    
    // Performs a single round of the Threefry2x32 algorithm, with a rotation
    // amount 'rotation'.
    auto round = [&](uint32_t* v, uint32_t rotation) {
        v[0] += v[1];
        v[1] = rotate_left(v[1], rotation);
        v[1] ^= v[0];
    };
    
    // There are no known statistical flaws with 13 rounds of Threefry2x32.
    // We are conservative and use 20 rounds.
    x[0] = x[0] + ks[0];
    x[1] = x[1] + ks[1];
MADRONA_UNROLL
    for (int i = 0; i < 4; ++i) {
        round(x, rotations[i]);
    }
    
    x[0] = x[0] + ks[1];
    x[1] = x[1] + ks[2] + 1u;
MADRONA_UNROLL
    for (int i = 4; i < 8; ++i) {
        round(x, rotations[i]);
    }
    
    x[0] = x[0] + ks[2];
    x[1] = x[1] + ks[0] + 2u;
MADRONA_UNROLL
    for (int i = 0; i < 4; ++i) {
        round(x, rotations[i]);
    }
    
    x[0] = x[0] + ks[0];
    x[1] = x[1] + ks[1] + 3u;
MADRONA_UNROLL
    for (int i = 4; i < 8; ++i) {
        round(x, rotations[i]);
    }
    
    x[0] = x[0] + ks[1];
    x[1] = x[1] + ks[2] + 4u;
MADRONA_UNROLL
    for (int i = 0; i < 4; ++i) {
        round(x, rotations[i]);
    }

    RandKey out;
    out.a = x[0] + ks[2];
    out.b = x[1] + ks[0] + 5u;

    return out;
}

constexpr uint32_t bits32(RandKey k)
{
    return k.a ^ k.b;
}

constexpr uint64_t bits64(RandKey k)
{
    return ((uint64_t)k.b << 32_u64) | (uint64_t)k.a;
}

constexpr int32_t sampleI32(RandKey k, int32_t a, int32_t b)
{
    uint32_t s = (uint32_t)(b - a);

    // Lemire, Fast Random Number Generation in an Interval.
    // Algorithm 5. This is probably non-ideal for GPU but is unbiased.
    uint32_t x = bits32(k);

#ifdef MADRONA_GPU_MODE
    uint32_t l = x * s;
    uint32_t h = __umulhi(x, s);
#else
    uint32_t l, h;
    {
        uint64_t tmp = (uint64_t)x * (uint64_t)s;
        l = (uint32_t)tmp;
        h = (uint32_t)(tmp >> 32);
    }
#endif

    if (l < s) [[unlikely]] {
        // 2^32 % s == (2^32 - s) % s == -s % s
        uint32_t t = (0_u32 - s) % s;

        while (l < t) {
            // This might be suspect: reusing k but we're rejecting the random
            // number k generated, so maybe it's fine...?
            k = split_i(k, 0);
            x = bits32(k);

#ifdef MADRONA_GPU_MODE
            l = x * s;
            h = __umulhi(x, s);
#else
            {
                uint64_t tmp = (uint64_t)x * (uint64_t)s;
                l = (uint32_t)tmp;
                h = (uint32_t)(tmp >> 32);
            }
#endif
        }
    }

    return (int32_t)h + a;
}

constexpr int32_t sampleI32Biased(RandKey k, int32_t a, int32_t b)
{
    uint32_t s = (uint32_t)(b - a);
    uint32_t x = bits32(k);

    return utils::u32mulhi(x, s);
}

constexpr float sampleUniform(RandKey k)
{
    return bitsToFloat01(bits32(k));
}

constexpr bool sampleBool(RandKey k)
{
    uint32_t bits = bits32(k);

    uint32_t num_set = 
#ifdef MADRONA_GPU_MODE
        (uint32_t)__popc(bits);
#else
        std::popcount(bits);
#endif

    return (num_set & 1) == 0;
}

constexpr math::Vector2 sample2xUniform(RandKey k)
{
    return math::Vector2 {
        .x = bitsToFloat01(k.a),
        .y = bitsToFloat01(k.b),
    };
}

constexpr float bitsToFloat01(uint32_t rand_bits)
{
    // This implementation (and the one commented out below), generate random
    // numbers in the interval [0, 1). This is done by randomizing the mantissa
    // while leaving the exponent at 0. This means some small random numbers
    // near 0 won't be output (for example 2^-32 won't be output). The plus
    // side is that 1 - sampleUniform(k) will at most be equal to 1 - float
    // epsilon (not 1).
    //
    // random123 contains an implementation of this idea as well:
    // https://github.com/DEShawResearch/random123/blob/main/include/Random123/u01fixedpt.h
    // That library seems to advocate a version that will randomly generate
    // smaller floats as well (such as dividing by 2^-32) but only provides
    // implementations that generate in the range (0, 1]
    return (rand_bits >> 8_u32) * 0x1p-24f;

#if 0
    constexpr uint32_t exponent = 0x3f800000;
    uint32_t raw = (exponent | (rand_bits >> 9)) - 1;

#ifdef MADRONA_GPU_MODE
    return __uint_as_float(raw);
#else
    return std::bit_cast<float>(raw);
#endif
#endif
}

}

RNG::RNG()
    : k_(RandKey { 0, 0 }),
      count_(0)
{}

RNG::RNG(RandKey k)
    : k_(k),
      count_(0)
{}

RNG::RNG(uint32_t seed)
    : RNG(rand::initKey(seed))
{}

int32_t RNG::sampleI32(int32_t a, int32_t b)
{
    RandKey sample_k = advance();
    return rand::sampleI32(sample_k, a, b);
}

int32_t RNG::sampleI32Biased(int32_t a, int32_t b)
{
    RandKey sample_k = advance();
    return rand::sampleI32Biased(sample_k, a, b);
}

float RNG::sampleUniform()
{
    RandKey sample_k = advance();
    return rand::sampleUniform(sample_k);
}

bool RNG::sampleBool()
{
    RandKey sample_k = advance();
    return rand::sampleBool(sample_k);
}

RandKey RNG::randKey()
{
    return advance();
}

RandKey RNG::advance()
{
    RandKey sample_k = rand::split_i(k_, count_);
    count_ += 1;

    return sample_k;
}

}
