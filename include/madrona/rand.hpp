#pragma once

#include <madrona/macros.hpp>
#include <madrona/types.hpp>
#include <madrona/math.hpp>

namespace madrona {

struct RandKey {
    uint32_t a;
    uint32_t b;
};

namespace rand {

constexpr inline RandKey initKey(uint32_t seed, uint32_t seed_upper = 0);
constexpr inline RandKey split_i(
  RandKey src, uint32_t idx, uint32_t idx_upper = 0);

constexpr inline uint32_t bits32(RandKey k);
constexpr inline uint64_t bits64(RandKey k);

constexpr inline int32_t sampleI32(RandKey k, int32_t a, int32_t b);
constexpr inline int32_t sampleI32Biased(RandKey k, int32_t a, int32_t b);
constexpr inline float sampleUniform(RandKey k);
constexpr inline bool sampleBool(RandKey k);
constexpr inline math::Vector2 sample2xUniform(RandKey k);
constexpr inline float bitsToFloat01(uint32_t rand_bits);


}

class RNG {
public:
    inline RNG();
    inline RNG(RandKey k);
    inline RNG(uint32_t seed);

    inline int32_t sampleI32(int32_t a, int32_t b);
    inline int32_t sampleI32Biased(int32_t a, int32_t b);
    inline float sampleUniform();
    inline bool sampleBool();

    inline RandKey randKey();

    RNG(const RNG &) = default;
    RNG(RNG &&) = default;
    RNG & operator=(const RNG &) = default;
    RNG & operator=(RNG &&) = default;

private:
    inline RandKey advance();

    RandKey k_;
    uint32_t count_;
};

}

#include "rand.inl"
