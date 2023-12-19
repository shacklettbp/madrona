#pragma once

#include <madrona/math.hpp>
#include <madrona/macros.hpp>

#include <bit>

namespace madrona::render {

// Stack overflow
// https://stackoverflow.com/questions/1659440/32-bit-to-16-bit-floating-point-conversion
// IEEE-754 16-bit floating-point format (without infinity):
// 1-5-10, exp-15, +-131008.0, +-6.1035156E-5, +-5.9604645E-8, 3.311 digits
inline float f16tof32(const uint16_t x) { 
    const uint32_t e = (x & 0x7C00) >> 10; // exponent
    const uint32_t m = (x & 0x03FF) << 13; // mantissa

    // evil log2 bit hack to count leading zeros in denormalized format
    const uint32_t v = std::bit_cast<uint32_t>((float)m) >> 23;

    // sign : normalized : denormalized
    return std::bit_cast<float>(
        (x & 0x8000) << 16 | (e != 0) * ((e + 112) << 23 | m) |
        ((e == 0) & (m != 0)) * ((v - 37) << 23 |
            ((m << (150 - v)) & 0x007FE000)));
}

// IEEE-754 16-bit floating-point format (without infinity):
// 1-5-10, exp-15, +-131008.0, +-6.1035156E-5, +-5.9604645E-8, 3.311 digits
inline uint16_t f32tof16(const float x) {
    // round-to-nearest-even: add last bit after truncated mantissa
    const uint32_t b = std::bit_cast<uint32_t>(x) + 0x00001000;
    const uint32_t e = (b & 0x7F800000) >> 23; // exponent
    // mantissa; in line below: 0x007FF000 = 0x00800000-0x00001000 =
    // decimal indicator flag - initial rounding
    const uint32_t m = b & 0x007FFFFF;
    return (b & 0x80000000) >> 16 | (e > 112) * 
            ((((e - 112) << 10) & 0x7C00) | m >> 13) |
        ((e < 113) & (e > 101)) *
            ((((0x007FF000 + m) >> (125 - e)) + 1) >> 1) |
        (e > 143) * 0x7FFF; // sign : normalized : denormalized : saturate
}

inline math::Vector3 encodeNormalTangent(const math::Vector3 &normal,
                                         const math::Vector4 &tangent_plussign)
{
    using namespace math;

    auto packHalf2x16 = [](const Vector2 &v) {
#if defined(MADRONA_MSVC)
        uint16_t x_half = f32tof16(v.x);
        uint16_t y_half = f32tof16(v.y);
#else
#if defined(MADRONA_GCC)
        _Float16 x_half, y_half;
#elif defined(MADRONA_CLANG)
        __fp16 x_half, y_half;
#else
        STATIC_UNIMPLEMEMENTED();
#endif
        x_half = v.x;
        y_half = v.y;
#endif

        return uint32_t(std::bit_cast<uint16_t>(y_half)) << 16 |
            uint32_t(std::bit_cast<uint16_t>(x_half));
    };

    auto packSnorm2x16 = [](const Vector2 &v) {
        uint16_t x = roundf(fminf(fmaxf(v.x, -1.f), 1.f) * 32767.f);
        uint16_t y = roundf(fminf(fmaxf(v.y, -1.f), 1.f) * 32767.f);

        return uint32_t(x) << 16 | uint32_t(y);
    };

    // https://knarkowicz.wordpress.com/2014/04/16/octahedron-normal-vector-encoding/
    auto octWrap = [](const Vector2 &v) {
        return Vector2 {
            (1.f - fabsf(v.y)) * (v.x >= 0.f ? 1.f : -1.f),
            (1.f - fabsf(v.x)) * (v.y >= 0.f? 1.f : -1.f),
        };
    };
 
    auto octEncode = [&octWrap](Vector3 n) {
        n /= (fabsf(n.x) + fabsf(n.y) + fabsf(n.z));

        Vector2 nxy {n.x, n.y};

        nxy = n.z >= 0.0f ? nxy : octWrap(nxy);
        nxy = nxy * 0.5f + 0.5f;
        return nxy;
    };

    Vector3 tangent = {
        tangent_plussign.x,
        tangent_plussign.y,
        tangent_plussign.z,
    };
    float bitangent_sign = tangent_plussign.w;

    uint32_t nxy = packHalf2x16(Vector2 {normal.x, normal.y});
    uint32_t nzsign = packHalf2x16(Vector2 {normal.z, bitangent_sign});

    Vector2 octtan = octEncode(tangent);
    uint32_t octtan_snorm = packSnorm2x16(octtan);

    return Vector3 {
        std::bit_cast<float>(nxy),
        std::bit_cast<float>(nzsign),
        std::bit_cast<float>(octtan_snorm),
    };
}

}
