#pragma once

#include <madrona/math.hpp>
#include <madrona/macros.hpp>

#include <bit>

namespace madrona::render {

inline math::Vector3 encodeNormalTangent(const math::Vector3 &normal,
                                         const math::Vector4 &tangent_plussign)
{
    using namespace math;

    auto packHalf2x16 = [](const Vector2 &v) {
#if defined(MADRONA_GCC)
        _Float16 x_half, y_half;
#elif defined(MADRONA_CLANG)
        __fp16 x_half, y_half;
#else
        STATIC_UNIMPLEMEMENTED();
#endif

        x_half = v.x;
        y_half = v.y;

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
