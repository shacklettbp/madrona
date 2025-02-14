#ifndef MADRONA_VK_UTILS_HLSL_INCLUDED
#define MADRONA_VK_UTILS_HLSL_INCLUDED

#include "math.hlsl"

// Ray Tracing Gems Chapter 6 (avoid self intersections)
float3 offsetRayOrigin(float3 o, float3 geo_normal)
{
#define GLOBAL_ORIGIN (1.0f / 32.0f)
#define FLOAT_SCALE (1.0f / 65536.0f)
#define INT_SCALE (256.0f)

    int3 int_offset = int3(geo_normal.x * INT_SCALE,
        geo_normal.y * INT_SCALE, geo_normal.z * INT_SCALE);

    float3 o_integer = float3(
        asfloat(asint(o.x) + ((o.x < 0) ? -int_offset.x : int_offset.x)),
        asfloat(asint(o.y) + ((o.y < 0) ? -int_offset.y : int_offset.y)),
        asfloat(asint(o.z) + ((o.z < 0) ? -int_offset.z : int_offset.z)));

    return float3(
        abs(o.x) < GLOBAL_ORIGIN ?
            o.x + FLOAT_SCALE * geo_normal.x : o_integer.x,
        abs(o.y) < GLOBAL_ORIGIN ?
            o.y + FLOAT_SCALE * geo_normal.y : o_integer.y,
        abs(o.z) < GLOBAL_ORIGIN ?
            o.z + FLOAT_SCALE * geo_normal.z : o_integer.z);

#undef GLOBAL_ORIGIN
#undef FLOAT_SCALE
#undef INT_SCALE
}

float3 cosineHemisphere(float2 uv)
{
    const float r = sqrt(uv.x);
    const float phi = 2.0f * M_PI * uv.y;
    float2 disk = r * float2(cos(phi), sin(phi));
    float3 hemisphere = float3(disk.x, disk.y,
        sqrt(max(0.0f, 1.0f - dot(disk, disk))));

    return hemisphere;
}

float3 concentricHemisphere(float2 uv)
{
    float2 c = 2.f * uv - 1.f;
    float2 d;
    if (c.x == 0.f && c.y == 0.f) {
        d = float2(0.f, 0.f);
    } else {
        float phi, r;
        if (abs(c.x) > abs(c.y))
        {
            r = c.x;
            phi = (c.y / c.x) * (M_PI / 4.f);
        } else {
            r = c.y;
            phi = (M_PI / 2.f) - (c.x / c.y) * (M_PI / 4.f);
        }

        d = r * float2(cos(phi), sin(phi));
    }

    float z = sqrt(max(0.f, 1.f - dot(d, d)));

    return float3(d.x, d.y, z);
}

float3 quatRotate(float4 quat, float3 dir)
{
    float3 pure = quat.yzw;
    float scalar = quat.x;

    return 2.f * dot(pure, dir) * pure +
        (2.f * scalar * scalar - 1.f) * dir +
        2.f * scalar * cross(pure, dir);
}

float3 quatInvRotate(float4 quat, float3 dir)
{
    return quatRotate(float4(quat.x, -quat.yzw), dir);
}

float2 dirToLatLong(float3 dir)
{
    float3 n = normalize(dir);
    
    return float2(atan2(n.x, -n.z) * (M_1_PI / 2.f) + 0.5f, acos(n.y) * M_1_PI);
}

// For the two equal area functions below to correctly map between eachother
// There needs to be a sign function that returns 1.0 for 0.0 and
// -1.0 for -0.0. This is due to the discontinuity in the octahedral map
// between the upper left triangle and bottom left triangle for example
// (same on right). Check image in RT Gems 16.5.4.2, light purple and dark
// purple for example.
float signPreserveZero(float v)
{
    int32_t i = asint(v);

    return (i < 0) ? -1.0 : 1.0;
}

// Ray Tracing Gems 16.5.4.2
// Better description in Clarberg's
// "Fast Equal-Area Mapping of the (Hemi)Sphere using SIMD"
float3 octSphereMap(float2 u)
{
    u = u * 2.f - 1.f;

    // Compute radius r (branchless)
    float d = 1.f - (abs(u.x) + abs(u.y));
    float r = 1.f - abs(d);

    // Compute phi in the first quadrant (branchless, except for the
    // division-by-zero test), using sign(u) to map the result to the
    // correct quadrant below
    float phi = (r == 0.f) ? 0.f :
        (M_PI_4 * ((abs(u.y) - abs(u.x)) / r + 1.f));

    float f = r * sqrt(2.f - r * r);

    // abs() around f * cos/sin(phi) is necessary because they can return
    // negative 0 due to floating precision
    float x = signPreserveZero(u.x) * abs(f * cos(phi));
    float y = signPreserveZero(u.y) * abs(f * sin(phi));
    float z = signPreserveZero(d) * (1.f - r * r);

    return float3(x, y, z);
}

float2 invOctSphereMap(float3 dir)
{
    float r = sqrt(1.f - abs(dir.z));
    float phi = atan2(abs(dir.y), abs(dir.x));

    float2 uv;
    uv.y = r * phi * M_2_PI;
    uv.x = r - uv.y;

    if (dir.z < 0.f) {
        uv = 1.f - uv.yx;
    }

    uv.x *= signPreserveZero(dir.x);
    uv.y *= signPreserveZero(dir.y);

    return uv * 0.5f + 0.5f;
}

float3 octahedralVectorDecode(float2 f)
{
     f = f * 2.0 - 1.0;
     // https://twitter.com/Stubbesaurus/status/937994790553227264
     float3 n = float3(f.x, f.y, 1.f - abs(f.x) - abs(f.y));
     float t = clamp(-n.z, 0.0, 1.0);
     n.x += n.x >= 0.0 ? -t : t;
     n.y += n.y >= 0.0 ? -t : t;
     return normalize(n);
}

float2 unpackHalf2x16(uint32_t x)
{
    return float2(f16tof32(x), f16tof32(x >> 16));
}

float2 unpackSnorm2x16(uint32_t x)
{
    int2 halves = int2(x & 0xFFFF, x >> 16);
    precise float2 unpacked = max((float2)halves / 32767.f, -1.f);

    return unpacked;
}

void decodeNormalTangent(in uint3 packed, out float3 normal,
                         out float4 tangentAndSign)
{
    float2 ab = unpackHalf2x16(packed.x);
    float2 cd = unpackHalf2x16(packed.y);

    normal = float3(ab.x, ab.y, cd.x);
    float sign = cd.y;

    float2 oct_tan = unpackSnorm2x16(packed.z);
    float3 tangent = octahedralVectorDecode(oct_tan);

    tangentAndSign = float4(tangent, sign);
}

float3 transformPosition(float3x4 o2w, float3 p)
{
    return float3(dot(o2w[0].xyz, p) + o2w[0].w,
                  dot(o2w[1].xyz, p) + o2w[1].w,
                  dot(o2w[2].xyz, p) + o2w[2].w);
}

float3 transformVector(float3x4 o2w, float3 v)
{
    return float3(dot(o2w[0].xyz, v),
                  dot(o2w[1].xyz, v),
                  dot(o2w[2].xyz, v));
}

float3 transformNormal(float3x4  w2o, float3 n)
{
    return float3(dot(w2o._m00_m10_m20, n),
                  dot(w2o._m01_m11_m21, n),
                  dot(w2o._m02_m12_m22, n));
}

float3 sampleSphereUniform(float2 uv)
{
    float z = 1.f - 2.f * uv.x;
    float r = sqrt(max(0.f, 1.f - z * z));
    float phi = 2.f * M_PI * uv.y;
    return float3(r * cos(phi), r * sin(phi), z);
}

// http://lolengine.net/blog/2013/09/21/picking-orthogonal-vector-combing-coconuts
// Alternative: https://jcgt.org/published/0006/01/01/
float3 getOrthogonalVec(float3 v)
{
    return abs(v.x) > abs(v.z) ?
        float3(-v.y, v.x, 0.0) :
        float3(0.0, -v.z, v.y);
}

float rgbToLuminance(float3 rgb)
{
    return 0.2126f * rgb.x + 0.7152f * rgb.y + 0.0722f * rgb.z;
}

float4 hexToRgb(uint32_t hex)
{
    // Extract each color component and normalize to [0, 1] range
    float r = ((hex >> 16) & 0xFF) / 255.0f;
    float g = ((hex >> 8) & 0xFF) / 255.0f;
    float b = (hex & 0xFF) / 255.0f;

    return float4(r, g, b, 1.0);
}

#endif
