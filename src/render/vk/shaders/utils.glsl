#ifndef MADRONA_VK_UTILS_GLSL_INCLUDED
#define MADRONA_VK_UTILS_GLSL_INCLUDED

#include "math.glsl"

// Ray Tracing Gems Chapter 6 (avoid self intersections)
vec3 offsetRayOrigin(vec3 o, vec3 geo_normal)
{
#define GLOBAL_ORIGIN (1.0f / 32.0f)
#define FLOAT_SCALE (1.0f / 65536.0f)
#define INT_SCALE (256.0f)

    i32vec3 int_offset = i32vec3(geo_normal.x * INT_SCALE,
        geo_normal.y * INT_SCALE, geo_normal.z * INT_SCALE);

    vec3 o_integer = vec3(
        intBitsToFloat(
            floatBitsToInt(o.x) + ((o.x < 0) ? -int_offset.x : int_offset.x)),
        intBitsToFloat(
            floatBitsToInt(o.y) + ((o.y < 0) ? -int_offset.y : int_offset.y)),
        intBitsToFloat(
            floatBitsToInt(o.z) + ((o.z < 0) ? -int_offset.z : int_offset.z)));

    return vec3(
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

vec3 cosineHemisphere(vec2 uv)
{
    const float r = sqrt(uv.x);
    const float phi = 2.0f * M_PI * uv.y;
    vec2 disk = r * vec2(cos(phi), sin(phi));
    vec3 hemisphere = vec3(disk.x, disk.y,
        sqrt(max(0.0f, 1.0f - dot(disk, disk))));

    return hemisphere;
}

vec3 concentricHemisphere(vec2 uv)
{
    vec2 c = 2.f * uv - 1.f;
    vec2 d;
    if (c.x == 0.f && c.y == 0.f) {
        d = vec2(0.f);
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

        d = r * vec2(cos(phi), sin(phi));
    }

    float z = sqrt(max(0.f, 1.f - dot(d, d)));

    return vec3(d.x, d.y, z);
}

vec3 quatRotate(vec4 quat, vec3 dir)
{
    vec3 pure = quat.yzw;
    float scalar = quat.x;

    return 2.f * dot(pure, dir) * pure +
        (2.f * scalar * scalar - 1.f) * dir +
        2.f * scalar * cross(pure, dir);
}

vec3 quatInvRotate(vec4 quat, vec3 dir)
{
    return quatRotate(vec4(quat.x, -quat.yzw), dir);
}

vec2 dirToLatLong(vec3 dir)
{
    vec3 n = normalize(dir);
    
    return vec2(atan(n.x, -n.z) * (M_1_PI / 2.f) + 0.5f, acos(n.y) * M_1_PI);
}

// For the two equal area functions below to correctly map between eachother
// There needs to be a sign function that returns 1.0 for 0.0 and
// -1.0 for -0.0. This is due to the discontinuity in the octahedral map
// between the upper left triangle and bottom left triangle for example
// (same on right). Check image in RT Gems 16.5.4.2, light purple and dark
// purple for example.
float signPreserveZero(float v)
{
    int32_t i = floatBitsToInt(v);

    return (i < 0) ? -1.0 : 1.0;
}

// Ray Tracing Gems 16.5.4.2
// Better description in Clarberg's
// "Fast Equal-Area Mapping of the (Hemi)Sphere using SIMD"
vec3 octSphereMap(vec2 u)
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

    return vec3(x, y, z);
}

vec2 invOctSphereMap(vec3 dir)
{
    float r = sqrt(1.f - abs(dir.z));
    float phi = atan(abs(dir.y), abs(dir.x));

    vec2 uv;
    uv.y = r * phi * M_2_PI;
    uv.x = r - uv.y;

    if (dir.z < 0.f) {
        uv = 1.f - uv.yx;
    }

    uv.x *= signPreserveZero(dir.x);
    uv.y *= signPreserveZero(dir.y);

    return uv * 0.5f + 0.5f;
}

vec3 octahedralVectorDecode(vec2 f)
{
     f = f * 2.0 - 1.0;
     // https://twitter.com/Stubbesaurus/status/937994790553227264
     vec3 n = vec3(f.x, f.y, 1.f - abs(f.x) - abs(f.y));
     float t = clamp(-n.z, 0.0, 1.0);
     n.x += n.x >= 0.0 ? -t : t;
     n.y += n.y >= 0.0 ? -t : t;
     return normalize(n);
}

void decodeNormalTangent(in u32vec3 packed, out vec3 normal,
                         out vec4 tangentAndSign)
{
    vec2 ab = unpackHalf2x16(packed.x);
    vec2 cd = unpackHalf2x16(packed.y);

    normal = vec3(ab.x, ab.y, cd.x);
    float sign = cd.y;

    vec2 oct_tan = unpackSnorm2x16(packed.z);
    vec3 tangent = octahedralVectorDecode(oct_tan);

    tangentAndSign = vec4(tangent, sign);
}

vec3 transformPosition(mat4x3 o2w, vec3 p)
{
    return o2w[0] * p.x + o2w[1] * p.y + o2w[2] * p.z + o2w[3];
}

vec3 transformVector(mat4x3 o2w, vec3 v)
{
    return o2w[0] * v.x + o2w[1] * v.y + o2w[2] * v.z;
}

vec3 transformNormal(mat4x3 w2o, vec3 n)
{
    return vec3(dot(w2o[0], n), dot(w2o[1], n), dot(w2o[2], n));
}

vec3 sampleSphereUniform(vec2 uv)
{
    float z = 1.f - 2.f * uv.x;
    float r = sqrt(max(0.f, 1.f - z * z));
    float phi = 2.f * M_PI * uv.y;
    return vec3(r * cos(phi), r * sin(phi), z);
}

// http://lolengine.net/blog/2013/09/21/picking-orthogonal-vector-combing-coconuts
// Alternative: https://jcgt.org/published/0006/01/01/
vec3 getOrthogonalVec(vec3 v)
{
    return abs(v.x) > abs(v.z) ?
        vec3(-v.y, v.x, 0.0) :
        vec3(0.0, -v.z, v.y);
}

float rgbToLuminance(vec3 rgb)
{
    return 0.2126f * rgb.x + 0.7152f * rgb.y + 0.0722f * rgb.z;
}

#endif
