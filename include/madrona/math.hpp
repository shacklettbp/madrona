/*
 * Copyright 2021-2022 Brennan Shacklett and contributors
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 */
#pragma once

#include <cstdint>
#include <cmath>
#include <cfloat>

namespace madrona {
namespace math {

constexpr inline float pi {3.14159265358979323846264338327950288f};

namespace helpers {

inline constexpr float toRadians(float degrees)
{
    constexpr float mult = pi / 180.f;
    return mult * degrees;
}

inline constexpr float minf(float a, float b)
{
    return a < b ? a : b;
}

inline constexpr float maxf(float a, float b)
{
    return a > b ? a : b;
}

}

struct Vector2 {
    float x;
    float y;

    inline float dot(const Vector2 &o) const
    {
        return x * o.x + y * o.y;
    }

    inline float length2() const
    {
        return x * x + y * y;
    }

    inline float length() const
    {
        return sqrtf(length2());
    }

    inline float invLength() const
    {
#ifdef MADRONA_GPU_MODE
        return rsqrtf(length2());
#else
        return 1.f / length();
#endif
    }

    inline Vector2 & operator+=(const Vector2 &o)
    {
        x += o.x;
        y += o.y;

        return *this;
    }

    inline Vector2 & operator-=(const Vector2 &o)
    {
        x -= o.x;
        y -= o.y;

        return *this;
    }

    inline Vector2 & operator+=(float o)
    {
        x += o;
        y += o;

        return *this;
    }

    inline Vector2 & operator-=(float o)
    {
        x -= o;
        y -= o;

        return *this;
    }

    inline Vector2 & operator*=(float o)
    {
        x *= o;
        y *= o;

        return *this;
    }

    inline Vector2 & operator/=(float o)
    {
        float inv = 1.f / o;

        return *this *= inv;
    }

    friend inline Vector2 operator-(Vector2 v)
    {
        return Vector2 {
            -v.x,
            -v.y,
        };
    }

    friend inline Vector2 operator+(Vector2 a, const Vector2 &b)
    {
        a += b;

        return a;
    }

    friend inline Vector2 operator-(Vector2 a, const Vector2 &b)
    {
        a -= b;

        return a;
    }

    friend inline Vector2 operator+(Vector2 a, float b)
    {
        a += b;

        return a;
    }

    friend inline Vector2 operator-(Vector2 a, float b)
    {
        a -= b;

        return a;
    }

    friend inline Vector2 operator*(Vector2 a, float b)
    {
        a *= b;

        return a;
    }

    friend inline Vector2 operator/(Vector2 a, float b)
    {
        a /= b;
        return a;
    }

    friend inline Vector2 operator+(float a, Vector2 b)
    {
        return b + a;
    }

    friend inline Vector2 operator-(float a, Vector2 b)
    {
        return -b + a;
    }

    friend inline Vector2 operator*(float a, Vector2 b)
    {
        return b * a;
    }

    friend inline Vector2 operator/(float a, Vector2 b)
    {
        return Vector2 {
            a / b.x,
            a / b.y,
        };
    }
};

struct Vector3 {
    float x;
    float y;
    float z;

    inline float dot(const Vector3 &o) const
    {
        return x * o.x + y * o.y + z * o.z;
    }

    inline Vector3 cross(const Vector3 &o) const
    {
        return Vector3 {
            y * o.z - z * o.y,
            z * o.x - x * o.z,
            x * o.y - y * o.x,
        };
    }

    inline float length2() const
    {
        return x * x + y * y + z * z;
    }

    inline float length() const
    {
        return sqrtf(length2());
    }

    inline float invLength() const
    {
#ifdef MADRONA_GPU_MODE
        return rsqrtf(length2());
#else
        // FIXME: is there a CPU fast path here?
        return 1.f / length();
#endif
    }

    inline float distance(const Vector3 &o) const
    {
        return (*this - o).length();
    }

    inline float distance2(const Vector3 &o) const
    {
        return (*this - o).length2();
    }

    [[nodiscard]] inline Vector3 normalize() const
    {
        return *this * invLength();
    } 

    inline float & operator[](uint32_t i)
    {
        switch (i) {
            default:
            case 0:
                return x;
            case 1:
                return y;
            case 2:
                return z;
        }
    }

    inline Vector3 & operator+=(const Vector3 &o)
    {
        x += o.x;
        y += o.y;
        z += o.z;

        return *this;
    }

    inline Vector3 & operator-=(const Vector3 &o)
    {
        x -= o.x;
        y -= o.y;
        z -= o.z;

        return *this;
    }

    inline Vector3 & operator+=(float o)
    {
        x += o;
        y += o;
        z += o;

        return *this;
    }

    inline Vector3 & operator-=(float o)
    {
        x -= o;
        y -= o;
        z -= o;

        return *this;
    }

    inline Vector3 & operator*=(float o)
    {
        x *= o;
        y *= o;
        z *= o;

        return *this;
    }

    inline Vector3 & operator/=(float o)
    {
        float inv = 1.f / o;

        return *this *= inv;
    }

    friend inline Vector3 operator-(Vector3 v)
    {
        return Vector3 {
            -v.x,
            -v.y,
            -v.z,
        };
    }

    friend inline Vector3 operator+(Vector3 a, const Vector3 &b)
    {
        a += b;

        return a;
    }

    friend inline Vector3 operator-(Vector3 a, const Vector3 &b)
    {
        a -= b;

        return a;
    }

    friend inline Vector3 operator+(Vector3 a, float b)
    {
        a += b;

        return a;
    }

    friend inline Vector3 operator-(Vector3 a, float b)
    {
        a -= b;

        return a;
    }

    friend inline Vector3 operator*(Vector3 a, float b)
    {
        a *= b;

        return a;
    }

    friend inline Vector3 operator/(Vector3 a, float b)
    {
        a /= b;
        return a;
    }

    friend inline Vector3 operator+(float a, Vector3 b)
    {
        return b + a;
    }

    friend inline Vector3 operator-(float a, Vector3 b)
    {
        return -b + a;
    }

    friend inline Vector3 operator*(float a, Vector3 b)
    {
        return b * a;
    }

    friend inline Vector3 operator/(float a, Vector3 b)
    {
        return Vector3 {
            a / b.x,
            a / b.y,
            a / b.z,
        };
    }

    static inline Vector3 min(Vector3 a, Vector3 b)
    {
        return Vector3 {
            helpers::minf(a.x, b.x),
            helpers::minf(a.x, b.x),
            helpers::minf(a.x, b.x),
        };
    }

    static inline Vector3 max(Vector3 a, Vector3 b)
    {
        return Vector3 {
            helpers::maxf(a.x, b.x),
            helpers::maxf(a.x, b.x),
            helpers::maxf(a.x, b.x),
        };
    }
};

struct Vector4 {
    float x;
    float y;
    float z;
    float w;
};

inline float dot(Vector3 a, Vector3 b)
{
    return a.dot(b);
}

inline Vector3 cross(Vector3 a, Vector3 b)
{
    return a.cross(b);
}

struct Quat {
    float w;
    float x;
    float y;
    float z;

    inline float length2() const
    {
        return w * w + x * x + y * y + z * z;
    }

    inline float length() const
    {
        return sqrtf(length2());
    }

    inline float invLength() const
    {
#ifdef MADRONA_GPU_MODE
        return rsqrtf(length2());
#else
        return 1.f / sqrtf(length2());
#endif
    }

    [[nodiscard]] inline Quat normalize() const
    {
        float inv_length = invLength();

        return Quat {
            w * inv_length,
            x * inv_length,
            y * inv_length,
            z * inv_length,
        };
    }

    [[nodiscard]] inline Quat inv() const
    {
        return Quat {
            w,
            -x,
            -y,
            -z,
        };
    }

    inline Quat & operator+=(Quat o)
    {
        w += o.w;
        x += o.x;
        y += o.y;
        z += o.z;

        return *this;
    }

    inline Quat & operator-=(Quat o)
    {
        w -= o.w;
        x -= o.x;
        y -= o.y;
        z -= o.z;

        return *this;
    }

    inline Quat & operator*=(Quat o)
    {
        w = (w * o.w - x * o.x - y * o.y - z * o.z);
        x = (w * o.x + x * o.w + y * o.z - z * o.y);
        y = (w * o.y - x * o.z + y * o.w + z * o.x);
        z = (w * o.z + x * o.y - y * o.x + z * o.w);

        return *this;
    }

    friend inline Quat operator+(Quat a, Quat b)
    {
        return a += b;
    }

    friend inline Quat operator-(Quat a, Quat b)
    {
        return a -= b;
    }

    friend inline Quat operator*(Quat a, Quat b)
    {
        return a *= b;
    }

    static inline Quat angleAxis(float angle, Vector3 normal)
    {
        float coshalf = cosf(angle / 2.f);
        float sinhalf = sinf(angle / 2.f);

        return Quat {
            coshalf,
            normal.x * sinhalf,
            normal.y * sinhalf,
            normal.z * sinhalf,
        };
    }

    inline Vector3 rotateDir(Vector3 v) const
    {
        Vector3 pure {x, y, z};
        float scalar = w;

        return 2.f * dot(pure, v) * pure +
            (2.f * scalar * scalar - 1.f) * v +
            2.f * scalar * cross(pure, v);
    }
};

struct AABB {
    Vector3 pMin;
    Vector3 pMax;

    inline float surfaceArea() const
    {
        Vector3 d = pMax - pMin;
        return 2.f * (d.x * d.y + d.x * d.z + d.y * d.z);
    }

    inline bool overlaps(const AABB &o) const
    {
        auto [a_min, a_max] = *this;
        auto [b_min, b_max] = o;

        return a_min.x < b_max.x && b_min.x < a_max.x &&
               a_min.y < b_max.y && b_min.y < a_max.y &&
               a_min.z < b_max.z && b_min.z < a_max.z;
    }
    
    inline bool contains(const AABB &o) const
    {
        auto [a_min, a_max] = *this;
        auto [b_min, b_max] = o;

        return a_min.x <= b_min.x &&
               a_min.y <= b_min.y &&
               a_min.z <= b_min.z &&
               a_max.x >= b_max.x &&
               a_max.y >= b_max.y &&
               a_max.z >= b_max.z; 
    }

    inline void expand(const Vector3 &p)
    {
        if (p.x < pMin.x) {
            pMin.x = p.x;
        } else if (p.x > pMax.x) {
            pMax.x = p.x;
        }

        if (p.y < pMin.y) {
            pMin.y = p.y;
        } else if (p.y > pMax.y) {
            pMax.y = p.y;
        }

        if (p.z < pMin.z) {
            pMin.z = p.z;
        } else if (p.z > pMax.z) {
            pMax.z = p.z;
        }
    }

    static inline AABB invalid()
    {
        return AABB {
            /* .pMin = */ Vector3 {FLT_MAX, FLT_MAX, FLT_MAX},
            /* .pMax = */ Vector3 {FLT_MIN, FLT_MIN, FLT_MIN},
        };
    }

    static inline AABB point(const Vector3 &p)
    {
        return AABB {
            /* .pMin = */ p,
            /* .pMax = */ p,
        };
    }

    static inline AABB merge(const AABB &a, const AABB &b)
    {
        return AABB {
            /* .pMin = */ Vector3::min(a.pMin, b.pMin),
            /* .pMax = */ Vector3::min(a.pMax, b.pMax),
        };
    }
};

struct Mat3x4 {
    Vector3 cols[4];

    static inline Mat3x4 fromRows(Vector4 row0, Vector4 row1, Vector4 row2)
    {
        return {
            Vector3 { row0.x, row1.x, row2.x },
            Vector3 { row0.y, row1.y, row2.y },
            Vector3 { row0.z, row1.z, row2.z },
            Vector3 { row0.w, row1.w, row2.w },
        };
    }

    static inline Mat3x4 fromTRS(Vector3 t, Quat r, float s = 1.f)
    {
        float x2 = r.x * r.x;
        float y2 = r.y * r.y;
        float z2 = r.z * r.z;
        float xz = r.x * r.z;
        float xy = r.x * r.y;
        float yz = r.y * r.z;
        float wx = r.w * r.x;
        float wy = r.w * r.y;
        float wz = r.w * r.z;

        return {{
            { s * (1.f - 2.f * (y2 + z2)), 2.f * (xy + wz), 2.f * (xz - wy) },
            { 2.f * (xy - wz), s * (1.f - 2.f * (x2 + z2)), 2.f * (yz + wx) },
            { 2.f * (xz + wy), 2.f * (yz - wx), s * (1.f - 2.f * (x2 + y2)) },
            t,
        }};
    }

    inline Vector3 txfmPoint(Vector3 p) const
    {
        return cols[0] * p.x + cols[1] * p.y + cols[2] * p.z + cols[3];
    }

    inline Vector3 txfmDir(Vector3 p) const
    {
        return cols[0] * p.x + cols[1] * p.y + cols[2] * p.z;
    }

    inline Mat3x4 compose(const Mat3x4 &o) const
    {
        return Mat3x4 {
            txfmDir(o.cols[0]),
            txfmDir(o.cols[1]),
            txfmDir(o.cols[2]),
            txfmPoint(o.cols[3]),
        };
    }
};

}
}
