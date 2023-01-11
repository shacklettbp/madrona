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

#include <madrona/types.hpp>

namespace madrona {
namespace math {

constexpr inline float pi {3.14159265358979323846264338327950288f};
constexpr inline float pi_d2 {pi / 2.f};
constexpr inline float pi_m2 {pi * 2.f};

namespace helpers {

inline constexpr float toRadians(float degrees)
{
    constexpr float mult = pi / 180.f;
    return mult * degrees;
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

    // Returns two vectors perpendicular to this vector
    // *this should be a normalized
    inline void frame(Vector3 *a, Vector3 *b) const
    {
        Vector3 arbitrary;
        if (fabsf(x) < 0.8) {
            arbitrary = Vector3 { 1, 0, 0 };
        } else {
            arbitrary = Vector3 { 0, 1, 0 };
        }

        *a = cross(arbitrary);
        *b = cross(*a);
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

    inline Vector3 projectOnto(const Vector3 &other) const
    {
        float thisDotOther = this->dot(other);
        float otherDotOther = other.dot(other);
        return other * (thisDotOther) / (otherDotOther);
    }

    inline float & operator[](CountT i)
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

    inline float operator[](CountT i) const
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

    friend inline Vector3 operator*(Vector3 a, Vector3 b)
    {
        return Vector3 {
            a.x * b.x,
            a.y * b.y,
            a.z * b.z
        };
    }



    static inline Vector3 min(Vector3 a, Vector3 b)
    {
        return Vector3 {
            fminf(a.x, b.x),
            fminf(a.y, b.y),
            fminf(a.z, b.z),
        };
    }

    static inline Vector3 max(Vector3 a, Vector3 b)
    {
        return Vector3 {
            fmaxf(a.x, b.x),
            fmaxf(a.y, b.y),
            fmaxf(a.z, b.z),
        };
    }

    static constexpr inline Vector3 zero()
    {
        return Vector3 {
            0,
            0,
            0,
        };
    }
};

struct Vector4 {
    float x;
    float y;
    float z;
    float w;

    inline Vector3 xyz() const
    {
        return Vector3 {
            x,
            y,
            z,
        };
    }

    static inline Vector4 fromVector3(Vector3 v, float w)
    {
        return Vector4 {
            v.x,
            v.y,
            v.z,
            w,
        };
    }
};

inline Vector4 makeVector4(const Vector3 &xyz, float w)
{
    return { xyz.x, xyz.y, xyz.z, w };
}

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

    inline Vector3 rotateVec(Vector3 v) const
    {
        Vector3 pure {x, y, z};
        float scalar = w;

        Vector3 pure_x_v = cross(pure, v);
        Vector3 pure_x_pure_x_v = cross(pure, pure_x_v);
        
        return v + 2.f * ((pure_x_v * scalar) + pure_x_pure_x_v);
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

    static inline Quat fromAngularVec(Vector3 v)
    {
        return Quat {
            0,
            v.x,
            v.y,
            v.z,
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
        // Slightly cleaner to implement in terms of operator* because then we
        // don't need to worry about overwriting members that will be used
        // later in the multiplication computation
        return *this = (*this * o);
    }

    inline Quat & operator*=(float f)
    {
        w *= f;
        x *= f;
        y *= f;
        z *= f;

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
        return Quat {
            (a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z),
            (a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y),
            (a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x),
            (a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w),
        };
    }

    friend inline Quat operator*(Quat a, float b)
    {
        a *= b;

        return a;
    }

    friend inline Quat operator*(float b, Quat a)
    {
        a *= b;

        return a;
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

    inline bool rayIntersects(Vector3 ray_o, Vector3 inv_ray_d,
                              float ray_t_min, float ray_t_max)
    {
        // Ray tracing gems II, chapter 2
        
        // Absolute distances to lower and upper box coordinates
        math::Vector3 t_lower = (pMin - ray_o) * inv_ray_d;
        math::Vector3 t_upper = (pMax - ray_o) * inv_ray_d;
        // The four t-intervals (for x-/y-/z-slabs, and ray p(t))
        math::Vector4 t_mins =
            Vector4::fromVector3(Vector3::min(t_lower, t_upper), ray_t_min);
        math::Vector4 t_maxes = 
            Vector4::fromVector3(Vector3::max(t_lower, t_upper), ray_t_max);
        // Easy to remember: ``max of mins, and min of maxes''

        auto max_component = [](Vector4 v) {
            return fmaxf(v.x, fmaxf(v.y, fmaxf(v.z, v.w)));
        };

        auto min_component = [](Vector4 v) {
            return fminf(v.x, fminf(v.y, fminf(v.z, v.w)));
        };
       
        float t_box_min = max_component(t_mins);
        float t_box_max = min_component(t_maxes);
        return t_box_min <= t_box_max;
    }

    static inline AABB invalid()
    {
        return AABB {
            /* .pMin = */ Vector3 {FLT_MAX, FLT_MAX, FLT_MAX},
            /* .pMax = */ Vector3 {-FLT_MAX, -FLT_MAX, -FLT_MAX},
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
            /* .pMax = */ Vector3::max(a.pMax, b.pMax),
        };
    }
};

struct Mat3x3 {
    Vector3 cols[3];

    static inline Mat3x3 fromQuat(Quat r)
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
            { 
                1.f - 2.f * (y2 + z2),
                2.f * (xy + wz),
                2.f * (xz - wy),
            },
            {
                2.f * (xy - wz),
                1.f - 2.f * (x2 + z2),
                2.f * (yz + wx),
            },
            {
                2.f * (xz + wy),
                2.f * (yz - wx),
                1.f - 2.f * (x2 + y2),
            },
        }};
    }

    static inline Mat3x3 fromRS(Quat r, Vector3 s)
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

        Vector3 ds = 2.f * s;

        return {{
            { 
                s.x - ds.x * (y2 + z2),
                ds.x * (xy + wz),
                ds.x * (xz - wy),
            },
            {
                ds.y * (xy - wz),
                s.y - ds.y * (x2 + z2),
                ds.y * (yz + wx),
            },
            {
                ds.z * (xz + wy),
                ds.z * (yz - wx),
                s.z - ds.z * (x2 + y2),
            },
        }};
    }

    Vector3 & operator[](CountT i)
    {
        return cols[i];
    }

    Vector3 operator[](CountT i) const
    {
        return cols[i];
    }

    inline Vector3 operator*(Vector3 v)
    {
        return cols[0] * v.x + cols[1] * v.y + cols[2] * v.z;
    }

    inline Mat3x3 operator*(const Mat3x3 &o)
    {
        return Mat3x3 {
            *this * o.cols[0],
            *this * o.cols[1],
            *this * o.cols[2],
        };
    }

    inline Mat3x3 & operator*=(const Mat3x3 &o)
    {
        return *this = (*this * o);
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

    static inline Mat3x4 fromTRS(Vector3 t, Quat r,
                                 Vector3 s = { 1.f, 1.f, 1.f })
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

        Vector3 ds = 2.f * s;

        return {{
            { 
                s.x - ds.x * (y2 + z2),
                ds.x * (xy + wz),
                ds.x * (xz - wy),
            },
            {
                ds.y * (xy - wz),
                s.y - ds.y * (x2 + z2),
                ds.y * (yz + wx),
            },
            {
                ds.z * (xz + wy),
                ds.z * (yz - wx),
                s.z - ds.z * (x2 + y2),
            },
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

constexpr inline Vector3 up { 0, 0, 1 };
constexpr inline Vector3 fwd { 0, 1, 0 };
constexpr inline Vector3 right { 1, 0, 0 };

}
}
