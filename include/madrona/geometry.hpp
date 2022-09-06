#pragma once

#include <cstdint>
#include <cmath>

namespace madrona {

class Vector3 {
public:
    float x;
    float y;
    float z;

    inline float dot(const Vector3 &o) const
    {
        return x * o.x + y * o.y + z * o.z;
    }

    inline float length2() const
    {
        return x * x + y * y + z * z;
    }

    inline float length() const
    {
        return sqrtf(length2());
    }

    inline float distance(const Vector3 &o) const
    {
        return (*this - o).length();
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

    friend inline Vector3 operator+(Vector3 a, const Vector3 &b)
    {
        a.x += b.x;
        a.y += b.y;
        a.z += b.z;

        return a;
    }

    friend inline Vector3 operator-(Vector3 a, const Vector3 &b)
    {
        a.x -= b.x;
        a.y -= b.y;
        a.z -= b.z;

        return a;
    }

    friend inline Vector3 operator+(Vector3 a, float b)
    {
        a.x += b;
        a.y += b;
        a.z += b;

        return a;
    }

    friend inline Vector3 operator-(Vector3 a, float b)
    {
        a.x -= b;
        a.y -= b;
        a.z -= b;

        return a;
    }

    friend inline Vector3 operator*(Vector3 a, float b)
    {
        a.x *= b;
        a.y *= b;
        a.z *= b;

        return a;
    }

    friend inline Vector3 operator/(Vector3 a, float b)
    {
        float inv = 1.f / b;

        return a * inv;
    }
};

struct AABB {
    Vector3 pMin;
    Vector3 pMax;
};

}
