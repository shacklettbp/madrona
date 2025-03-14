#ifndef MADRONA_GPU_MODE
#include <bit>
#endif

namespace madrona::math {

inline constexpr float toRadians(float degrees)
{
    constexpr float mult = pi / 180.f;
    return mult * degrees;
}

// https://en.wikipedia.org/wiki/Fast_inverse_square_root
inline constexpr float rsqrtApprox(float x)
{
#ifdef MADRONA_GPU_MODE
    const float y = __uint_as_float(0x5F1FFFF9 - (__float_as_uint(x) >> 1));
	return y * (0.703952253f * (2.38924456f - x * y * y));
#else
    using std::bit_cast;
    const float y = bit_cast<float>(0x5F1FFFF9 - (bit_cast<uint32_t>(x) >> 1));
	return y * (0.703952253f * (2.38924456f - x * y * y));
#endif
}

inline constexpr float clamp(float v, float a, float b)
{
    return fmin(b, fmax(a, v));
}

bool solveQuadraticUnsafe(
    float a, float b, float c, float *t1, float *t2)
{
    float det = b * b - 4.f * a * c;
    if (det < 0.f) {
        return false;
    }
    
    float sqrt_det = sqrtf(det);
    float rcp_2a = 1.f / (2.f * a);
    
    *t1 = (-b - sqrt_det) * rcp_2a;
    *t2 = (-b + sqrt_det) * rcp_2a;

    return true;
}

float Vector2::dot(const Vector2 &o) const
{
    return x * o.x + y * o.y;
}

float Vector2::length2() const
{
    return x * x + y * y;
}

float Vector2::length() const
{
    return sqrtf(length2());
}

float Vector2::invLength() const
{
#ifdef MADRONA_GPU_MODE
    return rsqrtf(length2());
#else
    return 1.f / length();
#endif
}

float & Vector2::operator[](CountT i)
{
#ifdef MADRONA_GPU_MODE
    // For some reason, the CUDA compiler will sometimes insert a BSYNC 
    // instruction when using the switch statement. The ternary leads
    // to the least amount of instructions.
    return (i == 0) ? x : y;
#else
    switch (i) {
        case 0:
            return x;
        case 1:
            return y;
        default:
            MADRONA_UNREACHABLE();
    }
#endif
}

float Vector2::operator[](CountT i) const
{
#ifdef MADRONA_GPU_MODE
    // For some reason, the CUDA compiler will sometimes insert a BSYNC 
    // instruction when using the switch statement. The ternary leads
    // to the least amount of instructions.
    return (i == 0) ? x : y;
#else
    switch (i) {
        case 0:
            return x;
        case 1:
            return y;
        default:
            MADRONA_UNREACHABLE();
    }
#endif
}

constexpr Vector2 & Vector2::operator+=(const Vector2 &o)
{
    x += o.x;
    y += o.y;

    return *this;
}

constexpr Vector2 & Vector2::operator-=(const Vector2 &o)
{
    x -= o.x;
    y -= o.y;

    return *this;
}

constexpr Vector2 & Vector2::operator+=(float o)
{
    x += o;
    y += o;

    return *this;
}

constexpr Vector2 & Vector2::operator-=(float o)
{
    x -= o;
    y -= o;

    return *this;
}

constexpr Vector2 & Vector2::operator*=(float o)
{
    x *= o;
    y *= o;

    return *this;
}

constexpr Vector2 & Vector2::operator/=(float o)
{
    float inv = 1.f / o;

    return *this *= inv;
}

constexpr Vector2 operator-(Vector2 v)
{
    return Vector2 {
        -v.x,
        -v.y,
    };
}

constexpr Vector2 operator+(Vector2 a, const Vector2 &b)
{
    a += b;

    return a;
}

constexpr Vector2 operator-(Vector2 a, const Vector2 &b)
{
    a -= b;

    return a;
}

constexpr Vector2 operator+(Vector2 a, float b)
{
    a += b;

    return a;
}

constexpr Vector2 operator-(Vector2 a, float b)
{
    a -= b;

    return a;
}

constexpr Vector2 operator*(Vector2 a, float b)
{
    a *= b;

    return a;
}

constexpr Vector2 operator/(Vector2 a, float b)
{
    a /= b;
    return a;
}

constexpr Vector2 operator+(float a, Vector2 b)
{
    return b + a;
}

constexpr Vector2 operator-(float a, Vector2 b)
{
    return -b + a;
}

constexpr Vector2 operator*(float a, Vector2 b)
{
    return b * a;
}

constexpr Vector2 operator/(float a, Vector2 b)
{
    return Vector2 {
        a / b.x,
        a / b.y,
    };
}

Vector2 Vector2::min(Vector2 a, Vector2 b)
{
    return Vector2 {
        fminf(a.x, b.x),
        fminf(a.y, b.y),
    };
}

Vector2 Vector2::max(Vector2 a, Vector2 b)
{
    return Vector2 {
        fmaxf(a.x, b.x),
        fmaxf(a.y, b.y),
    };
}

float Vector3::dot(const Vector3 &o) const
{
    return x * o.x + y * o.y + z * o.z;
}

Vector3 Vector3::cross(const Vector3 &o) const
{
    return Vector3 {
        y * o.z - z * o.y,
        z * o.x - x * o.z,
        x * o.y - y * o.x,
    };
}

// Returns two vectors perpendicular to this vector
// *this should be a normalized
void Vector3::frame(Vector3 *a, Vector3 *b) const
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

float Vector3::length2() const
{
    return x * x + y * y + z * z;
}

float Vector3::length() const
{
    return sqrtf(length2());
}

float Vector3::invLength() const
{
#ifdef MADRONA_GPU_MODE
    return rsqrtf(length2());
#else
    // FIXME: is there a CPU fast path here?
    return 1.f / length();
#endif
}

float Vector3::distance(const Vector3 &o) const
{
    return (*this - o).length();
}

float Vector3::distance2(const Vector3 &o) const
{
    return (*this - o).length2();
}

Vector3 Vector3::normalize() const
{
    return *this * invLength();
} 

constexpr Vector2 Vector3::xy() const
{
    return Vector2 { .x = x, .y = y };
}

constexpr Vector2 Vector3::yz() const
{
    return Vector2 { .x = y, .y = z };
}

constexpr Vector2 Vector3::xz() const
{
    return Vector2 { .x = x, .y = z };
}

constexpr Vector2 Vector3::yx() const
{
    return Vector2 { .x = y, .y = x };
}

constexpr Vector2 Vector3::zy() const
{
    return Vector2 { .x = z, .y = y };
}

constexpr Vector2 Vector3::zx() const
{
    return Vector2 { .x = z, .y = x };
}

float & Vector3::operator[](CountT i)
{
#ifdef MADRONA_GPU_MODE
    // For some reason, the CUDA compiler will sometimes insert a BSYNC 
    // instruction when using the switch statement. The ternary leads
    // to the least amount of instructions.
    return (i == 0) ? x : ((i == 1) ? y : z);
#else
    switch (i) {
        case 0:
            return x;
        case 1:
            return y;
        case 2:
            return z;
        default:
            MADRONA_UNREACHABLE();
    }
#endif
}

float Vector3::operator[](CountT i) const
{
#ifdef MADRONA_GPU_MODE
    // For some reason, the CUDA compiler will sometimes insert a BSYNC 
    // instruction when using the switch statement. The ternary leads
    // to the least amount of instructions.
    return (i == 0) ? x : ((i == 1) ? y : z);
#else
    switch (i) {
        case 0:
            return x;
        case 1:
            return y;
        case 2:
            return z;
        default:
            MADRONA_UNREACHABLE();
    }
#endif
}

constexpr Vector3 & Vector3::operator+=(const Vector3 &o)
{
    x += o.x;
    y += o.y;
    z += o.z;

    return *this;
}

constexpr Vector3 & Vector3::operator-=(const Vector3 &o)
{
    x -= o.x;
    y -= o.y;
    z -= o.z;

    return *this;
}

constexpr Vector3 & Vector3::operator+=(float o)
{
    x += o;
    y += o;
    z += o;

    return *this;
}

constexpr Vector3 & Vector3::operator-=(float o)
{
    x -= o;
    y -= o;
    z -= o;

    return *this;
}

constexpr Vector3 & Vector3::operator*=(float o)
{
    x *= o;
    y *= o;
    z *= o;

    return *this;
}

constexpr Vector3 & Vector3::operator/=(float o)
{
    float inv = 1.f / o;

    return *this *= inv;
}

constexpr Vector3 operator-(Vector3 v)
{
    return Vector3 {
        -v.x,
        -v.y,
        -v.z,
    };
}

constexpr Vector3 operator+(Vector3 a, const Vector3 &b)
{
    a += b;

    return a;
}

constexpr Vector3 operator-(Vector3 a, const Vector3 &b)
{
    a -= b;

    return a;
}

constexpr Vector3 operator+(Vector3 a, float b)
{
    a += b;

    return a;
}

constexpr Vector3 operator-(Vector3 a, float b)
{
    a -= b;

    return a;
}

constexpr Vector3 operator*(Vector3 a, float b)
{
    a *= b;

    return a;
}

constexpr Vector3 operator/(Vector3 a, float b)
{
    a /= b;
    return a;
}

constexpr Vector3 operator+(float a, Vector3 b)
{
    return b + a;
}

constexpr Vector3 operator-(float a, Vector3 b)
{
    return -b + a;
}

constexpr Vector3 operator*(float a, Vector3 b)
{
    return b * a;
}

constexpr Vector3 operator/(float a, Vector3 b)
{
    return Vector3 {
        a / b.x,
        a / b.y,
        a / b.z,
    };
}

Vector3 Vector3::min(Vector3 a, Vector3 b)
{
    return Vector3 {
        fminf(a.x, b.x),
        fminf(a.y, b.y),
        fminf(a.z, b.z),
    };
}

Vector3 Vector3::max(Vector3 a, Vector3 b)
{
    return Vector3 {
        fmaxf(a.x, b.x),
        fmaxf(a.y, b.y),
        fmaxf(a.z, b.z),
    };
}

constexpr Vector3 Vector3::zero()
{
    return Vector3 {
        0,
        0,
        0,
    };
}

constexpr Vector3 Vector3::one()
{
    return Vector3 {
        1,
        1,
        1,
    };
}

constexpr Vector3 Vector3::all(float v)
{
    return Vector3 {
        v, v, v
    };
}

float dot(Vector2 a, Vector2 b)
{
    return a.x * b.x + a.y * b.y;
}

float dot(Vector3 a, Vector3 b)
{
    return a.dot(b);
}

Vector3 cross(Vector3 a, Vector3 b)
{
    return a.cross(b);
}

Vector3 reflect(Vector3 direction, Vector3 normal)
{
    return direction - (2.f * direction.dot(normal) * normal) /
        normal.dot(normal);
}

Mat3x3 outerProduct(Vector3 a, Vector3 b)
{
    return Mat3x3 {{
        a * b.x,
        a * b.y,
        a * b.z,
    }};
}

Vector3 normalize(Vector3 v)
{
    return v.normalize();
}

float sqr(float x)
{
    return x * x;
}

Vector3 Vector4::xyz() const
{
    return Vector3 {
        x,
        y,
        z,
    };
}


float & Vector4::operator[](CountT i)
{
    switch (i) {
        default:
        case 0:
            return x;
        case 1:
            return y;
        case 2:
            return z;
        case 3:
            return w;
    }
}

float Vector4::operator[](CountT i) const
{
    switch (i) {
        default:
        case 0:
            return x;
        case 1:
            return y;
        case 2:
            return z;
        case 3:
            return w;
    }
}

Vector4 Vector4::fromVec3W(Vector3 v, float w)
{
    return Vector4 {
        v.x,
        v.y,
        v.z,
        w,
    };
}

Vector4 Vector4::operator*(float s) const
{
    return Vector4 {
        x * s,
        y * s,
        z * s,
        w * s
    };
}

inline Vector4 Vector4::operator+(const Vector4 &o) const
{
    return Vector4 {
        x + o.x,
        y + o.y,
        z + o.z,
        w + o.w
    };
}

constexpr Vector4 Vector4::zero()
{
    return Vector4 {
        0,
        0,
        0,
        0,
    };
}

constexpr Vector4 Vector4::one()
{
    return Vector4 {
        1,
        1,
        1,
        1,
    };
}


float Quat::length2() const
{
    return w * w + x * x + y * y + z * z;
}

float Quat::length() const
{
    return sqrtf(length2());
}

float Quat::invLength() const
{
#ifdef MADRONA_GPU_MODE
    return rsqrtf(length2());
#else
    return 1.f / sqrtf(length2());
#endif
}

Quat Quat::normalize() const
{
    float inv_length = invLength();

    return Quat {
        w * inv_length,
        x * inv_length,
        y * inv_length,
        z * inv_length,
    };
}

Quat Quat::inv() const
{
    return Quat {
        w,
        -x,
        -y,
        -z,
    };
}

Vector3 Quat::extractPYR() const
{
    float pitch = asin(-2.0f*(x*z - w*y));
    float yaw = std::atan2(2.0f*(y*z + w*x), w*w - x*x - y*y + z*z);
    float roll = std::atan2(2.0f*(x*y + w*z), w*w + x*x - y*y - z*z);

    return {
        pitch, yaw, roll
    };
}

Vector3 Quat::rotateVec(Vector3 v) const
{
    Vector3 pure {x, y, z};
    float scalar = w;

    Vector3 pure_x_v = cross(pure, v);
    Vector3 pure_x_pure_x_v = cross(pure, pure_x_v);
    
    return v + 2.f * ((pure_x_v * scalar) + pure_x_pure_x_v);
}

Quat Quat::angleAxis(float angle, Vector3 normal)
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

Quat Quat::fromAngularVec(Vector3 v)
{
    return Quat {
        0,
        v.x,
        v.y,
        v.z,
    };
}

Quat Quat::fromBasis(Vector3 a, Vector3 b, Vector3 c)
{
    //Modified from glm::quat_cast
#if 0
===============================================================================
The MIT License
-------------------------------------------------------------------------------
Copyright (c) 2005 - G-Truc Creation

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
#endif

    float four_x_squared_minus1 = a.x - b.y - c.z;
    float four_y_squared_minus1 = b.y - a.x - c.z;
    float four_z_squared_minus_1 = c.z - a.x - b.y;
    float four_w_squared_minus1 = a.x + b.y + c.z;

    int biggest_index = 0;
    float four_biggest_squared_minus1 = four_w_squared_minus1;
    if(four_x_squared_minus1 > four_biggest_squared_minus1) {
        four_biggest_squared_minus1 = four_x_squared_minus1;
        biggest_index = 1;
    }

    if(four_y_squared_minus1 > four_biggest_squared_minus1) {
        four_biggest_squared_minus1 = four_y_squared_minus1;
        biggest_index = 2;
    }

    if(four_z_squared_minus_1 > four_biggest_squared_minus1) {
        four_biggest_squared_minus1 = four_z_squared_minus_1;
        biggest_index = 3;
    }

    float biggest_val = sqrtf(four_biggest_squared_minus1 + 1.f) * 0.5f;
    float mult = 0.25f / biggest_val;

    switch(biggest_index) {
    case 0:
        return {
            biggest_val, 
            (b.z - c.y) * mult,
            (c.x - a.z) * mult,
            (a.y - b.x) * mult,
        };
    case 1:
        return {
            (b.z - c.y) * mult,
            biggest_val,
            (a.y + b.x) * mult,
            (c.x + a.z) * mult,
        };
    case 2:
        return {
            (c.x - a.z) * mult,
            (a.y + b.x) * mult,
            biggest_val,
            (b.z + c.y) * mult,
        };
    case 3:
        return {
            (a.y - b.x) * mult,
            (c.x + a.z) * mult,
            (b.z + c.y) * mult,
            biggest_val,
        };
    default: MADRONA_UNREACHABLE();
    }
}

constexpr Quat Quat::id()
{
    return { 1.f, 0.f, 0.f, 0.f };
}

Quat & Quat::operator+=(Quat o)
{
    w += o.w;
    x += o.x;
    y += o.y;
    z += o.z;

    return *this;
}

Quat & Quat::operator-=(Quat o)
{
    w -= o.w;
    x -= o.x;
    y -= o.y;
    z -= o.z;

    return *this;
}

Quat & Quat::operator*=(Quat o)
{
    // Slightly cleaner to implement in terms of operator* because then we
    // don't need to worry about overwriting members that will be used
    // later in the multiplication computation
    return *this = (*this * o);
}

Quat & Quat::operator*=(float f)
{
    w *= f;
    x *= f;
    y *= f;
    z *= f;

    return *this;
}

Quat operator+(Quat a, Quat b)
{
    return a += b;
}

Quat operator-(Quat a, Quat b)
{
    return a -= b;
}

Quat operator*(Quat a, Quat b)
{
    return Quat {
        (a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z),
        (a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y),
        (a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x),
        (a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w),
    };
}

Quat operator*(Quat a, float b)
{
    a *= b;

    return a;
}

Quat operator*(float b, Quat a)
{
    a *= b;

    return a;
}

Diag3x3 Diag3x3::inv() const
{
    return Diag3x3 {
        1.f / d0,
        1.f / d1,
        1.f / d2,
    };
}

Diag3x3 Diag3x3::fromVec(Vector3 v)
{
    return Diag3x3 {
        v.x,
        v.y,
        v.z,
    };
}

constexpr Diag3x3 Diag3x3::uniform(float scale)
{
    return Diag3x3 {
        scale,
        scale,
        scale,
    };
}

constexpr Diag3x3 Diag3x3::id()
{
    return Diag3x3::uniform(1.f);
}

Diag3x3 & Diag3x3::operator*=(Diag3x3 o)
{
    d0 *= o.d0;
    d1 *= o.d1;
    d2 *= o.d2;

    return *this;
}

Diag3x3 & Diag3x3::operator*=(float o)
{
    d0 *= o;
    d1 *= o;
    d2 *= o;

    return *this;
}

Diag3x3 & Diag3x3::operator/=(float o)
{
    d0 /= o;
    d1 /= o;
    d2 /= o;

    return *this;
}

float & Diag3x3::operator[](CountT i)
{
    switch (i) {
        default:
        case 0: return d0;
        case 1: return d1;
        case 2: return d2;
    }
}

float Diag3x3::operator[](CountT i) const
{
    switch (i) {
        default:
        case 0: return d0;
        case 1: return d1;
        case 2: return d2;
    }
}

Diag3x3 operator*(Diag3x3 a, Diag3x3 b)
{
    a *= b;
    return a;
}

Diag3x3 operator*(Diag3x3 a, float b)
{
    a *= b;
    return a;
}

Diag3x3 operator*(float a, Diag3x3 b)
{
    b *= a;
    return b;
}

Diag3x3 operator/(Diag3x3 a, float b)
{
    a /= b;
    return a;
}

Diag3x3 operator/(float a, Diag3x3 b)
{
    return {
        a / b.d0,
        a / b.d1,
        a / b.d2,
    };
}

Vector3 operator*(Diag3x3 d, Vector3 v)
{
    return Vector3 {
        d.d0 * v.x,
        d.d1 * v.y,
        d.d2 * v.z,
    };
}


Mat3x3::Transpose Mat3x3::transpose() const
{
    return Transpose { this };
}

Vector3 Mat3x3::Transpose::operator[](CountT i) const
{
    return Vector3 {
        src->cols[0][i],
        src->cols[1][i],
        src->cols[2][i],
    };
}

Vector3 operator*(Mat3x3::Transpose t, Vector3 v)
{
    return Vector3 {
        dot(t.src->cols[0], v),
        dot(t.src->cols[1], v),
        dot(t.src->cols[2], v),
    };
}

float Mat3x3::determinant() const
{
    Vector3 c0 = cols[0];
    Vector3 c1 = cols[1];
    Vector3 c2 = cols[2];

    return c0.x * (c1.y * c2.z - c2.y * c1.z) -
           c0.y * (c1.x * c2.z - c2.x * c1.z) +
           c0.z * (c1.x * c2.y - c2.x * c1.y);
}

Mat3x3 Mat3x3::fromQuat(Quat r)
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

Mat3x3 Mat3x3::fromRS(Quat r, Diag3x3 s)
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

    Diag3x3 ds = 2.f * s;

    return {{
        { 
            s.d0 - ds.d0 * (y2 + z2),
            ds.d0 * (xy + wz),
            ds.d0 * (xz - wy),
        },
        {
            ds.d1 * (xy - wz),
            s.d1 - ds.d1 * (x2 + z2),
            ds.d1 * (yz + wx),
        },
        {
            ds.d2 * (xz + wy),
            ds.d2 * (yz - wx),
            s.d2 - ds.d2 * (x2 + y2),
        },
    }};
}

Mat3x3 Mat3x3::skewSym(Vector3 v)
{
    return Mat3x3 {{
        { 0, -v.z, v.y },
        { v.z, 0, -v.x },
        { -v.y, v.x, 0 },
    }};
}

Vector3 & Mat3x3::operator[](CountT i)
{
    return cols[i];
}

Vector3 Mat3x3::operator[](CountT i) const
{
    return cols[i];
}

Mat3x3 & Mat3x3::operator+=(const Mat3x3 &o)
{
    cols[0] += o[0];
    cols[1] += o[1];
    cols[2] += o[2];

    return *this;
}

Mat3x3 & Mat3x3::operator-=(const Mat3x3 &o)
{
    cols[0] -= o[0];
    cols[1] -= o[1];
    cols[2] -= o[2];
    
    return *this;
}

Vector3 Mat3x3::operator*(Vector3 v) const
{
    return cols[0] * v.x + cols[1] * v.y + cols[2] * v.z;
}

Mat3x3 Mat3x3::operator*(const Mat3x3 &o) const
{
    return Mat3x3 {
        *this * o[0],
        *this * o[1],
        *this * o[2],
    };
}

Mat3x3 & Mat3x3::operator*=(const Mat3x3 &o)
{
    return *this = (*this * o);
}

Mat3x3 & Mat3x3::operator*=(float s)
{
    cols[0] *= s;
    cols[1] *= s;
    cols[2] *= s;

    return *this;
}

Mat3x3 operator+(Mat3x3 a, const Mat3x3 &b)
{
    return (a += b);
}

Mat3x3 operator-(Mat3x3 a, const Mat3x3 &b)
{
    return (a -= b);
}

Mat3x3 operator*(const Mat3x3 &m, Diag3x3 d)
{
    return Mat3x3 {{
        m[0] * d.d0,
        m[1] * d.d1,
        m[2] * d.d2,
    }};
}

Mat3x3 operator*(Diag3x3 d, const Mat3x3 &m)
{
    return Mat3x3 {{
        { d * m[0] },
        { d * m[1] },
        { d * m[2] },
    }};
}

Mat3x3 operator*(Mat3x3 a, Mat3x3::Transpose b)
{
    return Mat3x3 {
        a * b[0],
        a * b[1],
        a * b[2],
    };
}

Mat3x3 operator*(Mat3x3::Transpose a, Mat3x3 b)
{
    return Mat3x3 {
        a * b[0],
        a * b[1],
        a * b[2],
    };
}

Mat3x3 operator*(float s, const Mat3x3 &m)
{
    return Mat3x3 {{
        s * m[0],
        s * m[1],
        s * m[2],
    }};
}

Mat3x3 operator*(const Mat3x3 &m, float s)
{
    return s * m;
}

Mat3x3 operator/(const Mat3x3 &m, float s)
{
    return Mat3x3 {{
        m[0] / s,
        m[1] / s,
        m[2] / s,
    }};
}

Symmetric3x3 Symmetric3x3::AAT(Mat3x3 A)
{
    auto [a11, a21, a31] = A[0];
    auto [a12, a22, a32] = A[1];
    auto [a13, a23, a33] = A[2];

    return Symmetric3x3 {
        .diag = {
            a11 * a11 + a12 * a12 + a13 * a13,
            a21 * a21 + a22 * a22 + a23 * a23,
            a31 * a31 + a32 * a32 + a33 * a33,
        },
        .off = { 
            a11 * a21 + a12 * a22 + a13 * a23,
            a11 * a31 + a12 * a32 + a13 * a33,
            a21 * a31 + a22 * a32 + a23 * a33,
        },
    };
}

Symmetric3x3 Symmetric3x3::AXAT(Mat3x3 A, Symmetric3x3 X)
{
    auto [a11, a21, a31] = A[0];
    auto [a12, a22, a32] = A[1];
    auto [a13, a23, a33] = A[2];

    auto [x11, x22, x33] = X.diag;
    auto [x12, x13, x23] = X.off;

    float a11x11_a12x12_a13x13 = a11 * x11 + a12 * x12 + a13 * x13;
    float a11x12_a12x22_a13x23 = a11 * x12 + a12 * x22 + a13 * x23;
    float a11x13_a12x23_a13x33 = a11 * x13 + a12 * x23 + a13 * x33;

    float a21x11_a22x12_a23x13 = a21 * x11 + a22 * x12 + a23 * x13;
    float a21x12_a22x22_a23x23 = a21 * x12 + a22 * x22 + a23 * x23;
    float a21x13_a22x23_a23x33 = a21 * x13 + a22 * x23 + a23 * x33;

    return Symmetric3x3 {
        .diag = {
            a11 * a11x11_a12x12_a13x13 + a12 * a11x12_a12x22_a13x23 +
                a13 * a11x13_a12x23_a13x33,
            a21 * a21x11_a22x12_a23x13 + a22 * a21x12_a22x22_a23x23 +
                a23 * a21x13_a22x23_a23x33,
            a31 * (a31 * x11 + a32 * x12 + a33 * x13) +
                a32 * (a31 * x12 + a32 * x22 + a33 * x23) +
                a33 * (a31 * x13 + a32 * x23 + a33 * x33),
        },
        .off = {
            a21 * a11x11_a12x12_a13x13 + a22 * a11x12_a12x22_a13x23 +
                a23 * a11x13_a12x23_a13x33,
            a31 * a11x11_a12x12_a13x13 + a32 * a11x12_a12x22_a13x23 +
                a33 * a11x13_a12x23_a13x33,
            a31 * a21x11_a22x12_a23x13 + a32 * a21x12_a22x22_a23x23 +
                a33 * a21x13_a22x23_a23x33,
        },
    };
}

Symmetric3x3 Symmetric3x3::vvT(Vector3 v)
{
    return Symmetric3x3 {
        .diag = { v.x * v.x, v.y * v.y, v.z * v.z },
        .off = { v.x * v.y, v.x * v.z, v.y * v.z },
    };
}

Vector3 Symmetric3x3::operator[](CountT i) const
{
    switch (i) {
    case 0:
        return { diag[0], off[0], off[1] };
    case 1:
        return { off[0], diag[1], off[2] };
    case 2:
        return { off[1], off[2], diag[2] };
    default: MADRONA_UNREACHABLE();
    }
}

Symmetric3x3 & Symmetric3x3::operator+=(const Symmetric3x3 &o)
{
    diag += o.diag;
    off += o.off;

    return *this;
}

Symmetric3x3 & Symmetric3x3::operator-=(const Symmetric3x3 &o)
{
    diag -= o.diag;
    off -= o.off;

    return *this;
}

Symmetric3x3 & Symmetric3x3::operator*=(const Symmetric3x3 &o)
{
    return *this = (*this * o);
}

Symmetric3x3 & Symmetric3x3::operator*=(float s)
{
    diag *= s;
    off *= s;
    return *this;
}

Symmetric3x3 operator+(Symmetric3x3 a, Symmetric3x3 b)
{
    a += b;
    return a;
}

Symmetric3x3 operator-(Symmetric3x3 a, Symmetric3x3 b)
{
    a -= b;
    return a;
}

Symmetric3x3 operator*(Symmetric3x3 a, Symmetric3x3 b)
{
    auto [a11, a22, a33] = a.diag;
    auto [a12, a13, a23] = a.off;
    auto [b11, b22, b33] = b.diag;
    auto [b12, b13, b23] = a.off;

    return Symmetric3x3 { 
        .diag = {
            a11 * b11 + a12 * b12 + a13 * b13,
            a12 * b12 + a22 * b22 + a23 * b23,
            a13 * b13 + a23 * b23 + a33 * b33,
        },
        .off = {
            a11 * b12 + a12 * b22 + a13 * b23,
            a11 * b13 + a12 * b23 + a13 * b33,
            a12 * b13 + a22 * b23 + a23 * b33,
        },
    };
}

Symmetric3x3 operator*(Symmetric3x3 a, float b)
{
    a *= b;
    return a;
}

Symmetric3x3 operator*(float a, Symmetric3x3 b)
{
    b *= a;
    return b;
}

Vector3 Mat3x4::txfmPoint(Vector3 p) const
{
    return cols[0] * p.x + cols[1] * p.y + cols[2] * p.z + cols[3];
}

Vector3 Mat3x4::txfmDir(Vector3 p) const
{
    return cols[0] * p.x + cols[1] * p.y + cols[2] * p.z;
}

Mat3x4 Mat3x4::compose(const Mat3x4 &o) const
{
    return Mat3x4 {
        txfmDir(o.cols[0]),
        txfmDir(o.cols[1]),
        txfmDir(o.cols[2]),
        txfmPoint(o.cols[3]),
    };
}

void Mat3x4::decompose(Vector3 *out_translation,
                       Quat *out_rotation,
                       Diag3x3 *out_scale) const
{
    Diag3x3 scale {
        cols[0].length(),
        cols[1].length(),
        cols[2].length(),
    };

    if (dot(cross(cols[0], cols[1]), cols[2]) < 0.f) {
        scale.d0 *= -1.f;
    }

    Vector3 v1 = cols[0] / scale.d0;
    Vector3 v2 = cols[1] / scale.d1;
    Vector3 v3 = cols[2] / scale.d2;

    v2 = normalize(v2 - dot(v2, v1) * v1);
    v3 = v3 - dot(v3, v1) * v1;
    v3 -= dot(v3, v2) * v2;
    v3 = normalize(v3);

    Quat rot = Quat::fromBasis(v1, v2, v3);

    *out_translation = cols[3];
    *out_rotation = rot;
    *out_scale = scale;
}

Mat3x4 Mat3x4::fromRows(Vector4 row0, Vector4 row1, Vector4 row2)
{
    return {
        Vector3 { row0.x, row1.x, row2.x },
        Vector3 { row0.y, row1.y, row2.y },
        Vector3 { row0.z, row1.z, row2.z },
        Vector3 { row0.w, row1.w, row2.w },
    };
}

Mat3x4 Mat3x4::fromTRS(Vector3 t, Quat r, Diag3x3 s)
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

    Diag3x3 ds = 2.f * s;

    return {{
        { 
            s.d0 - ds.d0 * (y2 + z2),
            ds.d0 * (xy + wz),
            ds.d0 * (xz - wy),
        },
        {
            ds.d1 * (xy - wz),
            s.d1 - ds.d1 * (x2 + z2),
            ds.d1 * (yz + wx),
        },
        {
            ds.d2 * (xz + wy),
            ds.d2 * (yz - wx),
            s.d2 - ds.d2 * (x2 + y2),
        },
        t,
    }};
}

constexpr Mat3x4 Mat3x4::identity()
{
    return Mat3x4 {{
        { 1, 0, 0 },
        { 0, 1, 0 },
        { 0, 0, 1 },
        { 0, 0, 0 },
    }};
}


Vector4 Mat4x4::txfmPoint(Vector4 p) const
{
    return cols[0] * p.x + cols[1] * p.y + cols[2] * p.z + cols[3] * p.w;
}

Mat4x4 Mat4x4::compose(const Mat4x4 &o) const
{
    return Mat4x4 {
        txfmPoint(o.cols[0]),
        txfmPoint(o.cols[1]),
        txfmPoint(o.cols[2]),
        txfmPoint(o.cols[3]),
    };
}

constexpr Mat4x4 Mat4x4::identity()
{
    return Mat4x4 {{
        { 1, 0, 0, 0 },
        { 0, 1, 0, 0 },
        { 0, 0, 1, 0 },
        { 0, 0, 0, 1 },
    }};
}

float AABB::surfaceArea() const
{
    Vector3 d = pMax - pMin;
    return 2.f * (d.x * d.y + d.x * d.z + d.y * d.z);
}

float AABB::distance2(const AABB &o) const
{
    float dist2 = 0.f;
    MADRONA_UNROLL
    for (CountT i = 0; i < 3; i++) {
        float isect_min = fmaxf(pMin[i], o.pMin[i]);
        float isect_max = fminf(pMax[i], o.pMax[i]);

        float diff = isect_min - isect_max;
        if (diff > 0) {
            dist2 += diff * diff;
        }
    }

    return dist2;
}

inline Vector3 AABB::centroid() const
{
    return 0.5f * (pMin + pMax);
}

inline int AABB::maxDimension() const
{
    Vector3 d = pMax - pMin;

    if (d.x > d.y && d.x > d.z) {
        return 0;
    } else if (d.y > d.z) {
        return 1;
    } else {
        return 2;
    }
}

inline Vector3 AABB::offset(const Vector3 &p) const
{
    Vector3 o = p - pMin;

    if (pMax.x > pMin.x) {
        o.x /= pMax.x - pMin.x;
    }

    if (pMax.y > pMin.y) {
        o.y /= pMax.y - pMin.y;
    }

    if (pMax.z > pMin.z) {
        o.z /= pMax.z - pMin.z;
    }

    return o;
}

bool AABB::overlaps(const AABB &o) const
{
    auto [a_min, a_max] = *this;
    auto [b_min, b_max] = o;

    return a_min.x < b_max.x && b_min.x < a_max.x &&
           a_min.y < b_max.y && b_min.y < a_max.y &&
           a_min.z < b_max.z && b_min.z < a_max.z;
}

bool AABB::intersects(const AABB &o) const
{
    auto [a_min, a_max] = *this;
    auto [b_min, b_max] = o;

    // SAT
    bool x_sep = a_max.x < b_min.x || a_min.x > b_max.x;
    bool y_sep = a_max.y < b_min.y || a_min.y > b_max.y;
    bool z_sep = a_max.z < b_min.z || a_min.z > b_max.z;

    return !x_sep && !y_sep && !z_sep;
}

bool AABB::contains(const AABB &o) const
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

bool AABB::contains(const Vector3 &p) const
{
    auto [a_min, a_max] = *this;

    return a_min.x <= p.x &&
           a_min.y <= p.y &&
           a_min.z <= p.z &&
           a_max.x >= p.x &&
           a_max.y >= p.y &&
           a_max.z >= p.z; 
}

void AABB::expand(const Vector3 &p)
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

bool AABB::rayIntersects(Vector3 ray_o, Diag3x3 inv_ray_d,
                         float ray_t_min, float ray_t_max)
{
    // Ray tracing gems II, chapter 2
    
    // Absolute distances to lower and upper box coordinates
    math::Vector3 t_lower = inv_ray_d * (pMin - ray_o);
    math::Vector3 t_upper = inv_ray_d * (pMax - ray_o);
    // The four t-intervals (for x-/y-/z-slabs, and ray p(t))
    math::Vector4 t_mins =
        Vector4::fromVec3W(Vector3::min(t_lower, t_upper), ray_t_min);
    math::Vector4 t_maxes = 
        Vector4::fromVec3W(Vector3::max(t_lower, t_upper), ray_t_max);
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

bool AABB::rayIntersects(Vector3 ray_o, Diag3x3 inv_ray_d,
                         float ray_t_min, float ray_t_max,
                         float &hit_t, float &far_t)
{
    // Ray tracing gems II, chapter 2
    
    // Absolute distances to lower and upper box coordinates
    math::Vector3 t_lower = inv_ray_d * (pMin - ray_o);
    math::Vector3 t_upper = inv_ray_d * (pMax - ray_o);
    // The four t-intervals (for x-/y-/z-slabs, and ray p(t))
    math::Vector4 t_mins =
        Vector4::fromVec3W(Vector3::min(t_lower, t_upper), ray_t_min);
    math::Vector4 t_maxes = 
        Vector4::fromVec3W(Vector3::max(t_lower, t_upper), ray_t_max);
    // Easy to remember: ``max of mins, and min of maxes''

    auto max_component = [](Vector4 v) {
        return fmaxf(v.x, fmaxf(v.y, fmaxf(v.z, v.w)));
    };

    auto min_component = [](Vector4 v) {
        return fminf(v.x, fminf(v.y, fminf(v.z, v.w)));
    };
   
    float t_box_min = max_component(t_mins);
    float t_box_max = min_component(t_maxes);

    if (t_box_min <= t_box_max) {
        hit_t = t_box_min;
        far_t = t_box_max;
        return true;
    } else {
        // No intersection
        hit_t = FLT_MAX;
        far_t = FLT_MAX;
        return false;
    }
}

AABB AABB::applyTRS(const Vector3 &translation,
                    const Quat &rotation,
                    const Diag3x3 &scale) const
{
    // FIXME: this could all be more efficient with a center + width
    // AABB representation
    // FIXME: this matrix should to be row major
    auto rot_mat = Mat3x3::fromRS(rotation, scale);

     // RTCD page 86
     AABB txfmed;
MADRONA_UNROLL
     for (CountT i = 0; i < 3; i++) {
         txfmed.pMin[i] = txfmed.pMax[i] = translation[i];

MADRONA_UNROLL
         for (CountT j = 0; j < 3; j++) {
             // Flipped because rot_mat is column major
             float e = rot_mat[j][i] * pMin[j];
             float f = rot_mat[j][i] * pMax[j];

             if (e < f) {
                 txfmed.pMin[i] += e;
                 txfmed.pMax[i] += f;
             } else {
                 txfmed.pMin[i] += f;
                 txfmed.pMax[i] += e;
             }
         }
     }

     return txfmed;
}

float AABB::operator[](CountT i) const
{
    switch (i) {
        case 0: return pMin.x;
        case 1: return pMin.y;
        case 2: return pMin.z;
        case 3: return pMax.x;
        case 4: return pMax.y;
        case 5: return pMax.z;
        default: MADRONA_UNREACHABLE();
    }
}

AABB AABB::invalid()
{
    return AABB {
        /* .pMin = */ Vector3 {FLT_MAX, FLT_MAX, FLT_MAX},
        /* .pMax = */ Vector3 {-FLT_MAX, -FLT_MAX, -FLT_MAX},
    };
}

AABB AABB::point(const Vector3 &p)
{
    return AABB {
        /* .pMin = */ p,
        /* .pMax = */ p,
    };
}

AABB AABB::merge(const AABB &a, const AABB &b)
{
    return AABB {
        /* .pMin = */ Vector3::min(a.pMin, b.pMin),
        /* .pMax = */ Vector3::max(a.pMax, b.pMax),
    };
}

Vector2 AABB2D::centroid() const
{
    return 0.5f * (pMin + pMax);
}

float AABB2D::area() const
{
    Vector2 diff = pMax - pMin;
    return diff.x * diff.y;
}

}
