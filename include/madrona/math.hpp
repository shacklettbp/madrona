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
#include <madrona/macros.hpp>

namespace madrona {
namespace math {

struct Vector2;
struct Vector3;
struct Vector4;
struct Quat;
struct Mat3x3;
struct Mat3x4;

constexpr inline float pi {3.14159265358979323846264338327950288f};
constexpr inline float pi_d2 {pi / 2.f};
constexpr inline float pi_m2 {pi * 2.f};

inline constexpr float toRadians(float degrees);
inline constexpr float rsqrtApprox(float v);

// Solves a * x^2 + b * x + c = 0. Assumes a is not 0.
inline bool solveQuadraticUnsafe(
    float a, float b, float c, float *t1, float *t2);

struct Vector2 {
    float x;
    float y;

    inline float dot(const Vector2 &o) const;

    inline float length2() const;
    inline float length() const;
    inline float invLength() const;

    inline float & operator[](CountT i);
    inline float operator[](CountT i) const;

    constexpr inline Vector2 & operator+=(const Vector2 &o);
    constexpr inline Vector2 & operator-=(const Vector2 &o);

    constexpr inline Vector2 & operator+=(float o);
    constexpr inline Vector2 & operator-=(float o);
    constexpr inline Vector2 & operator*=(float o);
    constexpr inline Vector2 & operator/=(float o);

    constexpr friend inline Vector2 operator+(Vector2 v);
    constexpr friend inline Vector2 operator-(Vector2 v);

    constexpr friend inline Vector2 operator+(Vector2 a, const Vector2 &b);
    constexpr friend inline Vector2 operator-(Vector2 a, const Vector2 &b);

    constexpr friend inline Vector2 operator+(Vector2 a, float b);
    constexpr friend inline Vector2 operator-(Vector2 a, float b);
    constexpr friend inline Vector2 operator*(Vector2 a, float b);
    constexpr friend inline Vector2 operator/(Vector2 a, float b);

    constexpr friend inline Vector2 operator+(float a, Vector2 b);
    constexpr friend inline Vector2 operator-(float a, Vector2 b);
    constexpr friend inline Vector2 operator*(float a, Vector2 b);
    constexpr friend inline Vector2 operator/(float a, Vector2 b);

    static inline Vector2 min(Vector2 a, Vector2 b);
    static inline Vector2 max(Vector2 a, Vector2 b);
};

struct Vector3 {
    float x;
    float y;
    float z;

    inline float dot(const Vector3 &o) const;
    inline Vector3 cross(const Vector3 &o) const;

    // Returns two vectors perpendicular to this vector
    // *this should be a normalized
    inline void frame(Vector3 *a, Vector3 *b) const;

    inline float length2() const;
    inline float length() const;
    inline float invLength() const;

    inline float distance(const Vector3 &o) const;
    inline float distance2(const Vector3 &o) const;

    [[nodiscard]] inline Vector3 normalize() const;

    constexpr inline Vector2 xy() const;
    constexpr inline Vector2 yz() const;
    constexpr inline Vector2 xz() const;
    constexpr inline Vector2 yx() const;
    constexpr inline Vector2 zy() const;
    constexpr inline Vector2 zx() const;

    inline float & operator[](CountT i);
    inline float operator[](CountT i) const;

    constexpr inline Vector3 & operator+=(const Vector3 &o);
    constexpr inline Vector3 & operator-=(const Vector3 &o);

    constexpr inline Vector3 & operator+=(float o);
    constexpr inline Vector3 & operator-=(float o);
    constexpr inline Vector3 & operator*=(float o);
    constexpr inline Vector3 & operator/=(float o);

    constexpr friend inline Vector3 operator-(Vector3 v);

    constexpr friend inline Vector3 operator+(Vector3 a, const Vector3 &b);
    constexpr friend inline Vector3 operator-(Vector3 a, const Vector3 &b);

    constexpr friend inline Vector3 operator+(Vector3 a, float b);
    constexpr friend inline Vector3 operator-(Vector3 a, float b);
    constexpr friend inline Vector3 operator*(Vector3 a, float b);
    constexpr friend inline Vector3 operator/(Vector3 a, float b);

    constexpr friend inline Vector3 operator+(float a, Vector3 b);
    constexpr friend inline Vector3 operator-(float a, Vector3 b);
    constexpr friend inline Vector3 operator*(float a, Vector3 b);
    constexpr friend inline Vector3 operator/(float a, Vector3 b);

    static inline Vector3 min(Vector3 a, Vector3 b);
    static inline Vector3 max(Vector3 a, Vector3 b);

    static constexpr inline Vector3 zero();
    static constexpr inline Vector3 one();
    static constexpr inline Vector3 all(float v);
};

inline float dot(Vector2 a, Vector2 b);
inline float dot(Vector3 a, Vector3 b);
inline Vector3 cross(Vector3 a, Vector3 b);
inline Mat3x3 outerProduct(Vector3 a, Vector3 b);
inline Vector3 normalize(Vector3 v);
// Reflects the vector `direction` over the `normal`
inline Vector3 reflect(Vector3 direction, Vector3 normal);
inline float sqr(float x);

struct Vector4 {
    float x;
    float y;
    float z;
    float w;

    inline Vector3 xyz() const;

    inline float & operator[](CountT i);
    inline float operator[](CountT i) const;

    static inline Vector4 fromVec3W(Vector3 v, float w);

    inline Vector4 operator*(float a) const;
    inline Vector4 operator+(const Vector4 &) const;

    static constexpr inline Vector4 zero();
    static constexpr inline Vector4 one();
};

struct Quat {
    float w;
    float x;
    float y;
    float z;

    inline float length2() const;
    inline float length() const;
    inline float invLength() const;

    [[nodiscard]] inline Quat normalize() const;
    [[nodiscard]] inline Quat inv() const;

    inline Vector3 rotateVec(Vector3 v) const;

    static inline Quat angleAxis(float angle, Vector3 normal);
    static inline Quat fromAngularVec(Vector3 v);
    static inline Quat fromBasis(Vector3 a, Vector3 b, Vector3 c);
    static constexpr inline Quat id();

    inline Quat & operator+=(Quat o);
    inline Quat & operator-=(Quat o);
    inline Quat & operator*=(Quat o);
    inline Quat & operator*=(float f);

    friend inline Quat operator+(Quat a, Quat b);
    friend inline Quat operator-(Quat a, Quat b);
    friend inline Quat operator*(Quat a, Quat b);
    friend inline Quat operator*(Quat a, float b);
    friend inline Quat operator*(float b, Quat a);
};

struct Diag3x3 {
    float d0;
    float d1;
    float d2;

    inline Diag3x3 inv() const;

    static inline Diag3x3 fromVec(Vector3 v);

    static constexpr inline Diag3x3 uniform(float scale);
    static constexpr inline Diag3x3 id();

    inline Diag3x3 & operator*=(Diag3x3 o);
    inline Diag3x3 & operator*=(float o);
    inline Diag3x3 & operator/=(float o);

    inline float & operator[](CountT i);
    inline float operator[](CountT i) const;

    friend inline Diag3x3 operator*(Diag3x3 a, Diag3x3 b);
    friend inline Diag3x3 operator*(Diag3x3 a, float b);
    friend inline Diag3x3 operator*(float a, Diag3x3 b);
    friend inline Vector3 operator*(Diag3x3 d, Vector3 v);

    friend inline Diag3x3 operator/(Diag3x3 a, float b);
    friend inline Diag3x3 operator/(float a, Diag3x3 b);
};

struct Mat3x3 {
    struct Transpose {
        const Mat3x3 *src;

        inline Vector3 operator[](CountT i) const;

        friend inline Vector3 operator*(Transpose t, Vector3 v);
    };

    Vector3 cols[3];

    inline float determinant() const;
    inline Transpose transpose() const;

    static inline Mat3x3 fromQuat(Quat r);
    static inline Mat3x3 fromRS(Quat r, Diag3x3 s);

    inline Vector3 & operator[](CountT i);
    inline Vector3 operator[](CountT i) const;

    inline Mat3x3 & operator+=(const Mat3x3 &o);
    inline Mat3x3 & operator-=(const Mat3x3 &o);

    inline Vector3 operator*(Vector3 v) const;
    inline Mat3x3 operator*(const Mat3x3 &o) const;
    inline Mat3x3 & operator*=(const Mat3x3 &o);
    inline Mat3x3 & operator*=(float s);

    friend inline Mat3x3 operator+(Mat3x3 a, const Mat3x3 &b);
    friend inline Mat3x3 operator-(Mat3x3 a, const Mat3x3 &b);

    friend inline Mat3x3 operator*(const Mat3x3 &m, Diag3x3 d);
    friend inline Mat3x3 operator*(Diag3x3 d, const Mat3x3 &m);

    friend inline Mat3x3 operator*(Mat3x3 a, Mat3x3::Transpose b);
    friend inline Mat3x3 operator*(Mat3x3::Transpose a, Mat3x3 b);

    friend inline Mat3x3 operator*(float s, const Mat3x3 &m);
    friend inline Mat3x3 operator*(const Mat3x3 &m, float s);

    friend inline Mat3x3 operator/(const Mat3x3 &m, float s);
};

struct Symmetric3x3 {
    Vector3 diag;
    Vector3 off;

    // Computes A * A^T
    static inline Symmetric3x3 AAT(Mat3x3 A);
    // Computes A * X * A^T
    static inline Symmetric3x3 AXAT(Mat3x3 A, Symmetric3x3 X);
    // Computes v * v^T
    static inline Symmetric3x3 vvT(Vector3 v);

    inline Vector3 operator[](CountT i) const;

    inline Symmetric3x3 & operator+=(const Symmetric3x3 &o);
    inline Symmetric3x3 & operator-=(const Symmetric3x3 &o);
    inline Symmetric3x3 & operator*=(const Symmetric3x3 &o);
    inline Symmetric3x3 & operator*=(float s);

    friend inline Symmetric3x3 operator+(Symmetric3x3 a, Symmetric3x3 b);
    friend inline Symmetric3x3 operator-(Symmetric3x3 a, Symmetric3x3 b);

    friend inline Symmetric3x3 operator*(Symmetric3x3 a, Symmetric3x3 b);
    friend inline Symmetric3x3 operator*(Symmetric3x3 a, float b);
    friend inline Symmetric3x3 operator*(float a, Symmetric3x3 b);
};

struct Mat3x4 {
    Vector3 cols[4];

    inline Vector3 txfmPoint(Vector3 p) const;
    inline Vector3 txfmDir(Vector3 p) const;

    inline Mat3x4 compose(const Mat3x4 &o) const;

    inline void decompose(Vector3 *translation,
                          Quat *rotation,
                          Diag3x3 *scale) const;

    static inline Mat3x4 fromRows(Vector4 row0, Vector4 row1, Vector4 row2);
    static inline Mat3x4 fromTRS(Vector3 t, Quat r,
                                 Diag3x3 s = { 1.f, 1.f, 1.f });

    static constexpr inline Mat3x4 identity();
};

struct Mat4x4 {
    Vector4 cols[4];

    inline Vector4 txfmPoint(Vector4 p) const;
    inline Vector4 txfmDir(Vector4 p) const;

    inline Mat4x4 compose(const Mat4x4 &o) const;

    static constexpr inline Mat4x4 identity();
};

struct AABB {
    Vector3 pMin;
    Vector3 pMax;

    inline float surfaceArea() const;
    inline float distance2(const AABB &o) const;
    inline Vector3 centroid() const;
    inline int maxDimension() const;
    inline Vector3 offset(const Vector3 &p) const;

    inline bool overlaps(const AABB &o) const;
    // intersects returns true if AABBs are overlapping or if
    // they are exactly touching with no overlap.
    inline bool intersects(const AABB &o) const;

    inline bool contains(const AABB &o) const;
    inline bool contains(const Vector3 &p) const;
    inline void expand(const Vector3 &p);

    inline bool rayIntersects(Vector3 ray_o, Diag3x3 inv_ray_d,
                              float ray_t_min, float ray_t_max);

    inline bool rayIntersects(Vector3 ray_o, Diag3x3 inv_ray_d,
                              float ray_t_min, float ray_t_max,
                              float &hit_t, float &far_t);

    [[nodiscard]] inline AABB applyTRS(const Vector3 &translation,
                                       const Quat &rotation,
                                       const Diag3x3 &scale = {1, 1, 1}) const;

    inline float operator[](CountT i) const;

    static inline AABB invalid();
    static inline AABB point(const Vector3 &p);
    static inline AABB merge(const AABB &a, const AABB &b);
};

struct AABB2D {
    Vector2 pMin;
    Vector2 pMax;

    inline Vector2 centroid() const;
    inline float area() const;
};

constexpr inline Vector3 up { 0, 0, 1 };
constexpr inline Vector3 fwd { 0, 1, 0 };
constexpr inline Vector3 right { 1, 0, 0 };

}

constexpr inline math::Vector3 worldUp = math::up;
constexpr inline math::Vector3 worldFwd = math::fwd;
constexpr inline math::Vector3 worldRight = math::right;
}

#include "math.inl"
