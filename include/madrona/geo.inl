namespace madrona::geo {

inline float intersect2DRayOriginCircle(
    math::Vector2 o,
    math::Vector2 d,
    float r)
{
    using namespace math;

    float r2 = r * r;
    float o_len2 = o.length2();

    float cylinder_hit_t;
    if (o_len2 <= r2) {
        // Ray starts inside circle 
        return 0.f;
    } else {
        // Need to find closest point between line and circle
        // Circle equation: x^2 + y^2 = r^2
        // ray(t) = o + t * d 
        // Substitute & solve:
        // (o.x + t * d.x)^2 +
        //     (o.y + t * d.y)^2 = r^2
        // Rearrange to quadratic equation:
        // a * t^2 + b * t + c = 0
        // a = d.x^2 + d.y^2
        // b = 2 * (d.x * o.x + d.y * o.y)
        // c = o.x * o.x + o.y * o.y - r^2

        float a = d.x * d.x + d.y * d.y;
        float b = 2.f * (d.x * o.x + d.y * o.y);
        float c = o.x * o.x + o.y * o.y - r2;

        // This ray is parallel to the Z axis and starts outside the cylinder
        // Special exit necessary to avoid div by 0
        if (a == 0.f) {
            return FLT_MAX;
        }

        float t1, t2;
        if (!solveQuadraticUnsafe(a, b, c, &t1, &t2)) {
            return FLT_MAX;
        }

        float t = fminf(t1, t2);
        if (t < 0) {
            return FLT_MAX;
        }

        return t;
    }
}

inline float intersectRayOriginSphere(
    math::Vector3 ray_o,
    math::Vector3 ray_d,
    float r)
{
    using namespace math;

    float a = ray_d.length2();
    float b = 2.f * dot(ray_d, ray_o);
    float c = ray_o.length2() - r * r;

    if (c <= 0.f) {
        return 0.f;
    }

    float t1, t2;
    if (!solveQuadraticUnsafe(a, b, c, &t1, &t2)) {
        return FLT_MAX;
    }

    return fminf(t1, t2);
}

inline float intersectRayZOriginCapsule(
    math::Vector3 ray_o,
    math::Vector3 ray_d,
    float r,
    float h)
{
    using namespace math;

    Vector2 o_2d { .x = ray_o.x, .y = ray_o.y };
    Vector2 d_2d { .x = ray_d.x, .y = ray_d.y };

    float cylinder_hit_t = intersect2DRayOriginCircle(
        o_2d, d_2d, r);

    if (cylinder_hit_t == FLT_MAX) {
        return FLT_MAX;
    }

    float cylinder_hit_z = ray_o.z + ray_d.z * cylinder_hit_t;

    if (cylinder_hit_z >= 0.f && cylinder_hit_z <= h) {
        return cylinder_hit_t;
    }

    float lower_sphere_t = intersectRayOriginSphere(ray_o, ray_d, r);

    Vector3 upper_ray_o = ray_o;
    upper_ray_o.z -= h;
    float upper_sphere_t = intersectRayOriginSphere(upper_ray_o, ray_d, r);

    return fminf(lower_sphere_t, upper_sphere_t);
}

}
