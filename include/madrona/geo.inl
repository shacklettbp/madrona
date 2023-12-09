namespace madrona::geo {

inline float intersect2DRayOriginCircle(
    math::Vector2 o,
    math::Vector2 d,
    float r)
{
    using namespace math;

    float r2 = r * r;
    float o_len2 = o.length2();

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

inline math::Vector3 computeTriangleGeoNormal(
    math::Vector3 ab,
    math::Vector3 ac,
    math::Vector3 bc)
{
    // Normals can be inaccurate even using Newell's method when one
    // edge is very long.
    // https://box2d.org/posts/2014/01/troublesome-triangle/
    // 
	// Always will use ab as the base, pick shorter of the other two
    // to compute cross product.

    math::Vector3 normal_bc = cross(ab, bc);
	math::Vector3 normal_ac = cross(ab, ac);
    return bc.length2() < ac.length2() ? normal_bc : normal_ac;
}

inline math::Vector3 triangleClosestPointToOrigin(
    math::Vector3 a,
    math::Vector3 b,
    math::Vector3 c,
    math::Vector3 ab,
    math::Vector3 ac)
{
    using namespace math;
    // RTCD 5.1.5. Assumes P is at origin

    // Check if P in vertex region outside A
    float d1 = dot(ab, a);
    float d2 = dot(ac, a);
    if (d1 >= 0.f && d2 >= 0.f) {
        return a; // barycentric coordinates (1,0,0)
    }

    // Check if P in vertex region outside B
    float d3 = dot(ab, b);
    float d4 = dot(ac, b);
    if (d3 <= 0.f && d3 <= d4) {
        return b; // barycentric coordinates (0,1,0)
    }

    // Check if P in edge region of AB, if so return projection of P onto AB
    float vc = d1 * d4 - d3 * d2;
    if (vc <= 0.f && d1 <= 0.f && d3 >= 0.f) {
        float v = d1 / (d1 - d3);
        return a + v * ab; // barycentric coordinates (1-v,v,0)
    }

    // Check if P in vertex region outside C
    float d5 = dot(ab, c);
    float d6 = dot(ac, c);
    if (d6 <= 0.f && d5 >= d6) {
        return c; // barycentric coordinates (0,0,1)
    }
        
    // Check if P in edge region of AC, if so return projection of P onto AC
    float vb = d5 * d2 - d1 * d6;
    if (vb <= 0.f && d2 <= 0.f && d6 >= 0.f) {
        float w = d2 / (d2 - d6);
        return a + w * ac; // barycentric coordinates (1-w,0,w)
    }

    // Check if P in edge region of BC, if so return projection of P onto BC
    float va = d3 * d6 - d5 * d4;
    if (va <= 0.f && (d4 - d3) <= 0.f && (d5 - d6) <= 0.f) {
        float w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return b + w * (c - b); // barycentric coordinates (0,1-w,w)
    }

    // P inside face region. Compute Q through its barycentric coordinates
    // (u,v,w)
    float denom = 1.f / (va + vb + vc);
    float v = vb * denom;
    float w = vc * denom;
    return a + ab * v + ac * w; //=u*a+v*b+w*c,u=va*denom=1.0f-v-w
}

}
