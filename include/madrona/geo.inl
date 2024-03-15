namespace madrona::geo {

template <typename Fn>
void HalfEdgeMesh::iterateFaceIndices(uint32_t face, Fn &&fn) const
{
    uint32_t hedge_idx = faceBaseHalfEdges[face];
    uint32_t start = hedge_idx;

    do {
        const HalfEdge &hedge = halfEdges[hedge_idx];
        fn(hedge.rootVertex);

        hedge_idx = hedge.next;
    } while (hedge_idx != start);
}

uint32_t HalfEdgeMesh::twinIDX(uint32_t half_edge_id) const
{
    if ((half_edge_id & 1) == 1) {
        return half_edge_id & 0xFFFF'FFFE;
    } else {
        return half_edge_id | 1;
    }
}

uint32_t HalfEdgeMesh::numEdges() const
{
    return numHalfEdges / 2;
}

uint32_t HalfEdgeMesh::edgeToHalfEdge(uint32_t edge_id) const
{
    return edge_id * 2;
}

inline float intersectRayOriginSphere(
    math::Vector3 ray_o,
    math::Vector3 ray_d,
    float r)
{
    using namespace math;

    // Accurate ray-sphere collision from Chapter 7 RT Gems I:
    // (see explanation below in capsule routine)

    // Counter intuitive that we compute c here and then derive discriminant
    // without it, but this is essential for precision. Note that the
    // early out will only trigger when o is close to the sphere, where
    // precision isn't an issue for this check.
    float r2 = math::sqr(r);
    float c = ray_o.length2() - r2;
    if (c <= 0.f) {
        return 0.f;
    }

    float b_prime = -dot(ray_o, ray_d);

    if (b_prime < 0.f) {
        return FLT_MAX;
    }

    float l2 = (ray_o + b_prime * ray_d).length2();
    float delta = r2 - l2;

    if (delta < 0.f) {
        return FLT_MAX;
    }

    float q = b_prime + sqrtf(delta);
    return c / q;
}

inline float intersectRayZOriginCapsule(
    math::Vector3 ray_o,
    math::Vector3 ray_d,
    float r,
    float h)
{
    using namespace math;

    auto intersect2DRayOriginCircle = [](
        Vector2 o,
        Vector2 d,
        float r)
    {
        // The naive way to do this is with the quadratic equation, as below:
        // Unfortunately, this is SUPER imprecise when ray_o is moderately
        // ~500? units away from the origin. Better solution from RT gems 
        // afterwards.
        //
        // Need to find closest point between line and circle
        // Circle equation: x^2 + y^2 = r^2
        // ray(t) = o + t * d 
        // Substitute & solve:
        // (o.x + t * d.x)^2 +
        //     (o.y + t * d.y)^2 = r^2
        // Rearrange to quadratic equation:
        // a * t^2 + 2.f * b * t + c = 0
        // a = d.x^2 + d.y^2
        // b = d.x * o.x + d.y * o.y
        // c = o.x * o.x + o.y * o.y - r^2
        
        // Accurate solution, Chapter 7, Ray Tracing Gems I:

        float a = d.length2();

        // This ray is parallel to the Z axis and starts outside the cylinder
        // Special exit necessary to avoid div by 0
        if (a == 0.f) {
            return FLT_MAX;
        }

        float r2 = math::sqr(r);

        float c = o.length2() - r2;
        if (c <= 0.f) {
            return 0.f;
        }

        // f = o because capsule is at origin
        float b_prime = -dot(o, d);
        if (b_prime < 0.f) {
            return FLT_MAX;
        }

        float l2 = (o + (b_prime / a) * d).length2();
        float delta = r2 - l2;

        if (delta < 0.f ) {
            return FLT_MAX;
        }

        float q = b_prime + sqrtf(a * delta);

        return c / q;
    };

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
