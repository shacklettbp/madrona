#pragma once

#include <madrona/math.hpp>

#include <array>
#include <cassert>
#include <cfloat>

#ifdef MADRONA_GJK_DEBUG
#include <cstdio>
#endif

namespace madrona::geo {

/*
GJK Implementation. This is intended to be a private implementation file for
geo.cpp, but factored out into a header so the functions can be individually
unit tested.

Original implementation followed the description in:
Collision Detection in Interactive 3D Environments, Gino van den Bergen
Section 4.3 (Mostly 4.3.5 - 4.3.6)
We are no longer using the exact algorithm outlined here, but the general notes
about GJK are still very relevant:
- W = simplex (W_k = simplex at iteration k)
- v(conv(W_k)) = point on surface of simplex closest to origin
- w_k = new support point in direction v_k on minkowski difference 
  of hulls A and B (A - B) 
- v_{k+1} = v(conv(W_k U {w_k})) = point closest to origin on surface of new simplex
  formed by union of W_k and {w_k}
- W_{k+1} = smallest subset of (W_k U {w_k}) s.t. v_{k + 1} contained in conv(W_{k+1})
   - Another way to think about this is we reduce the simplex to the smallest
     set of points such that the closest point between the set and the origin is
     still v_{k+1}. For example v_{k+1} might be closest to the middle of a face
     on the simplex, in which case any vertices not contributing to that face
     should be deleted. Similarly if it's closest to an edge, only the two edge
     vertices should remain in the simplex
   - The text refers W_k U {w_k} as Y and to the minimal subset of this simplex (
     that becomes the new simplex) as X.

Two major questions in the above definition are how to compute
v(conv(W_k U {w_k})) efficiently and how to reduce (W_k U {w_k}) into the
mimimal subset X. In the textbook algorithm, these are done simultaneously by
Johnson's distance algorithm (4.3.3):
- The new support point v_{k+1} on the simplex can be represented as linear
  combination of the current points in Y with coefficients lambda_i. A key
  insight is that any point in Y with a lambda_i of 0 does not contribute to
  the minimal simplex needed to "support" point v_{k+1} ("support" just
  means that v(conv(X)) still equals v_{k+1})
    - The vanilla Johnson's algorithm simply does this by iterating over
      all possible combinations of X and solving for the parameters lambda_i
      that must result in the point v(aff(X)) = v(conv(X)).
      (aff(X) is the affine hull). 
- Johnson's algorithm can be significantly simplified by using a special casing
  approach based on barycentric coordinates and Voronoi regions.
  We can compute v_{k+1} (and determine which simplex points can be removed) in
  a case by case basis for each possible simplex size, leveraging the fact that
  GJK terminates when the minimal simplex has 4 points, because the origin
  must be included in the shape. This method of using Voronoi regions is laid
  out in Erin Catto's GDC 2010 presentation. We also leverage the fact that
  the most recently added support point (w_k) must contribute to v_{k+1} to
  prune many test cases (in the worst case of testing against the tetrahedron,
  only the features touching w_k have to be tested). If w_k didn't contribute,
  the algorithm can just terminate since v_{k+1} should be equal to v_k.
- Unfortunately, the barycentrics approach above is not particularly fast or
  numerically stable, especially when trying to cull test cases.
  A relatively recent paper shows a better, top-down search solution, that
  is much more stable but also faster in some cases due to the top down
  approach:
      Improving the GJK Algorithm for Faster and More Reliable Distance
      Queries Between Two Convex Objects, Montanari et al, ToG 2017.
*/

using namespace math;

template <typename T>
struct GJK {
    Vector3 v;

    CountT nY;
    Vector3 Y[4];

    template <typename Fn>
    inline float computeDistance2(
        Fn &&support_fn, Vector3 init_v, float err_tolerance);

private:
    template <CountT dst>
    inline void storeSupports(
        Vector3 w, Vector3 a_support, Vector3 b_support)
    {
        Y[dst] = w;
        static_cast<T *>(this)->template storeSrcSupports<dst>(
            a_support, b_support);
    }

    template <CountT dst, CountT src>
    void moveSupports(float lambda)
    {
        Y[dst] = Y[src];
        static_cast<T *>(this)->template moveSrcSupports<dst, src>(lambda);
    }
};

struct GJKWithoutPoints : GJK<GJKWithoutPoints> {
private:
    template <CountT dst>
    inline void storeSrcSupports(Vector3, Vector3) {}
    template <CountT dst, CountT src>
    inline void moveSrcSupports(float) {}
friend struct GJK<GJKWithoutPoints>;
};

struct GJKWithPoints : GJK<GJKWithPoints> {
    float lambdas[4];
    Vector3 aPoints[4];
    Vector3 bPoints[4];

    inline void getClosestPoints(Vector3 *a_out, Vector3 *b_out)
    {
        switch (nY) {
        case 1: {
            *a_out = aPoints[0];
            *b_out = bPoints[0];
        } break;
        case 2: {
            *a_out = lambdas[0] * aPoints[0] + lambdas[1] * aPoints[1];
            *b_out = lambdas[0] * bPoints[0] + lambdas[1] * bPoints[1];
        } break;
        case 3: {
            *a_out = lambdas[0] * aPoints[0] + 
                lambdas[1] * aPoints[1] + lambdas[2] * aPoints[2];
            *b_out = lambdas[0] * bPoints[0] +
                lambdas[1] * bPoints[1] + lambdas[2] * bPoints[2];
        } break;
        case 4: {
            // 4 simplex means we enclose origin,
            // we don't know the closest point. Leave this case here so
            // user doesn't trigger undefined behavior calling getClosestPoints
        } break;
        default: MADRONA_UNREACHABLE();
        }
    }

private:
    template <CountT dst>
    inline void storeSrcSupports(Vector3 a_support, Vector3 b_support)
    {
        aPoints[dst] = a_support;
        bPoints[dst] = b_support;
    }

    template <CountT dst, CountT src>
    inline void moveSrcSupports(float lambda)
    {
        lambdas[dst] = lambda;
        aPoints[dst] = aPoints[src];
        bPoints[dst] = bPoints[src];
    }

friend struct GJK<GJKWithPoints>;
};

struct GJKSimplexSolveState {
    Vector3 v;
    float vLen2;
    std::array<float, 4> lambdas;
};

MADRONA_ALWAYS_INLINE inline
bool gjkCompareSigns(float a, float b)
{
    return (a > 0 && b > 0) || (a < 0 && b < 0);
}

MADRONA_ALWAYS_INLINE inline
GJKSimplexSolveState gjkSolve1Simplex(Vector3 Y0)
{
    return {
        .v = Y0,
        .vLen2 = Y0.length2(),
        .lambdas = { 1, 0, 0, 0 },
    };
}

MADRONA_ALWAYS_INLINE inline
GJKSimplexSolveState gjkSolve2Simplex(
    Vector3 Y0, Vector3 Y1)
{
    // S1D from paper. 
    Vector3 s1 = Y1;
    Vector3 s2 = Y0;

    Vector3 t = s2 - s1;
    float t_len2 = t.length2();

    float mu_max;
    float s1_I, s2_I;
    {
        mu_max = s1.x - s2.x;
        s1_I = s1.x;
        s2_I = s2.x;

        float mu_y = s1.y - s2.y;
        if (fabsf(mu_y) > fabsf(mu_max)) {
            mu_max = mu_y;
            s1_I = s1.y;
            s2_I = s2.y;
        }

        float mu_z = s1.z - s2.z;
        if (fabsf(mu_z) > fabsf(mu_max)) {
            mu_max = mu_z;
            s1_I = s1.z;
            s2_I = s2.z;
        }
    }

    // There seems to be a slight bug in the paper here:
    // Paper algorithm 3 says pO should be
    // Vector3 pO = (dot(s2, t) / t_len2) * t + s2;
    // This doesn't really make sense because t goes from s1 to s2, but
    // then we're adding s2?
    // Flipping this around seems to work. Also slight optimization to
    // only compute the needed coordinate of pO
    float pO_I = (dot(s2, t) / t_len2) * (s1_I - s2_I) + s2_I;

    float C1 = pO_I - s2_I;
    float C2 = s1_I - pO_I;

#ifdef MADRONA_GJK_DEBUG
    printf("2 simplex:\n");
    printf("  (%f %f %f)\n  (%f %f %f)\n",
        s1.x, s1.y, s1.z, s2.x, s2.y, s2.z);
    printf("%f %f %f\n", mu_max, C1, C2);
#endif

    if (gjkCompareSigns(mu_max, C1) && gjkCompareSigns(mu_max, C2)) {
        float lambda2 = C2 / mu_max;
        Vector3 v = s1 + t * lambda2;

        float lambda1 = 1.f - lambda2;
        return {
            .v = v,
            .vLen2 = v.length2(),
            .lambdas = { lambda2, lambda1, 0.f, 0.f },
        };
    } else {
        return {
            .v = s1,
            .vLen2 = s1.length2(),
            .lambdas = { 0.f, 1.f, 0.f, 0.f },
        };
    }
}


MADRONA_ALWAYS_INLINE inline
GJKSimplexSolveState gjkSolve3Simplex(
    Vector3 Y0, Vector3 Y1, Vector3 Y2)
{
    // S2D from paper

    Vector3 s1 = Y2;
    Vector3 s2 = Y1;
    Vector3 s3 = Y0;

#ifdef MADRONA_GJK_DEBUG
    printf("3 simplex\n");
    printf("  (%f %f %f)\n  (%f %f %f)\n  (%f %f %f)\n",
        s1.x, s1.y, s1.z, s2.x, s2.y, s2.z, s3.x, s3.y, s3.z);
#endif

    Vector3 n = cross(s2 - s1, s3 - s1);
    float n_len2 = n.length2();
    Vector3 pO = dot(s1, n) * n / n_len2;

    float mu_max;
    Vector2 s1_2D, s2_2D, s3_2D, pO_2D;
    {
        // 3x3 determinant, eliminate first row and last column
        // Bottow row is 1s (eq 23).
        float M_14 = s2.y * s3.z - s3.y * s2.z
                   - s1.y * s3.z + s3.y * s1.z
                   + s1.y * s2.z - s2.y * s1.z;

        // Eliminate second row and last column
        float M_24 = s2.x * s3.z - s3.x * s2.z
                   - s1.x * s3.z + s3.x * s1.z
                   + s1.x * s2.z - s2.x * s1.z;

        // Eliminate third row and last column
        float M_34 = s2.x * s3.y - s3.x * s2.y
                   - s1.x * s3.y + s3.x * s1.y
                   + s1.x * s2.y - s2.x * s1.y;

#ifdef MADRONA_GJK_DEBUG
        printf("%f %f %f\n", M_14, M_24, M_34);
#endif

        float M_14_abs = fabsf(M_14);
        float M_24_abs = fabsf(M_24);
        float M_34_abs = fabsf(M_34);

        if (M_14_abs >= M_24_abs && M_14_abs >= M_34_abs) {
            mu_max = M_14;
            s1_2D = { s1.y, s1.z };
            s2_2D = { s2.y, s2.z };
            s3_2D = { s3.y, s3.z };
            pO_2D = { pO.y, pO.z };
        } else if (M_24_abs >= M_34_abs) {
            mu_max = M_24;
            s1_2D = { s1.x, s1.z };
            s2_2D = { s2.x, s2.z };
            s3_2D = { s3.x, s3.z };
            pO_2D = { pO.x, pO.z };
        } else {
            mu_max = M_34;
            s1_2D = { s1.x, s1.y };
            s2_2D = { s2.x, s2.y };
            s3_2D = { s3.x, s3.y };
            pO_2D = { pO.x, pO.y };
        }
    }

    float C1 = pO_2D.x * s2_2D.y + pO_2D.y * s3_2D.x + s2_2D.x * s3_2D.y 
             - pO_2D.x * s3_2D.y - pO_2D.y * s2_2D.x - s3_2D.x * s2_2D.y;

    float C2 = pO_2D.x * s3_2D.y + pO_2D.y * s1_2D.x + s3_2D.x * s1_2D.y 
             - pO_2D.x * s1_2D.y - pO_2D.y * s3_2D.x - s1_2D.x * s3_2D.y;

    float C3 = pO_2D.x * s1_2D.y + pO_2D.y * s2_2D.x + s1_2D.x * s2_2D.y
             - pO_2D.x * s2_2D.y - pO_2D.y * s1_2D.x - s2_2D.x * s1_2D.y;

    bool cmp_signs[] = {
        gjkCompareSigns(mu_max, C1),
        gjkCompareSigns(mu_max, C2),
        gjkCompareSigns(mu_max, C3),
    };

#ifdef MADRONA_GJK_DEBUG
    printf("(%f %f) (%f %f) (%f %f)\n",
        s1_2D.x, s1_2D.y, s2_2D.x, s2_2D.y, s3_2D.x, s3_2D.y);
    printf("%f (%f %f %f)\n", mu_max, C1, C2, C3);

    printf("%d %d %d\n",
        (int)cmp_signs[0], (int)cmp_signs[1], (int)cmp_signs[2]);
#endif


    if (cmp_signs[0] && cmp_signs[1] && cmp_signs[2]) {
        float lambda2 = C2 / mu_max;
        float lambda3 = C3 / mu_max;
        float lambda1 = 1.f - lambda2 - lambda3;

        Vector3 v = s1 * lambda1 + s2 * lambda2 + s3 * lambda3;
        return {
            .v = v,
            .vLen2 = v.length2(),
            .lambdas = { lambda3, lambda2, lambda1, 0.f },
        };
    }

    // cmp_signs negation here doesn't follow paper for same reason as 
    // gjkSolve4Simplex

    GJKSimplexSolveState res;
    res.vLen2 = FLT_MAX;
    if (!cmp_signs[1]) {
        res = gjkSolve2Simplex(Y0, Y2);
        res.lambdas = { res.lambdas[0], 0.f, res.lambdas[1], 0.f };
    }

    if (!cmp_signs[2]) {
        GJKSimplexSolveState sub = gjkSolve2Simplex(Y1, Y2);
        if (sub.vLen2 < res.vLen2) {
            res = sub;
            res.lambdas = { 0.f, res.lambdas[0], res.lambdas[1], 0.f };
        }
    }

    // Not in paper
    if (!cmp_signs[0]) {
        GJKSimplexSolveState sub = gjkSolve2Simplex(Y0, Y1);
        if (sub.vLen2 < res.vLen2) {
            res = sub;
            res.lambdas = { res.lambdas[0], res.lambdas[1], 0.f, 0.f };
        }
    }

    assert(res.vLen2 != FLT_MAX);
    return res;
}

MADRONA_ALWAYS_INLINE inline
GJKSimplexSolveState gjkSolve4Simplex(
    Vector3 Y0, Vector3 Y1, Vector3 Y2, Vector3 Y3)
{
    // S3D in paper
    // s1, s2, s3, s4 from paper are labelled in reverse order
    Vector3 s1 = Y3;
    Vector3 s2 = Y2;
    Vector3 s3 = Y1;
    Vector3 s4 = Y0;

#ifdef MADRONA_GJK_DEBUG
    printf("4 simplex\n");
    printf("  (%.9g %.9g %.9g)\n  (%.9g %.9g %.9g)\n  (%.9g %.9g %.9g)\n  (%.9g %.9g %.9g)\n",
        s1.x, s1.y, s1.z, s2.x, s2.y, s2.z, s3.x, s3.y, s3.z,
        s4.x, s4.y, s4.z);
#endif

    /* M = [
        [ s1.x, s2.x, s3.x, s4.x ],
        [ s1.y, s2.y, s3.y, s4.y ],
        [ s1.z, s2.z, s3.z, s4.z ],
        [    1,    1,    1,    1 ],
    ] */

    // Going to compute the determinant as the sum of cofactors eliminating
    // the bottom row (3x3 determinants). Remember that bottom row cofactor
    // signs are -1, 1, -1, 1
    auto det = [](Vector3 a, Vector3 b, Vector3 c) {
        return dot(a, cross(b, c));
    };

    float C_41 = -det(s2, s3, s4);
    float C_42 = det(s1, s3, s4);
    float C_43 = -det(s1, s2, s4);
    float C_44 = det(s1, s2, s3);

    float det_M = C_41 + C_42 + C_43 + C_44;

    bool cmp_signs[4] = {
        gjkCompareSigns(det_M, C_41),
        gjkCompareSigns(det_M, C_42), 
        gjkCompareSigns(det_M, C_43),
        gjkCompareSigns(det_M, C_44),
    };

#ifdef MADRONA_GJK_DEBUG
    printf("%d %d %d %d\n", (int)cmp_signs[0], (int)cmp_signs[1], 
        (int)cmp_signs[2], (int)cmp_signs[3]);
    printf("%f, %f %f %f %f\n", det_M, C_41, C_42, C_43, C_44);
#endif

    if (cmp_signs[0] && cmp_signs[1] && cmp_signs[2] && cmp_signs[3]) {
        float lambda1 = C_41 / det_M;
        float lambda2 = C_42 / det_M;
        float lambda3 = C_43 / det_M;
        float lambda4 = 1.f - lambda1 - lambda2 - lambda3;

        Vector3 v = s1 * lambda1 + s2 * lambda2 + s3 * lambda3 + s4 * lambda4;

        return {
            .v = v,
            .vLen2 = v.length2(),
            .lambdas = { lambda4, lambda3, lambda2, lambda1 },
        };
    }

    // Deviation from paper in conditions to check the faces:
    // if det_M is 0, CompareSigns calls will be false! This would cause the 
    // function to test nothing, return FLT_MAX and terminate (if we
    // ignore several asserts along the way). Not 100% sure but this seems
    // risky because this case is quite common: Consider a cube tested against
    // the origin. The two triangles that make up the cube face will make a
    // degenerate tetrahedron that will trigger this issue. In the cube case
    // this doesn't really matter since continuing to iterate after the first
    // triangle won't get any closer but is this true in general?
    // For now, we're going to just use the negation of the positive 
    // CompareSigns check above, which will cause the 0 case for det_M or C_4J
    // to check the face.
    GJKSimplexSolveState res;
    res.vLen2 = FLT_MAX;
    if (!cmp_signs[1]) {
        // s4, s3, s1
        res = gjkSolve3Simplex(Y0, Y1, Y3);
        res.lambdas = { res.lambdas[0], res.lambdas[1], 0.f, res.lambdas[2] };
    }

    if (!cmp_signs[2]) {
        // s4, s2, s1
        GJKSimplexSolveState sub = gjkSolve3Simplex(Y0, Y2, Y3);

        if (sub.vLen2 < res.vLen2) {
            res = sub;
            res.lambdas = { res.lambdas[0], 0.f, res.lambdas[1], res.lambdas[2] };
        }
    }

    if (!cmp_signs[3]) {
        // s3, s2, s1
        GJKSimplexSolveState sub = gjkSolve3Simplex(Y1, Y2, Y3);

        if (sub.vLen2 < res.vLen2) {
            res = sub;
            res.lambdas = { 0.f, res.lambdas[0], res.lambdas[1], res.lambdas[2] };
        }
    }

    // This final check isn't included in the paper 
    if (!cmp_signs[0]) {
        // s4, s3, s2
        GJKSimplexSolveState sub = gjkSolve3Simplex(Y0, Y1, Y2);

        if (sub.vLen2 < res.vLen2) {
            res = sub;
            res.lambdas = { res.lambdas[0], res.lambdas[1], res.lambdas[2], 0.f };
        }
    }

    assert(res.vLen2 != FLT_MAX);
    return res;
}

template <typename T>
template <typename Fn>
inline float GJK<T>::computeDistance2(
    Fn &&support_fn, Vector3 init_v, float err_tolerance2)
{
    v = init_v;
    nY = 0;

    float v_len2;
    float prev_v_len2 = FLT_MAX;

#ifdef MADRONA_GJK_DEBUG
    int _debug_num_gjk_iters = 0;
#endif

    while (true) {
#ifdef MADRONA_GJK_DEBUG
        printf("\nGJK Iter %d\n", _debug_num_gjk_iters++);
        printf("v (%f %f %f)\n", v.x, v.y, v.z);
#endif

        // Compute w_k 
        Vector3 a_support, b_support;
        Vector3 w = support_fn(v, &a_support, &b_support);

#ifdef MADRONA_GJK_DEBUG
        printf("w (%f %f %f)\n", w.x, w.y, w.z);
#endif

        GJKSimplexSolveState solve_state;
        switch (nY) {
        case 0: {
            // Note that this case only happens on the first iteration
            // The alternative would require inserting a second call
            // to support_fn before the loop and initializing with a
            // simplex of size 1
            storeSupports<0>(w, a_support, b_support);
            solve_state = gjkSolve1Simplex(Y[0]);
        } break;
        case 1: {
            storeSupports<1>(w, a_support, b_support);
            solve_state = gjkSolve2Simplex(Y[0], Y[1]);
        } break;
        case 2: {
            storeSupports<2>(w, a_support, b_support);
            solve_state = gjkSolve3Simplex(Y[0], Y[1], Y[2]);
        } break;
        case 3: {
            storeSupports<3>(w, a_support, b_support);
            solve_state = gjkSolve4Simplex(Y[0], Y[1], Y[2], Y[3]);
        } break;
        default: MADRONA_UNREACHABLE();
        }

#ifdef MADRONA_GJK_DEBUG
        printf("Post solve\n");
        printf("(%f %f %f) %f\n",
            solve_state.v.x, solve_state.v.y, solve_state.v.z,
            solve_state.vLen2);
        printf("(%f, %f, %f, %f)\n",
            solve_state.lambdas[0], solve_state.lambdas[1],
            solve_state.lambdas[2], solve_state.lambdas[3]);

        printf("%.9g => %.9g\n", prev_v_len2, solve_state.vLen2);
#endif

        // The below assert should be true in principle, but in reality
        // gjkSolve4Simplex for example can delegate to gjkSolve3Simplex
        // with a different order of points than the prior iteration,
        // resulting in extremely minor FP changes. This seems to happen
        // when a duplicate point is readded. We could check this with an if,
        // restore the old v (which we currently don't have saved) and 
        // exit immediately.
        //assert(solve_state.vLen2 <= prev_v_len2);
        //assert(solve_state.vLen2 - prev_v_len2 < 1e-10f); // nope
        assert(solve_state.vLen2 - prev_v_len2 < 1e-6f);

        // Compact the simplex to remove unnecessary points that don't support
        // solver_state.v
        {
            // This is written weirdly with a switch statement
            // to avoid dynamic indexing into the simplex state on the GPU.
            // We want the simplex state stored in registers.
            // Could simplify on CPU.

            nY = 0;

            auto addToSimplex = [this]<CountT src>(float lambda) {
                auto storeInSimplex = [this]<CountT dst>(
                    float lambda)
                {
                    (void)this;
                    moveSupports<dst, src>(lambda);
                };

                if (lambda == 0.f) {
                    return;
                }

                switch (nY) {
                case 0: {
                    storeInSimplex.template operator()<0>(lambda);
                } break;
                case 1: {
                    storeInSimplex.template operator()<1>(lambda);
                } break;
                case 2: {
                    storeInSimplex.template operator()<2>(lambda);
                } break;
                case 3: {
                    storeInSimplex.template operator()<3>(lambda);
                } break;
                default: MADRONA_UNREACHABLE();
                }

                nY += 1;
            };

            // Manually unroll
            addToSimplex.template operator()<0>(solve_state.lambdas[0]);
            addToSimplex.template operator()<1>(solve_state.lambdas[1]);
            addToSimplex.template operator()<2>(solve_state.lambdas[2]);
            addToSimplex.template operator()<3>(solve_state.lambdas[3]);
        }

#ifdef MADRONA_GJK_DEBUG
        printf("Post compact %u\n", (uint32_t)nY);
#endif

        // The origin must be inside the tetrahedron so distance is 0.
        if (nY == 4) {
            // Note we don't return solve_state.vLen2 in this case since
            // it can be very close to 0 but not 0 due to FP error.
            return 0.f;
        }

        // if v gets really small, the direction is pretty much meaningless.
        // Consider this as overlapping the origin.
        if (solve_state.vLen2 <= err_tolerance2) {
            return 0.f;
        }

        {
            float max_Y_len2 = Y[0].length2();
            MADRONA_UNROLL
            for (CountT i = 1; i < 4; i++) {
                float Y_len2 = Y[i].length2();
                if (i < nY && Y_len2 > max_Y_len2) {
                    max_Y_len2 = Y_len2;
                }
            }

            // v is too small relative to the size of the simplex. Same as
            // above case.
            if (solve_state.vLen2 <= FLT_EPSILON * max_Y_len2) {
                return 0.f;
            }
        }

        v_len2 = solve_state.vLen2;
        v = -solve_state.v;

        // Solution is not improving within margin of FP32 error.
        if (prev_v_len2 - v_len2 <= FLT_EPSILON * prev_v_len2) {
            break;
        }

        prev_v_len2 = v_len2;
    }

#ifdef MADRONA_GJK_DEBUG
    printf("Finished %f\n", v_len2);
#endif

    assert(v_len2 > 0.f);
    assert(nY > 0 && nY < 4);

    return v_len2;
}

}
