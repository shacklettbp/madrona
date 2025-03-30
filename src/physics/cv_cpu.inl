#include "cv.hpp"
#include "cv_cpu.hpp"
#include "cv_cpu_utils.hpp"
#ifndef MADRONA_GPU_MODE

namespace madrona::phys::cv {
namespace cpu_solver {

inline void computeAccRef(float *a_ref_res,
                          float *R_res,
                          float *v,
                          float *J,
                          uint32_t num_rows_j,
                          uint32_t num_cols_j,
                          float *r,
                          float *diag_approx,
                          float h)
{
    float time_const = 2.f * h;
    float damp_ratio = 1.f;
    float d_min = 0.9f,
          d_max = 0.95f,
          width = 0.001f,
          mid = 0.5f,
          power = 2.f;

    float k = 1.f / (d_max * d_max * time_const * time_const
        * damp_ratio * damp_ratio);
    float b = 2.f / (d_max * time_const);

    // a_ref first gets -b * (J @ v)
    cpu_utils::matVecMul<false, true>(a_ref_res, J, v, num_rows_j, num_cols_j);
    cpu_utils::vecScale(a_ref_res, -b, num_rows_j);

    // Then compute -k * imp * r
    for (uint32_t i = 0; i < num_rows_j; i++) {
        float imp_x = fabs(r[i]) / width;
        float imp_a = (1.f / powf(mid, power-1.f)) * powf(imp_x, power);
        float imp_b = 1.f - (1.f / powf(1.f - mid, power - 1)) *
                      powf(1.f - imp_x, power);

        float imp_y = (imp_x < mid) ? imp_a : imp_b;
        float imp = d_min + imp_y * (d_max - d_min);
        if (imp < d_min)
            imp = d_min;
        else if (imp > d_max)
            imp = d_max;
        imp = (imp_x > 1.f) ? d_max : imp;

        a_ref_res[i] -= k * imp * r[i];
        R_res[i] = ((1 - imp) / imp) * diag_approx[i];
    }
}

inline void adjustContactRegularization(float *R,
                                        float *mus,
                                        uint32_t dim)
{
    uint32_t con_dim = 3;
    constexpr float imp_ratio = 1.f;
    for (uint32_t i = 0; i < dim; i += con_dim) {
        // Impedance of tangential components: R[1] = R[0] / imp_ratio
        R[i + 1] = R[i] / imp_ratio;
        // Regularized cone mu is mu[1] * sqrt(R[1] / R[0])
        mus[i] = mus[i + 1] * sqrtf(R[i+1] / R[i]);
        for (uint32_t j = 2; j < con_dim; j++) {
            R[i + j] = R[i + 1] * mus[i + 1] * mus[i + 1] / (
                    mus[i + j] * mus[i + j]);
        }
    }
}

inline void nonlinearCG(Context &ctx,
                        float *res,
                        float *a_ref_c,
                        float *a_ref_e,
                        float *D_c,
                        float *D_e,
                        float tol,
                        float ls_tol,
                        uint32_t max_iter,
                        uint32_t ls_iters,
                        CVSolveData &cv_sing)
{
    uint32_t nv = cv_sing.totalNumDofs;
    float scale = 1.f / cv_sing.totalMass;

    float x[nv];
    float g[nv];
    float M_grad[nv];
    float M_grad_new[nv];
    float p[nv];
    uint32_t nv_bytes = nv * sizeof(float);

    // x0 = a_free
    memcpy(x, cv_sing.freeAcc, nv_bytes);
    // f(x0) and df(x0), M^{-1}df(x0)
    float fun = obj(g, x, D_c, D_e, a_ref_c, a_ref_e, ctx, cv_sing);
    memcpy(M_grad, g, nv_bytes);
    fullMSolveMul(ctx, M_grad, true);
    // p = -M_grad
    memcpy(p, M_grad, nv_bytes);
    cpu_utils::vecScale(p, -1.f, nv);
    uint32_t i = 0;
    for (; i < max_iter; i++) {
        // Convergence check
        if (scale * cpu_utils::norm(g, nv) < tol) break;

        // Exact line search
        float alpha = exactLineSearch(x, p, D_c, D_e, a_ref_c, a_ref_e,
                                      cv_sing.numRowsJc, cv_sing.numRowsJe,
                                      cv_sing.totalNumDofs, ls_tol, ls_iters,
                                      ctx, cv_sing);
        if (alpha == 0.f) break;

        // Update x
        cpu_utils::sclAdd(x, p, alpha, nv);
        // Temporary: dot(g, M_grad)
        float den = fmaxf(cpu_utils::dot(g, M_grad, nv), 1e-12f);

        // Check improvement
        float fun_new = obj(g, x, D_c, D_e, a_ref_c, a_ref_e, ctx, cv_sing);
        if (scale * (fun - fun_new) < tol) break;

        // Polak-Ribiere (Mgrad holds Mgrad_new - M_grad)
        memcpy(M_grad_new, g, nv_bytes);
        fullMSolveMul(ctx, M_grad_new, true);

        cpu_utils::sclAdd(M_grad, M_grad_new, -1, nv);
        cpu_utils::vecScale(M_grad, -1, nv);
        float beta = cpu_utils::dot(g, M_grad, nv) / den;
        beta = fmaxf(0.f, beta);

        // Update p_new = beta * p - Mgrad_new
        cpu_utils::vecScale(p, beta, nv);
        cpu_utils::sclAdd(p, M_grad_new, -1, nv);
        fun = fun_new;
        memcpy(M_grad, M_grad_new, nv_bytes);

        cpu_utils::vecScale(p, beta, nv);
    }
    printf("CG Iterations %d\n", i);
    memcpy(res, x, nv_bytes);
}

inline void fullMSolveMul(Context &ctx,
                          float *x,
                          bool solve)
{
    uint32_t world_id = ctx.worldID().idx;
    StateManager *state_mgr = getStateManager(ctx);
    BodyGroupMemory *all_mems = state_mgr->getWorldComponents<
        BodyGroupArchetype, BodyGroupMemory>(world_id);
    BodyGroupProperties *all_prop = state_mgr->getWorldComponents<
        BodyGroupArchetype, BodyGroupProperties>(world_id);
    CountT num_grps = state_mgr->numRows<BodyGroupArchetype>(world_id);

    uint32_t processed_dofs = 0;
    for (uint32_t i = 0; i < num_grps; i++) {
        BodyGroupMemory m = all_mems[i];
        BodyGroupProperties p = all_prop[i];
        float *x_ptr = x + processed_dofs;
        if (solve) {
            tasks::solveM(p, m, x_ptr);
        } else {
            float y[p.qvDim];
            tasks::mulM(p, m, x_ptr, y);
            memcpy(x_ptr, y, p.qvDim * sizeof(float));
        }
        processed_dofs += p.qvDim;
    }
}

inline float obj(float *grad_out,
                 float *x,
                 float *D_c,
                 float *D_e,
                 float *a_ref_c,
                 float *a_ref_e,
                 Context &ctx, CVSolveData &cv_sing)
{
    using namespace cpu_utils;
    uint32_t nv = cv_sing.totalNumDofs;
    uint32_t nv_bytes = nv * sizeof(float);

    float cost = 0.f;
    memset(grad_out, 0, nv_bytes);

    // Gauss: f(x) = 0.5 (x - a_free)^T M (x - a_free), grad = M (x - a_free)
    float x_min_a_free[nv];
    float Mx_min_a_free[nv];
    memcpy(x_min_a_free, x, nv_bytes);
    sclAdd(x_min_a_free, cv_sing.freeAcc, -1.f, nv);
    memcpy(Mx_min_a_free, x_min_a_free, nv_bytes);
    fullMSolveMul(ctx, Mx_min_a_free, false);

    cost += 0.5f * dot(x_min_a_free, Mx_min_a_free, nv);
    sclAdd(grad_out, Mx_min_a_free, 1, nv);

    // Contact constraints
    uint32_t nc = cv_sing.numRowsJc;
    float jar_c[nc]; // Jx - a_ref
    float grad_c[nc];
    matVecMul<false, true>(jar_c, cv_sing.J_c, x, cv_sing.numRowsJc, cv_sing.numColsJc);
    sclAdd(jar_c, a_ref_c, -1.f, nc);
    cost += s_c(grad_c, jar_c, D_c, cv_sing.mu, nc);
    matVecMul<true, false>(grad_out, cv_sing.J_c, grad_c, cv_sing.numRowsJc, cv_sing.numColsJc);

    // Equation constraints
    uint32_t ne = cv_sing.numRowsJe;
    float jar_e[ne];
    float grad_e[ne];
    matVecMul<false, true>(jar_e, cv_sing.J_e, x, cv_sing.numRowsJe, cv_sing.numColsJe);
    sclAdd(jar_e, a_ref_e, -1.f, ne);
    cost += s_e(grad_e, jar_e, D_e, ne);
    matVecMul<true, false>(grad_out, cv_sing.J_e, grad_e, cv_sing.numRowsJe, cv_sing.numColsJe);

    return cost;
}

inline float s_c(float *grad_out, float *jar, float *D_c, float *mus,
                 uint32_t dim)
{
    float cost = 0.f;
    memset(grad_out, 0, dim * sizeof(float));
    // Loop through each contact
    for (uint32_t i = 0; i < dim / 3; i++) {
        // Fetch data
        float jar_N = jar[3 * i];
        float jar_T1 = jar[3 * i + 1];
        float jar_T2 = jar[3 * i + 2];
        float mu = mus[3 * i];
        float mu1 = mus[3 * i + 1];
        float mu2 = mus[3 * i + 2];
        float Dn = D_c[3 * i];
        float D1 = D_c[3 * i + 1];
        float D2 = D_c[3 * i + 2];
        // Convert to cone space
        float N = jar_N * mu;
        float T1 = jar_T1 * mu1;
        float T2 = jar_T2 * mu2;
        float T = sqrtf(T1 * T1 + T2 * T2);
        float mid_weight = 1.f / (mu * mu * (1 + mu * mu));

        // Top zone
        if (N >= mu * T || (T <= 0 && N >= 0)) {
            continue;
        }
        // Bottom zone
        else if (mu * N + T <= 0 || (T <= 0 && N < 0)) {
            cost += 0.5f * (Dn * jar_N * jar_N +
                            D1 * jar_T1 * jar_T1 +
                            D2 * jar_T2 * jar_T2);
            grad_out[3 * i] = Dn * jar_N;
            grad_out[3 * i + 1] = D1 * jar_T1;
            grad_out[3 * i + 2] = D2 * jar_T2;
        }
        // Middle zone
        else {
            cost += 0.5f * Dn * mid_weight * (N - mu * T) * (N - mu * T);
            float tmp = Dn * mid_weight * (N - mu * T) * mu;
            grad_out[3 * i] = tmp;
            grad_out[3 * i + 1] = -(tmp / T) * T1 * mus[3 * i + 1];
            grad_out[3 * i + 2] = -(tmp / T) * T2 * mus[3 * i + 2];
        }
    }
    return cost;
}

float s_e(float *grad_out,
          float *jar,
          float *D_e,
          uint32_t dim) {
    float cost = 0.f;
    memset(grad_out, 0, dim * sizeof(float));
    for (uint32_t i = 0; i < dim; i++) {
        cost += 0.5f * D_e[i] * jar[i] * jar[i];
        grad_out[i] = D_e[i] * jar[i];
    }
    return cost;
}

float exactLineSearch(float *xk, float *pk, float *D_c, float *D_e,
                      float *a_ref_c, float *a_ref_e, uint32_t nc, uint32_t ne,
                      uint32_t nv, float ls_tol, uint32_t ls_iters,
                      Context &ctx, CVSolveData &cv_sing) {
    using namespace cpu_utils;

    struct Evals {
        float fun;
        float grad;
        float hess;
    };

    struct ContactStore {
        float quad0;
        float quad1;
        float quad2;
        float U0;
        float V0;
        float UU;
        float UV;
        float VV;
        float Dm;
    };

    // Search vector too small
    if (norm(pk, nv) < 1e-15f) return 0.f;

    uint32_t nv_bytes = nv * sizeof(float);

    // Precompute some values
    // 1. Gauss objective
    float x_min_a_free[nv];
    float Mx_min_a_free[nv];
    float Mpk[nv];
    // x - a_free, M(x - a_free)
    memcpy(x_min_a_free, xk, nv_bytes);
    sclAdd(x_min_a_free, cv_sing.freeAcc, -1.f, nv);
    memcpy(Mx_min_a_free, x_min_a_free, nv_bytes);
    fullMSolveMul(ctx, Mx_min_a_free, false);
    // M @ pk
    memcpy(Mpk, pk, nv_bytes);
    fullMSolveMul(ctx, Mpk, false);

    float x_min_M_x_min = dot(x_min_a_free, Mx_min_a_free, nv);
    float pMp = dot(pk, Mpk, nv);
    float pMx_free = dot(pk, Mx_min_a_free, nv);

    // 2. Cone constraints
    float Jx_aref_c[nc];
    float Jp_c[nc];
    matVecMul<false, true>(Jx_aref_c, cv_sing.J_c, xk, cv_sing.numRowsJc, cv_sing.numColsJc);
    matVecMul<false, true>(Jp_c, cv_sing.J_c, pk, cv_sing.numRowsJc, cv_sing.numColsJc);
    sclAdd(Jx_aref_c, a_ref_c, -1.f, nc);

    // For each contact, store some information to avoid re-computation
    ContactStore cs[nc / 3];
    for (uint32_t i = 0; i < nc / 3; i++) {
        // Components of J @ x - a_ref, J @ p
        float Jx_N = Jx_aref_c[3 * i];
        float Jx_T1 = Jx_aref_c[3 * i + 1];
        float Jx_T2 = Jx_aref_c[3 * i + 2];
        float Jp_N = Jp_c[3 * i];
        float Jp_T1 = Jp_c[3 * i + 1];
        float Jp_T2 = Jp_c[3 * i + 2];
        // Friction. Map to dual cone space
        float mu = cv_sing.mu[3 * i];
        float mu1 = cv_sing.mu[3 * i + 1];
        float mu2 = cv_sing.mu[3 * i + 2];
        // Weights
        float Dn = D_c[3 * i];
        float D1 = D_c[3 * i + 1];
        float D2 = D_c[3 * i + 2];
        float mid_weight = Dn / (mu * mu * (1 + mu * mu));

        cs[i].quad0 = 0.5f * (Dn * Jx_N * Jx_N +
                              D1 * Jx_T1 * Jx_T1 +
                              D2 * Jx_T2 * Jx_T2);
        cs[i].quad1 = (Dn * Jx_N * Jp_N +
                       D1 * Jx_T1 * Jp_T1 +
                       D2 * Jx_T2 * Jp_T2);
        cs[i].quad2 = 0.5f * (Dn * Jp_N * Jp_N +
                              D1 * Jp_T1 * Jp_T1 +
                              D2 * Jp_T2 * Jp_T2);
        // Map to dual cone space
        Jx_N *= mu;
        Jx_T1 *= mu1;
        Jx_T2 *= mu2;
        Jp_N *= mu;
        Jp_T1 *= mu1;
        Jp_T2 *= mu2;

        cs[i].U0 = Jx_N;
        cs[i].V0 = Jp_N;
        cs[i].UU = Jx_T1 * Jx_T1 + Jx_T2 * Jx_T2;
        cs[i].UV = Jx_T1 * Jp_T1 + Jx_T2 * Jp_T2;
        cs[i].VV = Jp_T1 * Jp_T1 + Jp_T2 * Jp_T2;
        cs[i].Dm = mid_weight;
    }

    // 3. Equality constraints
    float Jx_aref_e[ne];
    float Jp_e[ne];
    matVecMul<false, true>(Jx_aref_e, cv_sing.J_e, xk, cv_sing.numRowsJe, cv_sing.numColsJe);
    matVecMul<false, true>(Jp_e, cv_sing.J_e, pk, cv_sing.numRowsJe, cv_sing.numColsJe);
    sclAdd(Jx_aref_e, a_ref_e, -1.f, ne);

    auto phi = [&](float a) {
        // Process Gauss first
        float fun = 0.5f * a * a * pMp + a * pMx_free + 0.5f * x_min_M_x_min;
        float grad = a * pMp + pMx_free;
        float hess = pMp;

        // Then process cones
        for (uint32_t i = 0; i < nc / 3; i++) {
            ContactStore c = cs[i];
            float mu = cv_sing.mu[3 * i];
            float N = c.U0 + a * c.V0;
            float T_sqr = c.UU + a * (2 * c.UV + a * c.VV);
            float T = sqrtf(T_sqr);

            // Top zone
            if (N >= mu * T || (T <= 0 && N >= 0)) {
                continue;
            }
            // Bottom zone
            else if (mu * N + T <= 0 || (T <= 0 && N < 0)) {
                fun += a * a * c.quad2 + a * c.quad1 + c.quad0;
                grad += 2 * a * c.quad2 + c.quad1;
                hess += 2 * c.quad2;
            }
            // Middle zone
            else {
                float N1 = c.V0;
                float T1 = (c.UV + a * c.VV) / T;
                float T2 = c.VV / T - (c.UV + a * c.VV) * T1 / (T*T);
                fun += 0.5f*c.Dm*(N-mu*T)*(N-mu*T);
                grad += c.Dm*(N-mu*T)*(N1-mu*T1);
                hess += c.Dm*((N1-mu*T1)*(N1-mu*T1) + (N-mu*T)*(-mu*T2));
            }
        }

        // Finally process equality constraints
        for (uint32_t i = 0; i < ne; i++) {
            float orig = Jx_aref_e[i];
            float dj = Jp_e[i];
            float de = D_e[i];
            float N_search = orig + a * dj;
            fun += 0.5f * de * N_search * N_search;
            grad += de * (orig * dj + a * dj * dj);
            hess += de * dj * dj;
        }
        return Evals{fun, grad, hess};
    };

    float alpha = 0.f;
    Evals evals = phi(alpha);

    float alpha1 = alpha - evals.grad / evals.hess; // Newton step
    Evals evals_1 = phi(alpha1);
    if (evals.fun < evals_1.fun) {
        alpha1 = alpha;
    }

    evals = phi(alpha1);
    // Initial convergence
    if (fabsf(evals.grad) < ls_tol) {
        return alpha1;
    }

    // Opposing direction of gradient at alpha1
    float a_dir = (evals.grad < 0.f) ? 1.f : -1.f;
    uint32_t i = 0;
    for (; i < ls_iters; i++) {
        evals = phi(alpha1);
        // gradient moves in the opposite direction as alpha1, start bracketing
        if (evals.grad * a_dir > -1.f * ls_tol) { break; }
        // Converged
        if (fabsf(evals.grad) < ls_tol) { return alpha1; }

        // Newton step
        alpha1 -= evals.grad / evals.hess;
    }
    if (i == ls_iters) {
        // Failed to converge
        return alpha1;
    }

    // Bracketing to find where d_phi equals zero
    float alpha_low = alpha1;
    float alpha_high = alpha1 - evals.grad / evals.hess;
    evals = phi(alpha_low);
    if (evals.grad > 0.f) {
        float tmp = alpha_low;
        alpha_low = alpha_high;
        alpha_high = tmp;
    }

    uint32_t ib = 0;
    float alpha_mid = alpha_low;
    for (; ib < ls_iters; ib++) {
        alpha_mid = 0.5f * (alpha_low + alpha_high);
        evals = phi(alpha_mid);
        if (fabsf(evals.grad) < ls_tol) {
            return alpha_mid;
        }
        // Narrow the bracket
        if (evals.grad > 0.f) {
            alpha_high = alpha_mid;
        } else {
            alpha_low = alpha_mid;
        }

        // Bracketing is small
        if (fabsf(alpha_high - alpha_low) < ls_tol) {
            return alpha_mid;
        }
    }
    // Failed to converge
    return alpha_mid;
}

}
}
#endif
