#include "cv.hpp"
#include "cv_cpu.hpp"
#include "cv_cpu_utils.hpp"
#ifndef MADRONA_GPU_MODE

namespace madrona::phys::cv {
namespace cpu_solver {

inline float getImpedance(const float* solimp, float pos) {
    float imp;
    // flat function
    if (solimp[0] == solimp[1] || solimp[2] <= MINVAL) {
        imp = 0.5f * (solimp[0] + solimp[1]);
        return imp;
    }

    // x = abs((pos-margin) / width)
    float x = pos / solimp[2];
    if (x < 0) {
        x = -x;
    }

    // fully saturated
    if (x >= 1 || x <= 0) {
        imp = x >= 1 ? solimp[1] : solimp[0];
        return imp;
    }

    // linear
    float y;
    if (solimp[4] == 1) {
        y = x;
    }

      // y(x) = a*x^p if x<=midpoint
    else if (x <= solimp[3]) {
        float a = 1/powf(solimp[3], solimp[4]-1);
        y = a*powf(x, solimp[4]);
    }

    // y(x) = 1-b*(1-x)^p is x>midpoint
    else {
        float b = 1/powf(1-solimp[3], solimp[4]-1);
        y = 1-b*powf(1-x, solimp[4]);
      }

    // scale
    imp = solimp[0] + y*(solimp[1]-solimp[0]);
    return imp;
}

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
    float solimp[5] = {d_min, d_max, width, mid, power};

    float k = 1.f / (d_max * d_max * time_const * time_const
        * damp_ratio * damp_ratio);
    float b = 2.f / (d_max * time_const);

    // a_ref first gets -b * (J @ v)
    cpu_utils::matVecMul<false, true>(a_ref_res, J, v, num_rows_j, num_cols_j);
    cpu_utils::vecScale(a_ref_res, -b, num_rows_j);

    // Then compute -k * imp * r
    for (uint32_t i = 0; i < num_rows_j; i++) {
        float imp = getImpedance(solimp, r[i]);
        a_ref_res[i] -= k * imp * r[i];
        R_res[i] = fmaxf(MINVAL, (1.f - imp) * diag_approx[i] / imp);
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
        float friction0 = mus[i + 1];
        // Regularized cone mu is mu[1] * sqrt(R[1] / R[0])
        mus[i] = friction0 * sqrtf(R[i+1] / R[i]);
        for (uint32_t j = 2; j < con_dim; j++) {
            R[i + j] = R[i + 1] * friction0 * friction0 / (
                    mus[i + j] * mus[i + j]);
        }
    }
}

inline void nonlinearCG(Context &ctx,
                        float *res,
                        float *x0,
                        float *a_ref_c,
                        float *a_ref_e,
                        float *a_ref_f,
                        float *D_c,
                        float *D_e,
                        float *R_f,
                        float tol,
                        float ls_tol,
                        bool adaptive_ls,
                        uint32_t max_iter,
                        uint32_t ls_iters,
                        CVSolveData &cv_sing)
{
    using namespace cpu_utils;
    uint32_t nv = cv_sing.totalNumDofs;
    uint32_t nv_bytes = nv * sizeof(float);
    uint32_t nc = cv_sing.numRowsJc;
    uint32_t ne = cv_sing.numRowsJe;
    uint32_t nf = cv_sing.numRowsJf;
    float scale = 1.f / cv_sing.totalMass;

    float x[nv];
    float g[nv];
    float M_grad[nv];
    float M_grad_new[nv];
    float x_min_a_free[nv];
    float Mx_min_a_free[nv];
    float p[nv];
    float Mpk[nv];

    float jar_c[nc]; // J_c x - a_ref_c
    float jar_e[ne]; // J_e x - a_ref_e
    float jar_f[nf]; // J_f x - a_ref_f
    float jp_c[nc]; // J_c p
    float jp_e[ne]; // J_e p
    float jp_f[nf]; // J_f p

    // Load initial guess, compute some values
    memcpy(x, x0, nv_bytes);
    // x - a_free and Mx - a_free
    sclAddSet(x_min_a_free, x, cv_sing.freeAcc, -1.f, nv);
    memcpy(Mx_min_a_free, x_min_a_free, nv_bytes);
    fullMSolveMul(ctx, Mx_min_a_free, false);
    // J x - a_ref
    matVecMul<false, true>(jar_c, cv_sing.J_c, x, nc, nv);
    sclAdd(jar_c, a_ref_c, -1.f, nc);
    matVecMul<false, true>(jar_e, cv_sing.J_e, x, ne, nv);
    sclAdd(jar_e, a_ref_e, -1.f, ne);
    matVecMul<false, true>(jar_f, cv_sing.J_f, x, nf, nv);
    sclAdd(jar_f, a_ref_f, -1.f, nf);

    // f(x0) and df(x0), M^{-1}df(x0)
    float fun = obj(g, x_min_a_free, Mx_min_a_free, D_c, D_e, R_f,
        jar_c, jar_e, jar_f, cv_sing);
    memcpy(M_grad, g, nv_bytes);
    fullMSolveMul(ctx, M_grad, true);
    // p = -M_grad
    memcpy(p, M_grad, nv_bytes);
    vecScale(p, -1.f, nv);
    uint32_t i = 0;
    for (; i < max_iter; i++) {
        // Exact line search
        float ada_ls_tol;
        if (adaptive_ls) {
            ada_ls_tol = tol * ls_tol * norm(p, nv) / scale;
        } else {
            ada_ls_tol = ls_tol;
        }
        float alpha = exactLineSearch(p, x_min_a_free, Mx_min_a_free, Mpk,
                                      D_c, D_e, R_f, jar_c, jar_e, jar_f,
                                      jp_c, jp_e, jp_f, nc, ne, nf, nv,
                                      ada_ls_tol, ls_iters, ctx, cv_sing);
        if (alpha == 0.f) break;

        // Update x
        sclAdd(x, p, alpha, nv);
        sclAdd(x_min_a_free, p, alpha, nv);
        sclAdd(Mx_min_a_free, Mpk, alpha, nv);
        sclAdd(jar_c, jp_c, alpha, nc);
        sclAdd(jar_e, jp_e, alpha, ne);

        // Temporary: dot(g, M_grad)
        float den = fmaxf(dot(g, M_grad, nv), MINVAL);

        // Convergence check
        float fun_new = obj(g, x_min_a_free, Mx_min_a_free, D_c, D_e, R_f,
            jar_c, jar_e, jar_f, cv_sing);
        if (scale * (fun - fun_new) < tol) break;
        if (scale * norm(g, nv) < tol) break;

        // Polak-Ribiere (Mgrad holds Mgrad_new - M_grad)
        memcpy(M_grad_new, g, nv_bytes);
        fullMSolveMul(ctx, M_grad_new, true);
        sclAdd(M_grad, M_grad_new, -1, nv);
        // Now holds (M_grad - Mgrad_new)
        vecScale(M_grad, -1, nv);
        float beta = dot(g, M_grad, nv) / den;
        beta = fmaxf(0.f, beta);

        // Update p_new = beta * p - Mgrad_new
        vecScale(p, beta, nv);
        sclAdd(p, M_grad_new, -1, nv);
        fun = fun_new;
        memcpy(M_grad, M_grad_new, nv_bytes);
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
                 float *x_min_a_free,
                 float *Mx_min_a_free,
                 float *D_c,
                 float *D_e,
                 float *R_f,
                 float *jar_c,
                 float *jar_e,
                 float *jar_f,
                 CVSolveData &cv_sing)
{
    using namespace cpu_utils;
    uint32_t nv = cv_sing.totalNumDofs;
    uint32_t nv_bytes = nv * sizeof(float);

    float cost = 0.f;
    memset(grad_out, 0, nv_bytes);

    // Gauss: f(x) = 0.5 (x - a_free)^T M (x - a_free), grad = M (x - a_free)
    cost += 0.5f * dot(x_min_a_free, Mx_min_a_free, nv);
    sclAdd(grad_out, Mx_min_a_free, 1, nv);

    // Contact constraints
    uint32_t nc = cv_sing.numRowsJc;
    float grad_c[nc];
    cost += s_c(grad_c, jar_c, D_c, cv_sing.mu, nc);
    // J.T @ grad_c
    matVecMul<true, false>(grad_out, cv_sing.J_c, grad_c, nc, nv);

    // Equation constraints
    uint32_t ne = cv_sing.numRowsJe;
    float grad_e[ne];
    cost += s_e(grad_e, jar_e, D_e, ne);
    // J.T @ grad_e
    matVecMul<true, false>(grad_out, cv_sing.J_e, grad_e, ne, nv);

    // Frictional constraints
    uint32_t nf = cv_sing.numRowsJf;
    float grad_f[nf];
    cost += s_f(grad_f, jar_f, R_f, cv_sing.floss, nf);
    // J.T @ grad_f
    matVecMul<true, false>(grad_out, cv_sing.J_f, grad_f, nf, nv);


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

        // Top zone
        if (N >= mu * T || (T <= 0 && N >= 0)) {
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
            float Dm = Dn / (mu * mu * (1 + mu * mu));
            float NmT = N - mu * T;
            cost += 0.5f * Dm * NmT * NmT;
            float tmp = Dm * NmT * mu;
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
    for (uint32_t i = 0; i < dim; i++) {
        // Constraint is satisfied
        if (jar[i] >= 0.f) {
            grad_out[i] = 0.f;
            continue;
        }
        cost += 0.5f * D_e[i] * jar[i] * jar[i];
        grad_out[i] = D_e[i] * jar[i];
    }
    return cost;
}

float s_f(float *grad_out,
          float *jar,
          float *R_f,
          float *floss,
          uint32_t dim) {
    float cost = 0.f;
    for (uint32_t i = 0; i < dim; i++) {
        // linear negative
        if (jar[i] <= -R_f[i]*floss[i]) {
            cost += -0.5f * R_f[i] * floss[i]*floss[i] - floss[i]*jar[i];
            grad_out[i] = -floss[i];
        }
        // linear positive
        else if (jar[i] >= R_f[i]*floss[i]) {
            cost += -0.5f * R_f[i] * floss[i]*floss[i] + floss[i]*jar[i];
            grad_out[i] = floss[i];
        }
        // quadratic
        else {
            cost += 0.5f * (1.f / R_f[i]) * jar[i]*jar[i];
        }
    }
    return cost;
}

float exactLineSearch(float *pk, float *x_min_a_free,
                      float *Mx_min_a_free, float *Mpk,
                      float *D_c, float *D_e, float *R_f,
                      float *jar_c, float *jar_e, float *jar_f,
                      float *Jp_c, float *Jp_e, float *Jp_f,
                      uint32_t nc, uint32_t ne, uint32_t nf,
                      uint32_t nv, float ls_tol, uint32_t ls_iters,
                      Context &ctx, CVSolveData &cv_sing) {
    using namespace cpu_utils;
    uint32_t iter = 0;

    struct Evals {
        float alpha; // useful to save
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

    struct LimitStore {
        float quad0;
        float quad1;
        float quad2;
        float Jx;
    };

    struct FrictionStore {
        float quad0;
        float quad1;
        float quad2;
        float Rf;
        float floss;
        float Jx;
    };

    // Search vector too small
    if (norm(pk, nv) < MINVAL) return 0.f;
    uint32_t nv_bytes = nv * sizeof(float);

    // M @ pk
    memcpy(Mpk, pk, nv_bytes);
    fullMSolveMul(ctx, Mpk, false);

    // Precompute some values
    // 1. Gauss objective
    float x_min_M_x_min = dot(x_min_a_free, Mx_min_a_free, nv);
    float pMp = dot(pk, Mpk, nv);
    float pMx_free = dot(pk, Mx_min_a_free, nv);
    float quadGauss[3] = {0.5f * x_min_M_x_min, pMx_free, 0.5f * pMp};
    // 2. Cone constraints
    matVecMul<false, true>(Jp_c, cv_sing.J_c, pk, cv_sing.numRowsJc, nv);
    // 3. Equality constraints
    matVecMul<false, true>(Jp_e, cv_sing.J_e, pk, cv_sing.numRowsJe, nv);
    // 4. Friction loss
    matVecMul<false, true>(Jp_f, cv_sing.J_f, pk, cv_sing.numRowsJf, nv);

    // --- Precomputation of constraints ---
    // For each contact, store some information to avoid re-computation
    ContactStore cs[nc / 3];
    for (uint32_t i = 0; i < nc / 3; i++) {
        // Components of J @ x - a_ref, J @ p
        float Jx_N = jar_c[3 * i];
        float Jx_T1 = jar_c[3 * i + 1];
        float Jx_T2 = jar_c[3 * i + 2];
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

    LimitStore ls[ne];
    for (uint32_t i = 0; i < ne; i++) {
        float Jx = jar_e[i];
        float Jp = Jp_e[i];
        float de = D_e[i];
        ls[i].quad0 = 0.5f * de * Jx * Jx;
        ls[i].quad1 = de * Jx * Jp;
        ls[i].quad2 = 0.5f * de * Jp * Jp;
        ls[i].Jx = Jx;
    }

    FrictionStore ls_f[nf];
    for (uint32_t i = 0; i < nf; i++) {
        float Jx = jar_f[i];
        float Jp = Jp_f[i];
        float df = 1.f / R_f[i];
        ls_f[i].quad0 = 0.5f * df * Jx * Jx;
        ls_f[i].quad1 = df * Jx * Jp;
        ls_f[i].quad2 = 0.5f * df * Jp * Jp;
        ls_f[i].floss = cv_sing.floss[i];
        ls_f[i].Rf = R_f[i];
        ls_f[i].Jx = Jx;
    }

    // Line search function: updates with phi(a), d_phi(a), d2_phi(a)
    float quadTotal[3];
    auto phi = [&](Evals *evals) {
        float a = evals->alpha;
        float fun = 0.f;
        float grad = 0.f;
        float hess = 0.f;

        // Quadratic costs (wrst alpha^0, alpha^1, alpha^2)
        // process Gauss first
        quadTotal[0] = quadGauss[0];
        quadTotal[1] = quadGauss[1];
        quadTotal[2] = quadGauss[2];

        // Then process cones
        for (uint32_t i = 0; i < nc / 3; i++) {
            ContactStore c = cs[i];
            float mu = cv_sing.mu[3 * i];
            float N = c.U0 + a * c.V0;
            float T_sqr = c.UU + a * (2 * c.UV + a * c.VV);

            // No tangent force (just top or bottom)
            if (T_sqr <= 0) {
                // Bottom zone
                if (N < 0) {
                    quadTotal[0] += c.quad0;
                    quadTotal[1] += c.quad1;
                    quadTotal[2] += c.quad2;
                }
            }
            else {
                // Proceed as normal. Top zone
                float T = sqrtf(T_sqr);
                if (N >= mu * T) {
                    // nothing
                }
                // Bottom zone
                else if (mu * N + T <= 0) {
                    quadTotal[0] += c.quad0;
                    quadTotal[1] += c.quad1;
                    quadTotal[2] += c.quad2;
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
        }

        // process equality constraints
        for (uint32_t i = 0; i < ne; i++) {
            LimitStore ls_i = ls[i];
            float jx = ls_i.Jx;
            float jp = Jp_e[i];
            float N_search = jx + a * jp;
            // Active limit
            if (N_search < 0.f) {
                quadTotal[0] += ls_i.quad0;
                quadTotal[1] += ls_i.quad1;
                quadTotal[2] += ls_i.quad2;
            }
        }

        // process friction loss constraints
        for (uint32_t i = 0; i < nf; i++) {
            FrictionStore ls_i = ls_f[i];
            float jx = ls_i.Jx;
            float jp = Jp_e[i];
            float N_search = jx + a * jp;
            // Quadratic
            if (-ls_i.Rf < N_search && N_search < ls_i.Rf) {
                quadTotal[0] += ls_i.quad0;
                quadTotal[1] += ls_i.quad1;
                quadTotal[2] += ls_i.quad2;
            }

            else if (N_search <= -ls_i.Rf) {
                quadTotal[0] += -ls_i.floss * (0.5f * ls_i.Rf + jx);
                quadTotal[1] += -ls_i.floss * jp;
            }
            else {
                quadTotal[0] += -ls_i.floss * (0.5f * ls_i.Rf - jx);
                quadTotal[1] += ls_i.floss * jp;
            }
        }

        // Evaluate quadratic at alpha=a
        fun += a * a * quadTotal[2] + a * quadTotal[1] + quadTotal[0];
        grad += 2 * a * quadTotal[2] + quadTotal[1];
        hess += 2 * quadTotal[2];

        evals->fun = fun;
        evals->grad = grad;
        evals->hess = hess;
        iter++;
    };

    auto updateBracket = [&](Evals *p, Evals candidates[3], Evals *pnext) {
        int flag = 0;
        for (int i = 0; i < 3; i++) {
            if (p->grad < 0 && candidates[i].grad < 0
                && p->grad < candidates[i].grad) {
                *p = candidates[i];
                flag = 1;
            }
            else if (p->grad > 0 && candidates[i].grad > 0 &&
                     p->grad > candidates[i].grad) {
                *p = candidates[i];
                flag = 2;
            }
        }

        if (flag) {
            pnext->alpha = p->alpha - p->grad / p->hess;
            phi(pnext);
        }
        return flag;
    };

    Evals p0, p1, p2, pmid, p1next, p2next;
    p0.alpha = 0.f;
    phi(&p0);

    p1.alpha = p0.alpha - p0.grad / p0.hess;
    phi(&p1);
    if (p0.fun < p1.fun) {
        p1 = p0;
    }

    // Initial convergence
    if (fabsf(p1.grad) < ls_tol) {
        return p1.alpha;
    }

    // Opposing direction of gradient at alpha1
    float dir = p1.grad < 0.f ? 1.f : -1.f;

    // One-sided search
    while (p1.grad * dir <= -ls_tol && iter < ls_iters) {
        p2 = p1;

        // Newton step
        p1.alpha -= p1.grad / p1.hess;
        phi(&p1);

        // Check for convergence
        if (fabs(p1.grad) < ls_tol) {
            return p1.alpha;
        }
    }

    // Failed to bracket
    if (iter >= ls_iters) {
        return p1.alpha;
    }

    // Bracketing to find where d_phi equals zero
    p2next = p1;
    p1next.alpha = p1.alpha - p1.grad / p1.hess;
    phi(&p1next);

    // Bracketed search
    while (iter < ls_iters) {
        // midpoint evaluation
        pmid.alpha = 0.5f * (p1.alpha + p2.alpha);
        phi(&pmid);

        Evals candidates[3] = {p1next, p2next, pmid};
        // check candidates for convergence
        float best_cost = 0;
        int best_ind = -1;
        for (int i = 0; i < 3; i++) {
            if (fabsf(candidates[i].grad) < ls_tol &&
                (best_ind == -1 || candidates[i].fun < best_cost)) {
            best_cost = candidates[i].fun;
            best_ind = i;
          }
        }
        if (best_ind >= 0) {
            return candidates[best_ind].alpha;
        }

        // Update brackets
        int b1 = updateBracket(&p1, candidates, &p1next);
        int b2 = updateBracket(&p2, candidates, &p2next);
        // use midpoint if can't update
        if (!b1 && !b2) {
          return pmid.alpha;
        }
    }


    // choose bracket with best cost
    if (p1.fun <= p2.fun && p1.fun < p0.fun) {
        return p1.alpha;
    }
    if (p2.fun <= p1.fun && p2.fun < p0.fun) {
        return p2.alpha;
    }
    return 0;
}

}
}
#endif
