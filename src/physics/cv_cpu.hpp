#pragma once

#include <madrona/math.hpp>

#ifndef MADRONA_GPU_MODE
namespace madrona::phys::cv {
namespace cpu_solver {

void computeAccRef(
        float *a_ref_res,
        float *R_res,
        float *r,
        float *v,
        float *J,
        uint32_t num_rows_j,
        uint32_t num_cols_j,
        float *diag_approx,
        float h);

void adjustContactRegularization(
        float *R,
        float *mus,
        uint32_t dim);

void nonlinearCG(Context &ctx,
                 float *res,
                 float *x0,
                 float *a_ref_c,
                 float *a_ref_e,
                 float *D_c,
                 float *D_e,
                 float tol,
                 float ls_tol,
                 bool adaptive_ls,
                 uint32_t max_iter,
                 uint32_t ls_iters,
                 CVSolveData &cv_sing);

void fullMSolveMul(Context &ctx, float *x, bool solve);

float obj(float *grad_out,
          float *x_min_a_free,
          float *Mx_min_a_free,
          float *D_c,
          float *D_e,
          float *jar_c,
          float *jar_e,
          CVSolveData &cv_sing);

float s_c(float *grad_out,
          float *jar,
          float *D_c,
          float *mus,
          uint32_t dim);

float s_e(float *grad_out,
          float *jar,
          float *D_e,
          uint32_t dim);

float exactLineSearch(
    float *pk,
    float *x_min_a_free,
    float *Mx_min_a_free,
    float *Mpk,
    float *D_c,
    float *D_e,
    float *jar_c,
    float *jar_e,
    float *Jp_c,
    float *Jp_e,
    uint32_t nc,
    uint32_t ne,
    uint32_t nv,
    float ls_tol,
    uint32_t ls_iters,
    Context &ctx,
    CVSolveData &cv_sing);


}
}
#endif

#include "cv_cpu.inl"
