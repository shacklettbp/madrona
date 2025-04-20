#pragma once

#ifndef MADRONA_GPU_MODE
namespace madrona::phys::cv {
namespace cpu_solver {

// Pre-processed data for each constraint
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

struct SolverMemory {
    uint32_t nv;
    uint32_t nc;
    uint32_t ne;
    uint32_t nf;

    MemoryRange mem;
    void *memPtr;
    inline float * x();
    inline float * g();
    inline float * M_grad();
    inline float * M_grad_new();
    inline float * x_min_a_free();
    inline float * Mx_min_a_free();
    inline float * p();
    inline float * Mpk();
    // J @ x - a_ref for each constraint type
    inline float * jar_c();
    inline float * jar_e();
    inline float * jar_f();
    // J @ p for each constraint type
    inline float * jp_c();
    inline float * jp_e();
    inline float * jp_f();
    // Gradients of objective function for each constraint
    inline float * grad_c();
    inline float * grad_e();
    inline float * grad_f();
    // Reference accelerations
    inline float * acc_ref_c();
    inline float * acc_ref_e();
    inline float * acc_ref_f();
    // Constraint mass
    inline float * D_c();
    inline float * D_e();
    // Inverse constraint mass
    inline float * R_c();
    inline float * R_e();
    inline float * R_f();
    // Residuals
    inline float * res_c();
    inline float * res_e();
    inline float * res_f();
    // Pre-processed line search data
    inline ContactStore * cs();
    inline LimitStore * ls();
    inline FrictionStore * fs();

    static uint32_t numReqBytes(uint32_t nv,
                                uint32_t nc,
                                uint32_t ne,
                                uint32_t nf)
    {
        return sizeof(float) * (8 * nv + 7 * nc + 7 * ne + 6 * nf) +
               sizeof(ContactStore) * (nc / 3) +
               sizeof(LimitStore) * ne +
               sizeof(FrictionStore) * nf;
    }
};

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
                 SolverMemory &sm,
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
          float *R_f,
          float *jar_c,
          float *jar_e,
          float *jar_f,
          float *grad_c,
          float *grad_e,
          float *grad_f,
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

float s_f(float *grad_out,
          float *jar,
          float *R_f,
          float *floss,
          uint32_t dim);

float exactLineSearch(
    float *pk,
    float *x_min_a_free,
    float *Mx_min_a_free,
    float *Mpk,
    float *D_c,
    float *D_e,
    float *R_f,
    float *jar_c,
    float *jar_e,
    float *jar_f,
    float *Jp_c,
    float *Jp_e,
    float *Jp_f,
    ContactStore *cs,
    LimitStore *ls,
    FrictionStore *fs,
    uint32_t nc,
    uint32_t ne,
    uint32_t nf,
    uint32_t nv,
    float ls_tol,
    uint32_t ls_iters,
    Context &ctx,
    CVSolveData &cv_sing);


}
}
#endif

#include "cv_cpu.inl"
