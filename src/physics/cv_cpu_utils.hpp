#pragma once

#ifndef MADRONA_GPU_MODE
namespace madrona::phys::cv {

namespace cpu_utils {
template <bool transposed = false, bool reset = true>
void matVecMul(float *res,
               float *mat,
               float *vec,
               uint32_t num_rows,
               uint32_t num_cols);

float dot(float *vec_a,
          float *vec_b,
          uint32_t dim);

float norm(float *vec, uint32_t dim);

void sclAdd(float *res,
            float *vec,
            float coeff,
            uint32_t dim);

inline void sclAddSet(float *res,
                      float *a,
                      float *b,
                      float coeff,
                      uint32_t dim);

void vecScale(float *res,
              float coeff,
              uint32_t dim);

void vecCopy(float *res,
             float *vec,
             uint32_t dim);

}
}
#endif

#include "cv_cpu_utils.inl"
