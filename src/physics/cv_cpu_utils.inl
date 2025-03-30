#ifndef MADRONA_GPU_MODE

namespace madrona::phys::cv {
namespace cpu_utils {

template <bool transposed, bool reset>
void matVecMul(float *res,
               float *mat,
               float *vec,
               uint32_t num_rows,
               uint32_t num_cols)
{
    for (uint32_t i = 0; i < num_rows; i++) {
        if (reset) { res[i] = 0.f; }
        for (uint32_t j = 0; j < num_cols; j++) {
            if (transposed) {
                res[i] += mat[j * num_rows + i] * vec[j];
            } else {
                res[i] += mat[i * num_cols + j] * vec[j];
            }
        }
    }
}

inline float dot(float *vec_a, float *vec_b, uint32_t dim) {
    float res = 0.f;
    for (uint32_t i = 0; i < dim; i++) {
        res += vec_a[i] * vec_b[i];
    }
    return res;
}

inline float norm(float *vec, uint32_t dim) {
    return sqrtf(dot(vec, vec, dim));
}

inline void sclAdd(float *res, float *vec, float coeff, uint32_t dim) {
    for (uint32_t i = 0; i < dim; i++) {
        res[i] += coeff * vec[i];
    }
}

inline void vecScale(float *res, float coeff, uint32_t dim) {
    for (uint32_t i = 0; i < dim; i++) {
        res[i] *= coeff;
    }
}

inline void vecCopy(float *res, float *vec, uint32_t dim) {
    for (uint32_t i = 0; i < dim; i++) {
        res[i] = vec[i];
    }
}

}
}
#endif
