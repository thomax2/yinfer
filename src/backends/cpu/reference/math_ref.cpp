#include "math_ref.h"

namespace llm_engine {
namespace reference {

Status add_ref(
    const Tensor& A,
    const Tensor& B,
    Tensor& C
) {

    if (A.size() != B.size() || A.size() != C.size())
        return Status::SHAPE_MISMATCH;

    float* a = (float*)A.data;
    float* b = (float*)B.data;
    float* c = (float*)C.data;

    for (size_t i = 0; i < A.size(); ++i) {
        c[i] = a[i] + b[i];
    }

    return Status::SUCCESS;
}


Status matmul_ref(
    const Tensor& A,
    const Tensor& B,
    Tensor& C
) {

    if (A.shape.size() != 2 || B.shape.size() != 2)
        return Status::INVALID_ARGUMENT;

    int M = A.shape[0];
    int K = A.shape[1];
    int K2 = B.shape[0];
    int N = B.shape[1];

    if (K != K2)
        return Status::SHAPE_MISMATCH;

    float* a = (float*)A.data;
    float* b = (float*)B.data;
    float* c = (float*)C.data;

    for (int i = 0; i < M; i++) {

        for (int j = 0; j < N; j++) {

            float sum = 0;

            for (int k = 0; k < K; k++) {

                sum += a[i*K + k] * b[k*N + j];

            }

            c[i*N + j] = sum;

        }
    }

    return Status::SUCCESS;
}

} // reference
} // llm_engine