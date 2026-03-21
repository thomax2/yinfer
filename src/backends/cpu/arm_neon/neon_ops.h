#pragma once

#include "llm_engine/tensor.h"
#include "llm_engine/status.h"

namespace llm_engine {
namespace arm_neon {

Status matmul_neon(
    const Tensor& A,
    const Tensor& B,
    Tensor& C,
    float* workspace,
    bool transB = false
);

void add_neon(
    const Tensor& A,
    const Tensor& B,
    Tensor& C
);

void rmsnorm_neon(
    const float* x,
    const float* weight,
    float* y,
    int n,
    float eps
);

void rope_neon(
    float* x,
    const float* cos,
    const float* sin,
    int n
);

void swiglu_neon(
    const float* x,
    const float* up,
    float* y,
    int n
);

}
}