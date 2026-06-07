#pragma once

#include "llm_engine/tensor.h"

namespace llm_engine {
namespace arm_neon {

    
void gemm_kernel_8x12_neon(
    const float* A,
    const float* B,
    float* C,
    int K,
    int ldc
);

void gemm_kernel_f16_8x16_neon(
    const fp16_t* A,
    const fp16_t* B,
    fp16_t* C,
    int K,
    int ldc
);

}
}
