#pragma once

namespace llm_engine {
namespace arm_neon {

    
void gemm_kernel_8x12_neon(
    const float* A,
    const float* B,
    float* C,
    int K,
    int ldc
);

}
}