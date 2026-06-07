#include "backends/cpu/arm_neon/gemm.h"

#include <arm_neon.h>

namespace llm_engine {
namespace arm_neon {

void gemm_kernel_f16_8x16_neon(
    const fp16_t* A,
    const fp16_t* B,
    fp16_t* C,
    int K,
    int ldc
) {
    float16x8_t acc[8][2];
    for (int r = 0; r < 8; ++r) {
        acc[r][0] = vdupq_n_f16((fp16_t)0);
        acc[r][1] = vdupq_n_f16((fp16_t)0);
    }

    for (int k = 0; k < K; ++k) {
        const fp16_t* b_row = B + k * 16;
        float16x8_t b0 = vld1q_f16(b_row);
        float16x8_t b1 = vld1q_f16(b_row + 8);

        for (int r = 0; r < 8; ++r) {
            float16x8_t a = vdupq_n_f16(A[k * 8 + r]);
            acc[r][0] = vfmaq_f16(acc[r][0], a, b0);
            acc[r][1] = vfmaq_f16(acc[r][1], a, b1);
        }
    }

    for (int r = 0; r < 8; ++r) {
        vst1q_f16(C + r * ldc, acc[r][0]);
        vst1q_f16(C + r * ldc + 8, acc[r][1]);
    }
}

} // namespace arm_neon
} // namespace llm_engine
