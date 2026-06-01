#include "backends/cpu/arm_neon/neon_ops.h"
#include "backends/cpu/arm_neon/kernel_common.h"
#include <algorithm>
#include <arm_neon.h>

namespace llm_engine {
namespace arm_neon {

Status linear_decode_prepacked_neon(
    const float* x,
    const float* w_pack,
    float* y,
    int K,
    int N,
    const float* bias
) {
    if (!x || !w_pack || !y || K <= 0 || N <= 0) {
        return Status::INVALID_ARGUMENT;
    }

    int np = (N + NR - 1) / NR;
    for (int panel = 0; panel < np; ++panel) {
        const float* wp = w_pack + panel * K * NR;

        float32x4_t acc0 = vdupq_n_f32(0.0f);
        float32x4_t acc1 = vdupq_n_f32(0.0f);
        float32x4_t acc2 = vdupq_n_f32(0.0f);

        for (int k = 0; k < K; ++k) {
            const float* w_row = wp + k * NR;
            float xv = x[k];

            acc0 = vfmaq_n_f32(acc0, vld1q_f32(w_row), xv);
            acc1 = vfmaq_n_f32(acc1, vld1q_f32(w_row + 4), xv);
            acc2 = vfmaq_n_f32(acc2, vld1q_f32(w_row + 8), xv);
        }

        int col = panel * NR;
        int actual_n = std::min(NR, N - col);

        if (bias) {
            if (actual_n == NR) {
                acc0 = vaddq_f32(acc0, vld1q_f32(bias + col));
                acc1 = vaddq_f32(acc1, vld1q_f32(bias + col + 4));
                acc2 = vaddq_f32(acc2, vld1q_f32(bias + col + 8));
                vst1q_f32(y + col, acc0);
                vst1q_f32(y + col + 4, acc1);
                vst1q_f32(y + col + 8, acc2);
            } else {
                float temp[NR];
                vst1q_f32(temp, acc0);
                vst1q_f32(temp + 4, acc1);
                vst1q_f32(temp + 8, acc2);
                for (int i = 0; i < actual_n; ++i) {
                    y[col + i] = temp[i] + bias[col + i];
                }
            }
        } else {
            if (actual_n == NR) {
                vst1q_f32(y + col, acc0);
                vst1q_f32(y + col + 4, acc1);
                vst1q_f32(y + col + 8, acc2);
            } else {
                float temp[NR];
                vst1q_f32(temp, acc0);
                vst1q_f32(temp + 4, acc1);
                vst1q_f32(temp + 8, acc2);
                for (int i = 0; i < actual_n; ++i) {
                    y[col + i] = temp[i];
                }
            }
        }
    }

    return Status::SUCCESS;
}

} // namespace arm_neon
} // namespace llm_engine
