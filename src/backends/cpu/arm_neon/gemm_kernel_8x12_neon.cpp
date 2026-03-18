#include <arm_neon.h>
#include <stddef.h>

#include "gemm.h"

namespace llm_engine {
namespace arm_neon {

/*
C = A * B + C
*/
void gemm_kernel_8x12_neon(
    const float* A,
    const float* B,
    float* C,
    int K,
    int ldc
) {
    // 1. 初始化 24 个累加器 (占用 24 个寄存器)
    float32x4_t c00 = vdupq_n_f32(0), c01 = vdupq_n_f32(0), c02 = vdupq_n_f32(0);
    float32x4_t c10 = vdupq_n_f32(0), c11 = vdupq_n_f32(0), c12 = vdupq_n_f32(0);
    float32x4_t c20 = vdupq_n_f32(0), c21 = vdupq_n_f32(0), c22 = vdupq_n_f32(0);
    float32x4_t c30 = vdupq_n_f32(0), c31 = vdupq_n_f32(0), c32 = vdupq_n_f32(0);
    float32x4_t c40 = vdupq_n_f32(0), c41 = vdupq_n_f32(0), c42 = vdupq_n_f32(0);
    float32x4_t c50 = vdupq_n_f32(0), c51 = vdupq_n_f32(0), c52 = vdupq_n_f32(0);
    float32x4_t c60 = vdupq_n_f32(0), c61 = vdupq_n_f32(0), c62 = vdupq_n_f32(0);
    float32x4_t c70 = vdupq_n_f32(0), c71 = vdupq_n_f32(0), c72 = vdupq_n_f32(0);

    for(int k = 0; k < K; k++) {
        // 2. 加载 B 面板 (占用 3 个寄存器，整个 K 循环复用)
        float32x4_t b0 = vld1q_f32(B + 0);
        float32x4_t b1 = vld1q_f32(B + 4);
        float32x4_t b2 = vld1q_f32(B + 8);

        // 3. 流式加载 A 并计算 (每次只占用 1 个寄存器)
        // 第 0 行
        float32x4_t a_val = vdupq_n_f32(A[0]);
        c00 = vfmaq_f32(c00, b0, a_val); c01 = vfmaq_f32(c01, b1, a_val); c02 = vfmaq_f32(c02, b2, a_val);
        
        // 第 1 行
        a_val = vdupq_n_f32(A[1]);
        c10 = vfmaq_f32(c10, b0, a_val); c11 = vfmaq_f32(c11, b1, a_val); c12 = vfmaq_f32(c12, b2, a_val);

        // 第 2 行
        a_val = vdupq_n_f32(A[2]);
        c20 = vfmaq_f32(c20, b0, a_val); c21 = vfmaq_f32(c21, b1, a_val); c22 = vfmaq_f32(c22, b2, a_val);

        // 第 3 行
        a_val = vdupq_n_f32(A[3]);
        c30 = vfmaq_f32(c30, b0, a_val); c31 = vfmaq_f32(c31, b1, a_val); c32 = vfmaq_f32(c32, b2, a_val);

        // 第 4 行
        a_val = vdupq_n_f32(A[4]);
        c40 = vfmaq_f32(c40, b0, a_val); c41 = vfmaq_f32(c41, b1, a_val); c42 = vfmaq_f32(c42, b2, a_val);

        // 第 5 行
        a_val = vdupq_n_f32(A[5]);
        c50 = vfmaq_f32(c50, b0, a_val); c51 = vfmaq_f32(c51, b1, a_val); c52 = vfmaq_f32(c52, b2, a_val);

        // 第 6 行
        a_val = vdupq_n_f32(A[6]);
        c60 = vfmaq_f32(c60, b0, a_val); c61 = vfmaq_f32(c61, b1, a_val); c62 = vfmaq_f32(c62, b2, a_val);

        // 第 7 行
        a_val = vdupq_n_f32(A[7]);
        c70 = vfmaq_f32(c70, b0, a_val); c71 = vfmaq_f32(c71, b1, a_val); c72 = vfmaq_f32(c72, b2, a_val);

        // 4. 指针步进
        A += 8;
        B += 12;
    }

    // 5. 写回内存
    float* c_ptr = C;
    vst1q_f32(c_ptr + 0 * ldc, c00); vst1q_f32(c_ptr + 0 * ldc + 4, c01); vst1q_f32(c_ptr + 0 * ldc + 8, c02);
    vst1q_f32(c_ptr + 1 * ldc, c10); vst1q_f32(c_ptr + 1 * ldc + 4, c11); vst1q_f32(c_ptr + 1 * ldc + 8, c12);
    vst1q_f32(c_ptr + 2 * ldc, c20); vst1q_f32(c_ptr + 2 * ldc + 4, c21); vst1q_f32(c_ptr + 2 * ldc + 8, c22);
    vst1q_f32(c_ptr + 3 * ldc, c30); vst1q_f32(c_ptr + 3 * ldc + 4, c31); vst1q_f32(c_ptr + 3 * ldc + 8, c32);
    vst1q_f32(c_ptr + 4 * ldc, c40); vst1q_f32(c_ptr + 4 * ldc + 4, c41); vst1q_f32(c_ptr + 4 * ldc + 8, c42);
    vst1q_f32(c_ptr + 5 * ldc, c50); vst1q_f32(c_ptr + 5 * ldc + 4, c51); vst1q_f32(c_ptr + 5 * ldc + 8, c52);
    vst1q_f32(c_ptr + 6 * ldc, c60); vst1q_f32(c_ptr + 6 * ldc + 4, c61); vst1q_f32(c_ptr + 6 * ldc + 8, c62);
    vst1q_f32(c_ptr + 7 * ldc, c70); vst1q_f32(c_ptr + 7 * ldc + 4, c71); vst1q_f32(c_ptr + 7 * ldc + 8, c72);
}

} // namespace arm_neon
} // namespace llm_engine