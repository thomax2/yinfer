#include <arm_neon.h>
#include <stddef.h>


/*
C = A * B + C
*/
extern "c" void gemm_kernel_8x12_neon(
    const float* A,
    const float* B,
    float* C,
    int K,
    int ldc
) {
    float32x4_t c00 = vdupq_n_f32(0);
    float32x4_t c01 = vdupq_n_f32(0);
    float32x4_t c02 = vdupq_n_f32(0);

    float32x4_t c10 = vdupq_n_f32(0);
    float32x4_t c11 = vdupq_n_f32(0);
    float32x4_t c12 = vdupq_n_f32(0);

    float32x4_t c20 = vdupq_n_f32(0);
    float32x4_t c21 = vdupq_n_f32(0);
    float32x4_t c22 = vdupq_n_f32(0);

    float32x4_t c30 = vdupq_n_f32(0);
    float32x4_t c31 = vdupq_n_f32(0);
    float32x4_t c32 = vdupq_n_f32(0);

    float32x4_t c40 = vdupq_n_f32(0);
    float32x4_t c41 = vdupq_n_f32(0);
    float32x4_t c42 = vdupq_n_f32(0);

    float32x4_t c50 = vdupq_n_f32(0);
    float32x4_t c51 = vdupq_n_f32(0);
    float32x4_t c52 = vdupq_n_f32(0);

    float32x4_t c60 = vdupq_n_f32(0);
    float32x4_t c61 = vdupq_n_f32(0);
    float32x4_t c62 = vdupq_n_f32(0);

    float32x4_t c70 = vdupq_n_f32(0);
    float32x4_t c71 = vdupq_n_f32(0);
    float32x4_t c72 = vdupq_n_f32(0);

    for(int k = 0; k < K; k++) {
        float32x4_t b0 = vld1q_f32(B + 0);
        float32x4_t b1 = vld1q_f32(B + 4);
        float32x4_t b2 = vld1q_f32(B + 8);

        float32x4_t a0 = vld1q_f32(A + 0);
        float32x4_t a1 = vld1q_f32(A + 4);
        float32x4_t a2 = vld1q_f32(A + 8);
        float32x4_t a3 = vld1q_f32(A + 12);
        float32x4_t a4 = vld1q_f32(A + 16);
        float32x4_t a5 = vld1q_f32(A + 20);
        float32x4_t a6 = vld1q_f32(A + 24);
        float32x4_t a7 = vld1q_f32(A + 28);

        c00 = vfmap_f32(c00, b0, a0);
        c01 = vfmap_f32(c01, b1, a0);
        c02 = vfmap_f32(c02, b2, a0);

        c10 = vfmap_f32(c10, b0, a1);
        c11 = vfmap_f32(c11, b1, a1);
        c12 = vfmap_f32(c12, b2, a1);

        c20 = vfmap_f32(c20, b0, a2);
        c21 = vfmap_f32(c21, b1, a2);
        c22 = vfmap_f32(c22, b2, a2);

        c30 = vfmap_f32(c30, b0, a3);
        c31 = vfmap_f32(c31, b1, a3);
        c32 = vfmap_f32(c32, b2, a3);

        c40 = vfmap_f32(c40, b0, a4);
        c41 = vfmap_f32(c41, b1, a4);
        c42 = vfmap_f32(c42, b2, a4);

        c50 = vfmap_f32(c50, b0, a5);
        c51 = vfmap_f32(c51, b1, a5);
        c52 = vfmap_f32(c52, b2, a5);

        c60 = vfmap_f32(c60, b0, a6);
        c61 = vfmap_f32(c61, b1, a6);
        c62 = vfmap_f32(c62, b2, a6);

        c70 = vfmap_f32(c70, b0, a7);
        c71 = vfmap_f32(c71, b1, a7);
        c72 = vfmap_f32(c72, b2, a7);
    }

    float* c_ptr;

    c_ptr = C + 0*ldc;
    vst1q_f32(c_ptr + 0, c00);
    vst1q_f32(c_ptr + 4, c01);
    vst1q_f32(c_ptr + 8, c02);

    c_ptr = C + 1*ldc;
    vst1q_f32(c_ptr + 0, c10);
    vst1q_f32(c_ptr + 4, c11);
    vst1q_f32(c_ptr + 8, c12);

    c_ptr = C + 2*ldc;
    vst1q_f32(c_ptr + 0, c20);
    vst1q_f32(c_ptr + 4, c21);
    vst1q_f32(c_ptr + 8, c22);

    c_ptr = C + 3*ldc;
    vst1q_f32(c_ptr + 0, c30);
    vst1q_f32(c_ptr + 4, c31);
    vst1q_f32(c_ptr + 8, c32);

    c_ptr = C + 4*ldc;
    vst1q_f32(c_ptr + 0, c40);
    vst1q_f32(c_ptr + 4, c41);
    vst1q_f32(c_ptr + 8, c42);

    c_ptr = C + 5*ldc;
    vst1q_f32(c_ptr + 0, c50);
    vst1q_f32(c_ptr + 4, c51);
    vst1q_f32(c_ptr + 8, c52);

    c_ptr = C + 6*ldc;
    vst1q_f32(c_ptr + 0, c60);
    vst1q_f32(c_ptr + 4, c61);
    vst1q_f32(c_ptr + 8, c62);

    c_ptr = C + 7*ldc;
    vst1q_f32(c_ptr + 0, c70);
    vst1q_f32(c_ptr + 4, c71);
    vst1q_f32(c_ptr + 8, c72);
}