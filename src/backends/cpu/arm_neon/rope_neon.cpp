#include "neon_ops.h"
#include <arm_neon.h>

namespace llm_engine {
namespace arm_neon {

/*
    x 是输入矩阵 [seq_len, hidden_dim] 其中一行的指针
    
    实现：
        xi = xi * cos + x(i+1) * sin
        x(i+1) = x(i+1) * cos - xi * sin

    cos = cos(theta), theta = pos / (10000^(2i/dim))
    cos 和 sin 是预先计算好的旋转矩阵，存储在内存中，长度和输入 x 的维度相同。
*/ 
void rope_neon(
    float* x,
    const float* cos,
    const float* sin,
    int n
) {
    int half_n = n / 2;
    // 每次处理 4 个 float，完美契合 128-bit 寄存器
    for (int i = 0; i < half_n; i += 4) {
        // 分别加载前半段和后半段
        float32x4_t x1 = vld1q_f32(x + i);
        float32x4_t x2 = vld1q_f32(x + i + half_n);
        
        // 加载预计算好的 cos 和 sin
        float32x4_t c = vld1q_f32(cos + i);
        float32x4_t s = vld1q_f32(sin + i);
        
        // 执行 Half-and-Half 旋转公式：
        // x1_new = x1 * cos - x2 * sin
        float32x4_t x1_new = vmlsq_f32(vmulq_f32(x1, c), x2, s);
        
        // x2_new = x2 * cos + x1 * sin
        float32x4_t x2_new = vfmaq_f32(vmulq_f32(x2, c), x1, s);
        
        // 原地写回内存
        vst1q_f32(x + i, x1_new);
        vst1q_f32(x + i + half_n, x2_new);
    }
}

void rope_f16_neon(
    fp16_t* x,
    const fp16_t* cos,
    const fp16_t* sin,
    int n
) {
    int half_n = n / 2;
    int i = 0;
    for (; i <= half_n - 8; i += 8) {
        float16x8_t x1 = vld1q_f16(x + i);
        float16x8_t x2 = vld1q_f16(x + i + half_n);
        float16x8_t c = vld1q_f16(cos + i);
        float16x8_t s = vld1q_f16(sin + i);

        float16x8_t x1_new = vfmsq_f16(vmulq_f16(x1, c), x2, s);
        float16x8_t x2_new = vfmaq_f16(vmulq_f16(x2, c), x1, s);

        vst1q_f16(x + i, x1_new);
        vst1q_f16(x + i + half_n, x2_new);
    }
    for (; i < half_n; ++i) {
        float x1 = (float)x[i];
        float x2 = (float)x[i + half_n];
        float c = (float)cos[i];
        float s = (float)sin[i];
        x[i] = (fp16_t)(x1 * c - x2 * s);
        x[i + half_n] = (fp16_t)(x2 * c + x1 * s);
    }
}

} // namespace arm_neon
} // namespace llm_engine
