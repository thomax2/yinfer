#include "neon_ops.h"
#include <arm_neon.h>
#include <cmath>

namespace llm_engine {
namespace arm_neon {

/*
    y = x * weight / sqrt((x^2)/n + eps)
    n 是 向量的长度，通常是隐藏维度的大小，比如 1024 或 4096。
    weight 是可学习的缩放参数，和输入 x 的维度相同。
    eps 是一个小常数，防止除以零，通常取 1e-6 或 1e-5
*/ 

void rmsnorm_neon(
    const float* x,
    const float* weight,
    float* y,
    int n,
    float eps
) {
    float32x4_t vsum = vdupq_n_f32(0.f);

    int i = 0;
    for(; i <= n - 4; i += 4) {
        float32x4_t vx = vld1q_f32(x + i);
        vsum = vfmaq_f32(vsum, vx, vx); // sum += x^2, vsum 是寄存器，不需要每次都访问内存
    }

    float sum = vaddvq_f32(vsum); // 将寄存器中的 4 个元素 ( vsum = [v0, v1, v2, v3] ) 相加得到总和

    // tail case 处理剩余的元素
    for(; i < n; i++)
        sum += x[i] * x[i];;

    float mean = sum/ n;
    float scale = 1.0f / std::sqrt(mean + eps);

    float32x4_t vscale = vdupq_n_f32(scale);    //  1 / sqrt((x^2)/n + eps)

    i = 0;
    for(; i <= n - 4; i += 4) {
        float32x4_t vx = vld1q_f32(x + i);
        float32x4_t vw = vld1q_f32(weight + i);

        float32x4_t vy = vmulq_f32(vmulq_f32(vx, vscale), vw);
        vst1q_f32(y + i, vy);
    }

    for(; i < n; i++) {
        y[i] = x[i] * scale * weight[i];
    }
}

void rmsnorm_f16_neon(
    const fp16_t* x,
    const fp16_t* weight,
    fp16_t* y,
    int n,
    float eps
) {
    float32x4_t sum0 = vdupq_n_f32(0.0f);
    float32x4_t sum1 = vdupq_n_f32(0.0f);

    int i = 0;
    for (; i <= n - 8; i += 8) {
        float16x8_t hx = vld1q_f16(x + i);
        float32x4_t lo = vcvt_f32_f16(vget_low_f16(hx));
        float32x4_t hi = vcvt_f32_f16(vget_high_f16(hx));
        sum0 = vfmaq_f32(sum0, lo, lo);
        sum1 = vfmaq_f32(sum1, hi, hi);
    }

    float sum = vaddvq_f32(vaddq_f32(sum0, sum1));
    for (; i < n; ++i) {
        float v = (float)x[i];
        sum += v * v;
    }

    float scale = 1.0f / std::sqrt(sum / n + eps);
    float16x8_t hscale = vdupq_n_f16((fp16_t)scale);

    i = 0;
    for (; i <= n - 8; i += 8) {
        float16x8_t hx = vld1q_f16(x + i);
        float16x8_t hw = vld1q_f16(weight + i);
        float16x8_t hy = vmulq_f16(vmulq_f16(hx, hscale), hw);
        vst1q_f16(y + i, hy);
    }
    for (; i < n; ++i) {
        y[i] = (fp16_t)((float)x[i] * scale * (float)weight[i]);
    }
}

} // namespace arm_neon
} // namespace llm_engine
