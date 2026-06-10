#include "neon_ops.h"
#include <arm_neon.h>
#include <cmath>

namespace llm_engine {
namespace arm_neon {

/*
工业级高精度 exp(x) SIMD 实现
原理：e^x = 2^(x * log2(e)) = 2^(n + f) = 2^n * e^(f * ln(2))
*/
inline float32x4_t exp_neon_f32(float32x4_t x) {
    // 1. 安全边界控制 (Clamping) -------------------------------------
    // 超过 88.0296f 会导致单精度浮点溢出为 Inf
    // 低于 -87.3365f 实际上等于 0 (下溢)
    x = vmaxq_f32(x, vdupq_n_f32(-87.3365f));
    x = vminq_f32(x, vdupq_n_f32(88.0296f));

    // 2. 范围归约 (Range Reduction) ----------------------------------
    /*
        x * log2(e) = n + f
        x = n * ln(2) + r, 其中 r 是残差，满足 |r| <= ln(2)/2 ≈ 0.346
    */ 
    float32x4_t log2e = vdupq_n_f32(1.44269504f);   // log2(e) ≈ 1.44269504
    float32x4_t y = vmulq_f32(x, log2e);            // y = x * log2(e)
    
    int32x4_t n = vcvtnq_s32_f32(y); 
    float32x4_t fn = vcvtq_f32_s32(n);              // 浮点数的整数 n

    // r = x - n * ln(2)
    float32x4_t ln2 = vdupq_n_f32(0.69314718f);
    float32x4_t r = vmlsq_f32(x, fn, ln2);

    // 3. 多项式逼近 (Horner's Method) -------------------------------
    // 既然 r 极小，泰勒展开现在变得极其安全且精准！
    // e^r = P(r) ≈ 1 + r + r^2/2 + r^3/6 + r^4/24 + r^5/120
    float32x4_t poly = vdupq_n_f32(0.00833333f);                         // 1/120
    poly = vfmaq_f32(vdupq_n_f32(0.04166667f), poly, r);                 // + 1/24
    poly = vfmaq_f32(vdupq_n_f32(0.16666667f), poly, r);                 // + 1/6
    poly = vfmaq_f32(vdupq_n_f32(0.5f), poly, r);                        // + 1/2
    poly = vfmaq_f32(vdupq_n_f32(1.0f), poly, r);                        // + 1
    poly = vfmaq_f32(vdupq_n_f32(1.0f), poly, r);                        // + 1 (最终逼近 e^r)

    // 4. IEEE-754 浮点数位重组 (Bit-Hack for 2^n) --------------------
    // exp_int = 2^n
    int32x4_t offset = vdupq_n_s32(127);
    int32x4_t exp_int = vshlq_n_s32(vaddq_s32(n, offset), 23);
    
    // 告诉编译器：“别管类型了，直接把这块内存当成 float 对待” (零开销指令)
    float32x4_t exp_pow2 = vreinterpretq_f32_s32(exp_int);

    // 5. 组装结果: e^r * 2^n
    return vmulq_f32(poly, exp_pow2);
}

/*
sigmoid(x) = 1 / (1 + exp(-x))
*/
inline float32x4_t sigmoid_neon(float32x4_t x) {
    float32x4_t neg = vnegq_f32(x);             // vnegq_f32：取反。把向量里的每个数都变成负数。
    float32x4_t e = exp_neon_f32(neg);
    float32x4_t one = vdupq_n_f32(1.f);         // vdupq_n_f32：复制。把向量里的每个数都设置成 1.f
    return vdivq_f32(one, vaddq_f32(one, e));
}

/*
SwiGLU: out = x * sigmoid(x) * up，都是逐元素操作
*/
void swiglu_neon(
    const float* x,
    const float* up,
    float* y,
    int n
) {
    int i = 0;

    // SIMD 处理 4 个元素
    for (; i <= n - 4; i += 4) {
        float32x4_t vx = vld1q_f32(x + i);
        float32x4_t vup = vld1q_f32(up + i);

        float32x4_t vsig = sigmoid_neon(vx);    // sigmoid(x)

        float32x4_t vy = vmulq_f32(vmulq_f32(vx, vsig), vup);   // y = x * sigmoid(x) * up

        vst1q_f32(y + i, vy);
    }

    for (; i < n; i++) {
        float s = 1.f / (1.f + std::exp(-x[i]));
        y[i] = x[i] * s * up[i];
    }
}

void swiglu_f16_neon(
    const fp16_t* gate,
    const fp16_t* up,
    fp16_t* y,
    int n
) {
    int i = 0;
    for (; i <= n - 8; i += 8) {
        // Load eight FP16 values contiguously, widen to two FP32 vectors for
        // sigmoid/exp, then narrow back to FP16. This preserves the previous
        // FP32 nonlinear math while using NEON registers for the hot loop.
        float16x8_t g16 = vld1q_f16(gate + i);
        float16x8_t u16 = vld1q_f16(up + i);

        float32x4_t g0 = vcvt_f32_f16(vget_low_f16(g16));
        float32x4_t g1 = vcvt_f32_f16(vget_high_f16(g16));
        float32x4_t u0 = vcvt_f32_f16(vget_low_f16(u16));
        float32x4_t u1 = vcvt_f32_f16(vget_high_f16(u16));

        float32x4_t y0 = vmulq_f32(vmulq_f32(g0, sigmoid_neon(g0)), u0);
        float32x4_t y1 = vmulq_f32(vmulq_f32(g1, sigmoid_neon(g1)), u1);

        vst1q_f16(y + i, vcombine_f16(vcvt_f16_f32(y0), vcvt_f16_f32(y1)));
    }

    for (; i < n; ++i) {
        float g = (float)gate[i];
        float u = (float)up[i];
        float silu = g / (1.0f + std::exp(-g));
        y[i] = (fp16_t)(silu * u);
    }
}

} // namespace arm_neon
} // namespace llm_engine
