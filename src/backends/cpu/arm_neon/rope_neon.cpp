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

    // 
    const float sign_data[4] = {-1.f, 1.f, -1.f, 1.f};
    float32x4_t sign = vld1q_f32(sign_data);

    // n 必须是偶数（成对处理）
    for (int i = 0; i < n; i += 4) {
        float32x4_t vx = vld1q_f32(x + i);
        float32x4_t vcos = vld1q_f32(cos + i);
        float32x4_t vsin = vld1q_f32(sin + i);

        // [x0 x1 x2 x3] -> [x1 x0 x3 x2]
        float32x4_t vrev = vrev64q_f32(vx);

        // sign flip: [-x1, x0, -x3, x2]
        vrev = vmulq_f32(vrev, sign);

        /* 
            vmulq_f32 是逐元素乘法， vfmaq_f32 是逐元素乘法加法，res = vx * cos + vrev * sin
            
            vcos = [c0, c0, c1, c1], vsin = [s0, s0, s1, s1]
            想得到 res = [x0*c0 + (-x1)*s0, x1*c0 + x0*s0, x2*c1 + (-x3)*s1, x3*c1 + x2*s1]
            所以 vrev = [-x1, x0, -x3, x2] 需要 交换+负号
        */ 
        float32x4_t res = vfmaq_f32(vmulq_f32(vx, vcos), vrev, vsin);

        vst1q_f32(x + i, res);
    }
}

} // namespace arm_neon
} // namespace llm_engine