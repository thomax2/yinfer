#include "neon_ops.h"
#include <arm_neon.h>
#include <cmath>
#include <algorithm>
#include <limits>

namespace llm_engine {
namespace arm_neon {

// 复用你在 swiglu_neon.cpp 里的 exp 实现
inline float32x4_t exp_neon_f32(float32x4_t x) {
    // Clamp
    x = vmaxq_f32(x, vdupq_n_f32(-87.3365f));
    x = vminq_f32(x, vdupq_n_f32(88.0296f));

    // Range reduction
    float32x4_t log2e = vdupq_n_f32(1.44269504f);
    float32x4_t y = vmulq_f32(x, log2e);

    int32x4_t n = vcvtnq_s32_f32(y);
    float32x4_t fn = vcvtq_f32_s32(n);

    float32x4_t ln2 = vdupq_n_f32(0.69314718f);
    float32x4_t r = vmlsq_f32(x, fn, ln2);

    // Polynomial (Horner)
    float32x4_t poly = vdupq_n_f32(0.00833333f); // 1/120
    poly = vfmaq_f32(vdupq_n_f32(0.04166667f), poly, r);
    poly = vfmaq_f32(vdupq_n_f32(0.16666667f), poly, r);
    poly = vfmaq_f32(vdupq_n_f32(0.5f), poly, r);
    poly = vfmaq_f32(vdupq_n_f32(1.0f), poly, r);
    poly = vfmaq_f32(vdupq_n_f32(1.0f), poly, r);

    // 2^n
    int32x4_t offset = vdupq_n_s32(127);
    int32x4_t exp_int = vshlq_n_s32(vaddq_s32(n, offset), 23);
    float32x4_t exp_pow2 = vreinterpretq_f32_s32(exp_int);

    return vmulq_f32(poly, exp_pow2);
}


Status softmax_neon(const Tensor& input, Tensor& output) {
    if (input.dtype != DataType::FP32 || output.dtype != DataType::FP32)
        return Status::INVALID_ARGUMENT;

    if (input.size() != output.size())
        return Status::INVALID_ARGUMENT;

    // 默认按最后一维做 softmax
    int seq_len = input.shape.back();
    int batch_size = input.size() / seq_len;

    const float* in_ptr_base = input.ptr<float>();
    float* out_ptr_base = output.ptr<float>();

    for (int b = 0; b < batch_size; ++b) {
        const float* in_ptr = in_ptr_base + b * seq_len;
        float* out_ptr = out_ptr_base + b * seq_len;

        // =========================
        // 1. 求 max（NEON）
        // =========================
        float max_val = std::numeric_limits<float>::lowest();
        float32x4_t vmax_vec = vdupq_n_f32(max_val);

        int i = 0;
        for (; i <= seq_len - 4; i += 4) {
            float32x4_t v = vld1q_f32(in_ptr + i);
            vmax_vec = vmaxq_f32(vmax_vec, v);
        }

        // float tmp[4];
        // vst1q_f32(tmp, vmax_vec);
        // for (int j = 0; j < 4; j++) {
        //     max_val = std::max(max_val, tmp[j]);
        // }
        max_val = vmaxvq_f32(vmax_vec);

        for (; i < seq_len; i++) {
            max_val = std::max(max_val, in_ptr[i]);
        }

        // =========================
        // 2. exp(x - max) + sum
        // =========================
        float sum = 0.0f;
        float32x4_t vsum_vec = vdupq_n_f32(0.0f);
        float32x4_t vmax_broadcast = vdupq_n_f32(max_val);

        i = 0;
        for (; i <= seq_len - 4; i += 4) {
            float32x4_t v = vld1q_f32(in_ptr + i);

            // x - max
            v = vsubq_f32(v, vmax_broadcast);

            // exp
            float32x4_t vexp = exp_neon_f32(v);

            // store
            vst1q_f32(out_ptr + i, vexp);

            // sum
            vsum_vec = vaddq_f32(vsum_vec, vexp);
        }

        // reduce sum vector
        float32x2_t sum2 = vadd_f32(vget_low_f32(vsum_vec), vget_high_f32(vsum_vec));
        sum += vget_lane_f32(sum2, 0) + vget_lane_f32(sum2, 1);

        // tail
        for (; i < seq_len; i++) {
            float val = std::exp(in_ptr[i] - max_val);
            out_ptr[i] = val;
            sum += val;
        }

        // =========================
        // 3. normalize
        // =========================
        float inv_sum = 1.0f / sum;
        float32x4_t vinv = vdupq_n_f32(inv_sum);

        i = 0;
        for (; i <= seq_len - 4; i += 4) {
            float32x4_t v = vld1q_f32(out_ptr + i);
            v = vmulq_f32(v, vinv);
            vst1q_f32(out_ptr + i, v);
        }

        for (; i < seq_len; i++) {
            out_ptr[i] *= inv_sum;
        }
    }

    return Status::SUCCESS;
}

Status softmax_f16_neon(const Tensor& input, Tensor& output) {
    if (input.dtype != DataType::FP16 || output.dtype != DataType::FP16)
        return Status::INVALID_ARGUMENT;
    if (input.size() != output.size())
        return Status::INVALID_ARGUMENT;

    int seq_len = input.shape.back();
    int batch_size = input.size() / seq_len;
    const fp16_t* in_base = input.ptr<fp16_t>();
    fp16_t* out_base = output.ptr<fp16_t>();

    for (int b = 0; b < batch_size; ++b) {
        const fp16_t* in = in_base + (size_t)b * seq_len;
        fp16_t* out = out_base + (size_t)b * seq_len;

        float max_val = -std::numeric_limits<float>::infinity();
        int i = 0;
        for (; i <= seq_len - 8; i += 8) {
            float16x8_t hv = vld1q_f16(in + i);
            float32x4_t lo = vcvt_f32_f16(vget_low_f16(hv));
            float32x4_t hi = vcvt_f32_f16(vget_high_f16(hv));
            max_val = std::max(max_val, vmaxvq_f32(lo));
            max_val = std::max(max_val, vmaxvq_f32(hi));
        }
        for (; i < seq_len; ++i) {
            max_val = std::max(max_val, (float)in[i]);
        }

        float sum = 0.0f;
        for (i = 0; i < seq_len; ++i) {
            float e = std::exp((float)in[i] - max_val);
            out[i] = (fp16_t)e;
            sum += e;
        }

        float inv = 1.0f / sum;
        float16x8_t hinv = vdupq_n_f16((fp16_t)inv);
        for (i = 0; i <= seq_len - 8; i += 8) {
            float16x8_t hv = vld1q_f16(out + i);
            vst1q_f16(out + i, vmulq_f16(hv, hinv));
        }
        for (; i < seq_len; ++i) {
            out[i] = (fp16_t)((float)out[i] * inv);
        }
    }

    return Status::SUCCESS;
}

} // namespace arm_neon
} // namespace llm_engine
