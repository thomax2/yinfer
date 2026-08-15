#include "backends/cpu/arm_neon/neon_ops.h"
#include "llm_engine/memory/kv_cache.h"
#include "llm_engine/memory/workspace.h"
#include "llm_engine/metrics/metrics.h"
#include "llm_engine/tensor.h"
#include <cassert>
#include <cmath>
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <string>
#include <vector>
#include <arm_neon.h>

namespace llm_engine {
namespace arm_neon {

namespace {
bool debug_numeric_enabled() {
    const char* v = std::getenv("LLM_DEBUG_NUMERIC");
    if (!v) return false;
    std::string s(v);
    return s == "1" || s == "true" || s == "TRUE" || s == "on" || s == "ON";
}

bool attention_env_flag(const char* name) {
    const char* v = std::getenv(name);
    if (!v) return false;
    std::string s(v);
    return s == "1" || s == "true" || s == "TRUE" || s == "on" || s == "ON";
}

float attention_env_float(const char* name, float default_value) {
    const char* v = std::getenv(name);
    if (!v || !*v) return default_value;
    char* end = nullptr;
    float x = std::strtof(v, &end);
    if (end == v) return default_value;
    return x;
}

void dump_f16_stats(const char* tag, int layer_id, const fp16_t* data, int n) {
    int finite_count = 0;
    int nan_count = 0;
    int inf_count = 0;
    float min_v = std::numeric_limits<float>::infinity();
    float max_v = -std::numeric_limits<float>::infinity();
    for (int i = 0; i < n; ++i) {
        float v = (float)data[i];
        if (std::isnan(v)) {
            nan_count++;
        } else if (!std::isfinite(v)) {
            inf_count++;
        } else {
            finite_count++;
            min_v = std::min(min_v, v);
            max_v = std::max(max_v, v);
        }
    }
    std::cerr << "[NUMERIC] layer=" << layer_id
              << " stage=" << tag
              << " finite=" << finite_count
              << " nan=" << nan_count
              << " inf=" << inf_count
              << " min=" << min_v
              << " max=" << max_v
              << std::endl;
}

inline const fp16_t* paged_kv_token_ptr(
    const fp16_t* pages,
    int physical_block,
    int layer_id,
    int kv_head_id,
    int offset_in_block,
    int num_layers,
    int num_kv_heads,
    int block_size,
    int head_dim
) {
    return pages +
           (((((size_t)physical_block * num_layers + layer_id) * num_kv_heads + kv_head_id) *
                 block_size +
             offset_in_block) *
            head_dim);
}

inline float reduce_f32x8(float32x4_t lo, float32x4_t hi) {
    return vaddvq_f32(vaddq_f32(lo, hi));
}

inline float dot_f16_fp32(const fp16_t* a, const fp16_t* b, int n) {
    float32x4_t acc0 = vdupq_n_f32(0.0f);
    float32x4_t acc1 = vdupq_n_f32(0.0f);

    int i = 0;
    for (; i <= n - 8; i += 8) {
        float16x8_t av = vld1q_f16(a + i);
        float16x8_t bv = vld1q_f16(b + i);
        acc0 = vfmaq_f32(acc0,
                         vcvt_f32_f16(vget_low_f16(av)),
                         vcvt_f32_f16(vget_low_f16(bv)));
        acc1 = vfmaq_f32(acc1,
                         vcvt_f32_f16(vget_high_f16(av)),
                         vcvt_f32_f16(vget_high_f16(bv)));
    }

    float sum = reduce_f32x8(acc0, acc1);
    for (; i < n; ++i) {
        sum += (float)a[i] * (float)b[i];
    }
    return sum;
}

void attention_decode_score_f16_neon(
    const fp16_t* q_group,
    const fp16_t* k_cache,
    fp16_t* score,
    int num_rep,
    int seq_len,
    int head_dim,
    float scale
) {
    for (int qh = 0; qh < num_rep; ++qh) {
        const fp16_t* q = q_group + (size_t)qh * head_dim;
        fp16_t* score_row = score + (size_t)qh * seq_len;

        int t = 0;
        for (; t <= seq_len - 4; t += 4) {
            const fp16_t* k0 = k_cache + (size_t)(t + 0) * head_dim;
            const fp16_t* k1 = k_cache + (size_t)(t + 1) * head_dim;
            const fp16_t* k2 = k_cache + (size_t)(t + 2) * head_dim;
            const fp16_t* k3 = k_cache + (size_t)(t + 3) * head_dim;

            float32x4_t a00 = vdupq_n_f32(0.0f);
            float32x4_t a01 = vdupq_n_f32(0.0f);
            float32x4_t a10 = vdupq_n_f32(0.0f);
            float32x4_t a11 = vdupq_n_f32(0.0f);
            float32x4_t a20 = vdupq_n_f32(0.0f);
            float32x4_t a21 = vdupq_n_f32(0.0f);
            float32x4_t a30 = vdupq_n_f32(0.0f);
            float32x4_t a31 = vdupq_n_f32(0.0f);

            int d = 0;
            for (; d <= head_dim - 8; d += 8) {
                float16x8_t qv = vld1q_f16(q + d);
                float32x4_t qlo = vcvt_f32_f16(vget_low_f16(qv));
                float32x4_t qhi = vcvt_f32_f16(vget_high_f16(qv));

                float16x8_t k0v = vld1q_f16(k0 + d);
                a00 = vfmaq_f32(a00, qlo, vcvt_f32_f16(vget_low_f16(k0v)));
                a01 = vfmaq_f32(a01, qhi, vcvt_f32_f16(vget_high_f16(k0v)));

                float16x8_t k1v = vld1q_f16(k1 + d);
                a10 = vfmaq_f32(a10, qlo, vcvt_f32_f16(vget_low_f16(k1v)));
                a11 = vfmaq_f32(a11, qhi, vcvt_f32_f16(vget_high_f16(k1v)));

                float16x8_t k2v = vld1q_f16(k2 + d);
                a20 = vfmaq_f32(a20, qlo, vcvt_f32_f16(vget_low_f16(k2v)));
                a21 = vfmaq_f32(a21, qhi, vcvt_f32_f16(vget_high_f16(k2v)));

                float16x8_t k3v = vld1q_f16(k3 + d);
                a30 = vfmaq_f32(a30, qlo, vcvt_f32_f16(vget_low_f16(k3v)));
                a31 = vfmaq_f32(a31, qhi, vcvt_f32_f16(vget_high_f16(k3v)));
            }

            float s0 = reduce_f32x8(a00, a01);
            float s1 = reduce_f32x8(a10, a11);
            float s2 = reduce_f32x8(a20, a21);
            float s3 = reduce_f32x8(a30, a31);
            for (; d < head_dim; ++d) {
                float qv = (float)q[d];
                s0 += qv * (float)k0[d];
                s1 += qv * (float)k1[d];
                s2 += qv * (float)k2[d];
                s3 += qv * (float)k3[d];
            }

            score_row[t + 0] = (fp16_t)(s0 * scale);
            score_row[t + 1] = (fp16_t)(s1 * scale);
            score_row[t + 2] = (fp16_t)(s2 * scale);
            score_row[t + 3] = (fp16_t)(s3 * scale);
        }

        for (; t < seq_len; ++t) {
            const fp16_t* k = k_cache + (size_t)t * head_dim;
            score_row[t] = (fp16_t)(dot_f16_fp32(q, k, head_dim) * scale);
        }
    }
}

void attention_decode_value_f16_neon(
    const fp16_t* score,
    const fp16_t* v_cache,
    fp16_t* out_group,
    int num_rep,
    int seq_len,
    int head_dim
) {
    int qh = 0;
    for (; qh + 1 < num_rep; qh += 2) {
        const fp16_t* score0 = score + (size_t)qh * seq_len;
        const fp16_t* score1 = score + (size_t)(qh + 1) * seq_len;
        fp16_t* out0 = out_group + (size_t)qh * head_dim;
        fp16_t* out1 = out_group + (size_t)(qh + 1) * head_dim;

        int d = 0;
        for (; d <= head_dim - 16; d += 16) {
            float32x4_t a00 = vdupq_n_f32(0.0f);
            float32x4_t a01 = vdupq_n_f32(0.0f);
            float32x4_t a02 = vdupq_n_f32(0.0f);
            float32x4_t a03 = vdupq_n_f32(0.0f);
            float32x4_t a10 = vdupq_n_f32(0.0f);
            float32x4_t a11 = vdupq_n_f32(0.0f);
            float32x4_t a12 = vdupq_n_f32(0.0f);
            float32x4_t a13 = vdupq_n_f32(0.0f);

            for (int t = 0; t < seq_len; ++t) {
                float32x4_t s0 = vdupq_n_f32((float)score0[t]);
                float32x4_t s1 = vdupq_n_f32((float)score1[t]);
                const fp16_t* v = v_cache + (size_t)t * head_dim + d;
                float16x8_t v0 = vld1q_f16(v);
                float16x8_t v1 = vld1q_f16(v + 8);
                float32x4_t vlo0 = vcvt_f32_f16(vget_low_f16(v0));
                float32x4_t vhi0 = vcvt_f32_f16(vget_high_f16(v0));
                float32x4_t vlo1 = vcvt_f32_f16(vget_low_f16(v1));
                float32x4_t vhi1 = vcvt_f32_f16(vget_high_f16(v1));
                a00 = vfmaq_f32(a00, vlo0, s0);
                a01 = vfmaq_f32(a01, vhi0, s0);
                a02 = vfmaq_f32(a02, vlo1, s0);
                a03 = vfmaq_f32(a03, vhi1, s0);
                a10 = vfmaq_f32(a10, vlo0, s1);
                a11 = vfmaq_f32(a11, vhi0, s1);
                a12 = vfmaq_f32(a12, vlo1, s1);
                a13 = vfmaq_f32(a13, vhi1, s1);
            }

            vst1q_f16(out0 + d, vcombine_f16(vcvt_f16_f32(a00), vcvt_f16_f32(a01)));
            vst1q_f16(out0 + d + 8, vcombine_f16(vcvt_f16_f32(a02), vcvt_f16_f32(a03)));
            vst1q_f16(out1 + d, vcombine_f16(vcvt_f16_f32(a10), vcvt_f16_f32(a11)));
            vst1q_f16(out1 + d + 8, vcombine_f16(vcvt_f16_f32(a12), vcvt_f16_f32(a13)));
        }

        for (; d <= head_dim - 8; d += 8) {
            float32x4_t a00 = vdupq_n_f32(0.0f);
            float32x4_t a01 = vdupq_n_f32(0.0f);
            float32x4_t a10 = vdupq_n_f32(0.0f);
            float32x4_t a11 = vdupq_n_f32(0.0f);
            for (int t = 0; t < seq_len; ++t) {
                float32x4_t s0 = vdupq_n_f32((float)score0[t]);
                float32x4_t s1 = vdupq_n_f32((float)score1[t]);
                const fp16_t* v = v_cache + (size_t)t * head_dim + d;
                float16x8_t vv = vld1q_f16(v);
                float32x4_t vlo = vcvt_f32_f16(vget_low_f16(vv));
                float32x4_t vhi = vcvt_f32_f16(vget_high_f16(vv));
                a00 = vfmaq_f32(a00, vlo, s0);
                a01 = vfmaq_f32(a01, vhi, s0);
                a10 = vfmaq_f32(a10, vlo, s1);
                a11 = vfmaq_f32(a11, vhi, s1);
            }
            vst1q_f16(out0 + d, vcombine_f16(vcvt_f16_f32(a00), vcvt_f16_f32(a01)));
            vst1q_f16(out1 + d, vcombine_f16(vcvt_f16_f32(a10), vcvt_f16_f32(a11)));
        }

        for (; d < head_dim; ++d) {
            float sum0 = 0.0f;
            float sum1 = 0.0f;
            for (int t = 0; t < seq_len; ++t) {
                float vv = (float)v_cache[(size_t)t * head_dim + d];
                sum0 += (float)score0[t] * vv;
                sum1 += (float)score1[t] * vv;
            }
            out0[d] = (fp16_t)sum0;
            out1[d] = (fp16_t)sum1;
        }
    }

    for (; qh < num_rep; ++qh) {
        const fp16_t* score_row = score + (size_t)qh * seq_len;
        fp16_t* out = out_group + (size_t)qh * head_dim;

        int d = 0;
        for (; d <= head_dim - 16; d += 16) {
            float32x4_t acc0 = vdupq_n_f32(0.0f);
            float32x4_t acc1 = vdupq_n_f32(0.0f);
            float32x4_t acc2 = vdupq_n_f32(0.0f);
            float32x4_t acc3 = vdupq_n_f32(0.0f);

            for (int t = 0; t < seq_len; ++t) {
                float32x4_t sv = vdupq_n_f32((float)score_row[t]);
                const fp16_t* v = v_cache + (size_t)t * head_dim + d;
                float16x8_t v0 = vld1q_f16(v);
                float16x8_t v1 = vld1q_f16(v + 8);
                acc0 = vfmaq_f32(acc0, vcvt_f32_f16(vget_low_f16(v0)), sv);
                acc1 = vfmaq_f32(acc1, vcvt_f32_f16(vget_high_f16(v0)), sv);
                acc2 = vfmaq_f32(acc2, vcvt_f32_f16(vget_low_f16(v1)), sv);
                acc3 = vfmaq_f32(acc3, vcvt_f32_f16(vget_high_f16(v1)), sv);
            }

            vst1q_f16(out + d, vcombine_f16(vcvt_f16_f32(acc0), vcvt_f16_f32(acc1)));
            vst1q_f16(out + d + 8, vcombine_f16(vcvt_f16_f32(acc2), vcvt_f16_f32(acc3)));
        }

        for (; d <= head_dim - 8; d += 8) {
            float32x4_t acc0 = vdupq_n_f32(0.0f);
            float32x4_t acc1 = vdupq_n_f32(0.0f);
            for (int t = 0; t < seq_len; ++t) {
                float32x4_t sv = vdupq_n_f32((float)score_row[t]);
                const fp16_t* v = v_cache + (size_t)t * head_dim + d;
                float16x8_t vv = vld1q_f16(v);
                acc0 = vfmaq_f32(acc0, vcvt_f32_f16(vget_low_f16(vv)), sv);
                acc1 = vfmaq_f32(acc1, vcvt_f32_f16(vget_high_f16(vv)), sv);
            }
            vst1q_f16(out + d, vcombine_f16(vcvt_f16_f32(acc0), vcvt_f16_f32(acc1)));
        }

        for (; d < head_dim; ++d) {
            float sum = 0.0f;
            for (int t = 0; t < seq_len; ++t) {
                sum += (float)score_row[t] * (float)v_cache[(size_t)t * head_dim + d];
            }
            out[d] = (fp16_t)sum;
        }
    }
}

Status attention_decode_score_paged_f16_neon(
    const fp16_t* q_group,
    fp16_t* score,
    int num_rep,
    int seq_len,
    int head_dim,
    float scale,
    const fp16_t* k_pages,
    const int* block_table,
    int block_table_size,
    int block_size,
    int layer_id,
    int kv_head_id,
    int num_layers,
    int num_kv_heads,
    int num_physical_blocks
) {
    if (!q_group || !score || !k_pages || !block_table ||
        num_rep <= 0 || seq_len <= 0 || head_dim <= 0 ||
        block_size <= 0 || block_table_size <= 0) {
        return Status::INVALID_ARGUMENT;
    }

    int logical_blocks_needed = (seq_len + block_size - 1) / block_size;
    if (logical_blocks_needed > block_table_size) {
        return Status::INVALID_ARGUMENT;
    }

    for (int qh = 0; qh < num_rep; ++qh) {
        const fp16_t* q = q_group + (size_t)qh * head_dim;
        fp16_t* score_row = score + (size_t)qh * seq_len;

        for (int logical_block = 0; logical_block < logical_blocks_needed; ++logical_block) {
            int physical_block = block_table[logical_block];
            if (physical_block < 0 || physical_block >= num_physical_blocks) {
                return Status::INVALID_ARGUMENT;
            }

            int token_begin = logical_block * block_size;
            int valid_tokens = std::min(block_size, seq_len - token_begin);
            const fp16_t* k_block = paged_kv_token_ptr(
                k_pages,
                physical_block,
                layer_id,
                kv_head_id,
                0,
                num_layers,
                num_kv_heads,
                block_size,
                head_dim);

            int offset = 0;
            for (; offset <= valid_tokens - 4; offset += 4) {
                const fp16_t* k0 = k_block + (size_t)(offset + 0) * head_dim;
                const fp16_t* k1 = k_block + (size_t)(offset + 1) * head_dim;
                const fp16_t* k2 = k_block + (size_t)(offset + 2) * head_dim;
                const fp16_t* k3 = k_block + (size_t)(offset + 3) * head_dim;

                float32x4_t a00 = vdupq_n_f32(0.0f);
                float32x4_t a01 = vdupq_n_f32(0.0f);
                float32x4_t a10 = vdupq_n_f32(0.0f);
                float32x4_t a11 = vdupq_n_f32(0.0f);
                float32x4_t a20 = vdupq_n_f32(0.0f);
                float32x4_t a21 = vdupq_n_f32(0.0f);
                float32x4_t a30 = vdupq_n_f32(0.0f);
                float32x4_t a31 = vdupq_n_f32(0.0f);

                int d = 0;
                for (; d <= head_dim - 8; d += 8) {
                    float16x8_t qv = vld1q_f16(q + d);
                    float32x4_t qlo = vcvt_f32_f16(vget_low_f16(qv));
                    float32x4_t qhi = vcvt_f32_f16(vget_high_f16(qv));

                    float16x8_t k0v = vld1q_f16(k0 + d);
                    a00 = vfmaq_f32(a00, qlo, vcvt_f32_f16(vget_low_f16(k0v)));
                    a01 = vfmaq_f32(a01, qhi, vcvt_f32_f16(vget_high_f16(k0v)));

                    float16x8_t k1v = vld1q_f16(k1 + d);
                    a10 = vfmaq_f32(a10, qlo, vcvt_f32_f16(vget_low_f16(k1v)));
                    a11 = vfmaq_f32(a11, qhi, vcvt_f32_f16(vget_high_f16(k1v)));

                    float16x8_t k2v = vld1q_f16(k2 + d);
                    a20 = vfmaq_f32(a20, qlo, vcvt_f32_f16(vget_low_f16(k2v)));
                    a21 = vfmaq_f32(a21, qhi, vcvt_f32_f16(vget_high_f16(k2v)));

                    float16x8_t k3v = vld1q_f16(k3 + d);
                    a30 = vfmaq_f32(a30, qlo, vcvt_f32_f16(vget_low_f16(k3v)));
                    a31 = vfmaq_f32(a31, qhi, vcvt_f32_f16(vget_high_f16(k3v)));
                }

                float s0 = reduce_f32x8(a00, a01);
                float s1 = reduce_f32x8(a10, a11);
                float s2 = reduce_f32x8(a20, a21);
                float s3 = reduce_f32x8(a30, a31);
                for (; d < head_dim; ++d) {
                    float qv = (float)q[d];
                    s0 += qv * (float)k0[d];
                    s1 += qv * (float)k1[d];
                    s2 += qv * (float)k2[d];
                    s3 += qv * (float)k3[d];
                }

                int t = token_begin + offset;
                score_row[t + 0] = (fp16_t)(s0 * scale);
                score_row[t + 1] = (fp16_t)(s1 * scale);
                score_row[t + 2] = (fp16_t)(s2 * scale);
                score_row[t + 3] = (fp16_t)(s3 * scale);
            }

            for (; offset < valid_tokens; ++offset) {
                const fp16_t* k = k_block + (size_t)offset * head_dim;
                score_row[token_begin + offset] =
                    (fp16_t)(dot_f16_fp32(q, k, head_dim) * scale);
            }
        }
    }

    return Status::SUCCESS;
}

Status attention_decode_value_paged_f16_neon(
    const fp16_t* score,
    fp16_t* out_group,
    int num_rep,
    int seq_len,
    int head_dim,
    const fp16_t* v_pages,
    const int* block_table,
    int block_table_size,
    int block_size,
    int layer_id,
    int kv_head_id,
    int num_layers,
    int num_kv_heads,
    int num_physical_blocks
) {
    if (!score || !out_group || !v_pages || !block_table ||
        num_rep <= 0 || seq_len <= 0 || head_dim <= 0 ||
        block_size <= 0 || block_table_size <= 0) {
        return Status::INVALID_ARGUMENT;
    }

    int logical_blocks_needed = (seq_len + block_size - 1) / block_size;
    if (logical_blocks_needed > block_table_size) {
        return Status::INVALID_ARGUMENT;
    }

    for (int qh = 0; qh < num_rep; ++qh) {
        const fp16_t* score_row = score + (size_t)qh * seq_len;
        fp16_t* out = out_group + (size_t)qh * head_dim;

        int d = 0;
        for (; d <= head_dim - 16; d += 16) {
            float32x4_t acc0 = vdupq_n_f32(0.0f);
            float32x4_t acc1 = vdupq_n_f32(0.0f);
            float32x4_t acc2 = vdupq_n_f32(0.0f);
            float32x4_t acc3 = vdupq_n_f32(0.0f);

            for (int logical_block = 0; logical_block < logical_blocks_needed; ++logical_block) {
                int physical_block = block_table[logical_block];
                if (physical_block < 0 || physical_block >= num_physical_blocks) {
                    return Status::INVALID_ARGUMENT;
                }
                int token_begin = logical_block * block_size;
                int valid_tokens = std::min(block_size, seq_len - token_begin);
                const fp16_t* v_block = paged_kv_token_ptr(
                    v_pages,
                    physical_block,
                    layer_id,
                    kv_head_id,
                    0,
                    num_layers,
                    num_kv_heads,
                    block_size,
                    head_dim);

                for (int offset = 0; offset < valid_tokens; ++offset) {
                    float32x4_t sv = vdupq_n_f32((float)score_row[token_begin + offset]);
                    const fp16_t* v = v_block + (size_t)offset * head_dim + d;
                    float16x8_t v0 = vld1q_f16(v);
                    float16x8_t v1 = vld1q_f16(v + 8);
                    acc0 = vfmaq_f32(acc0, vcvt_f32_f16(vget_low_f16(v0)), sv);
                    acc1 = vfmaq_f32(acc1, vcvt_f32_f16(vget_high_f16(v0)), sv);
                    acc2 = vfmaq_f32(acc2, vcvt_f32_f16(vget_low_f16(v1)), sv);
                    acc3 = vfmaq_f32(acc3, vcvt_f32_f16(vget_high_f16(v1)), sv);
                }
            }

            vst1q_f16(out + d, vcombine_f16(vcvt_f16_f32(acc0), vcvt_f16_f32(acc1)));
            vst1q_f16(out + d + 8, vcombine_f16(vcvt_f16_f32(acc2), vcvt_f16_f32(acc3)));
        }

        for (; d <= head_dim - 8; d += 8) {
            float32x4_t acc0 = vdupq_n_f32(0.0f);
            float32x4_t acc1 = vdupq_n_f32(0.0f);

            for (int logical_block = 0; logical_block < logical_blocks_needed; ++logical_block) {
                int physical_block = block_table[logical_block];
                if (physical_block < 0 || physical_block >= num_physical_blocks) {
                    return Status::INVALID_ARGUMENT;
                }
                int token_begin = logical_block * block_size;
                int valid_tokens = std::min(block_size, seq_len - token_begin);
                const fp16_t* v_block = paged_kv_token_ptr(
                    v_pages,
                    physical_block,
                    layer_id,
                    kv_head_id,
                    0,
                    num_layers,
                    num_kv_heads,
                    block_size,
                    head_dim);

                for (int offset = 0; offset < valid_tokens; ++offset) {
                    float32x4_t sv = vdupq_n_f32((float)score_row[token_begin + offset]);
                    const fp16_t* v = v_block + (size_t)offset * head_dim + d;
                    float16x8_t vv = vld1q_f16(v);
                    acc0 = vfmaq_f32(acc0, vcvt_f32_f16(vget_low_f16(vv)), sv);
                    acc1 = vfmaq_f32(acc1, vcvt_f32_f16(vget_high_f16(vv)), sv);
                }
            }

            vst1q_f16(out + d, vcombine_f16(vcvt_f16_f32(acc0), vcvt_f16_f32(acc1)));
        }

        for (; d < head_dim; ++d) {
            float sum = 0.0f;
            for (int logical_block = 0; logical_block < logical_blocks_needed; ++logical_block) {
                int physical_block = block_table[logical_block];
                if (physical_block < 0 || physical_block >= num_physical_blocks) {
                    return Status::INVALID_ARGUMENT;
                }
                int token_begin = logical_block * block_size;
                int valid_tokens = std::min(block_size, seq_len - token_begin);
                const fp16_t* v_block = paged_kv_token_ptr(
                    v_pages,
                    physical_block,
                    layer_id,
                    kv_head_id,
                    0,
                    num_layers,
                    num_kv_heads,
                    block_size,
                    head_dim);
                for (int offset = 0; offset < valid_tokens; ++offset) {
                    sum += (float)score_row[token_begin + offset] *
                           (float)v_block[(size_t)offset * head_dim + d];
                }
            }
            out[d] = (fp16_t)sum;
        }
    }

    return Status::SUCCESS;
}

// 4 个 Query × 16 个 Key 的 QK^T 小块。K 已由当前物理页转置成
// [head_dim, 16]，因此每个 d 都能连续读取 16 个 Key。累加始终使用 FP32。
inline void attention_prefill_qk_tile_4x16(
    const fp16_t* q_tile,
    int q_row_stride,
    int active_rows,
    const fp16_t* k_tile,
    int head_dim,
    float scale,
    fp16_t* score,
    int score_row_stride,
    int query_base,
    int key_base,
    int valid_keys,
    int query_position_base
) {
    float32x4_t acc[4][4];
    for (int r = 0; r < 4; ++r) {
        for (int v = 0; v < 4; ++v) {
            acc[r][v] = vdupq_n_f32(0.0f);
        }
    }

    for (int d = 0; d < head_dim; ++d) {
        const fp16_t* packed_k = k_tile + (size_t)d * 16;
        const float16x8_t hk0 = vld1q_f16(packed_k);
        const float16x8_t hk1 = vld1q_f16(packed_k + 8);
        const float32x4_t k0 = vcvt_f32_f16(vget_low_f16(hk0));
        const float32x4_t k1 = vcvt_f32_f16(vget_high_f16(hk0));
        const float32x4_t k2 = vcvt_f32_f16(vget_low_f16(hk1));
        const float32x4_t k3 = vcvt_f32_f16(vget_high_f16(hk1));

        for (int r = 0; r < active_rows; ++r) {
            const float32x4_t qv = vdupq_n_f32(
                (float)q_tile[(size_t)r * q_row_stride + d]);
            acc[r][0] = vfmaq_f32(acc[r][0], k0, qv);
            acc[r][1] = vfmaq_f32(acc[r][1], k1, qv);
            acc[r][2] = vfmaq_f32(acc[r][2], k2, qv);
            acc[r][3] = vfmaq_f32(acc[r][3], k3, qv);
        }
    }

    const float negative_infinity = -std::numeric_limits<float>::infinity();
    for (int r = 0; r < active_rows; ++r) {
        alignas(16) float values[16];
        for (int v = 0; v < 4; ++v) {
            vst1q_f32(values + v * 4, vmulq_n_f32(acc[r][v], scale));
        }
        fp16_t* score_row = score + (size_t)(query_base + r) * score_row_stride;
        const int query_position = query_position_base + query_base + r;
        for (int lane = 0; lane < valid_keys; ++lane) {
            const int key_position = key_base + lane;
            // Chunk 内未来位置直接写 -inf，Softmax 后自然变为 0。
            score_row[key_position] = (fp16_t)(
                key_position <= query_position ? values[lane] : negative_infinity);
        }
    }
}

// Online Softmax 使用的 4 Query × 16 Key QK^T 微内核。
//
// 与 Dense-Score 版本不同，这里把 score 保持为 FP32，避免在每个 KV Tile
// 之间反复做 FP16 舍入。因果边界由调用者按 Query 行裁剪 valid_keys；这样
// 同一个 score Tile 可以紧接着参与 max / exp / P×V，不必落到完整 Score 矩阵。
inline void attention_prefill_qk_tile_4x16_f32(
    const fp16_t* q_tile,
    int q_row_stride,
    int active_rows,
    const fp16_t* k_tile,
    int head_dim,
    float scale,
    float* score_tile
) {
    float32x4_t acc[4][4];
    for (int r = 0; r < 4; ++r) {
        for (int v = 0; v < 4; ++v) {
            acc[r][v] = vdupq_n_f32(0.0f);
        }
    }

    for (int d = 0; d < head_dim; ++d) {
        const fp16_t* packed_k = k_tile + (size_t)d * 16;
        const float16x8_t hk0 = vld1q_f16(packed_k);
        const float16x8_t hk1 = vld1q_f16(packed_k + 8);
        const float32x4_t k0 = vcvt_f32_f16(vget_low_f16(hk0));
        const float32x4_t k1 = vcvt_f32_f16(vget_high_f16(hk0));
        const float32x4_t k2 = vcvt_f32_f16(vget_low_f16(hk1));
        const float32x4_t k3 = vcvt_f32_f16(vget_high_f16(hk1));

        for (int r = 0; r < active_rows; ++r) {
            const float32x4_t qv = vdupq_n_f32(
                (float)q_tile[(size_t)r * q_row_stride + d]);
            acc[r][0] = vfmaq_f32(acc[r][0], k0, qv);
            acc[r][1] = vfmaq_f32(acc[r][1], k1, qv);
            acc[r][2] = vfmaq_f32(acc[r][2], k2, qv);
            acc[r][3] = vfmaq_f32(acc[r][3], k3, qv);
        }
    }

    for (int r = 0; r < active_rows; ++r) {
        float* row = score_tile + (size_t)r * 16;
        for (int v = 0; v < 4; ++v) {
            vst1q_f32(row + v * 4, vmulq_n_f32(acc[r][v], scale));
        }
    }
}

Status attention_prefill_paged_online_f16_neon(
    const fp16_t* q_chunk,
    fp16_t* out_chunk,
    int query_rows,
    int start_position,
    int q_row_stride,
    int num_rep,
    int head_dim,
    float scale,
    const fp16_t* k_pages,
    const fp16_t* v_pages,
    const int* block_table,
    int block_table_size,
    int block_size,
    int layer_id,
    int kv_head_id,
    int num_layers,
    int num_kv_heads,
    int num_physical_blocks,
    fp16_t* k_tile_workspace,
    size_t k_tile_elements,
    float* output_acc_workspace,
    size_t output_acc_elements
) {
    if (!q_chunk || !out_chunk || !k_pages || !v_pages || !block_table ||
        !k_tile_workspace || !output_acc_workspace || query_rows <= 1 ||
        start_position < 0 || q_row_stride < num_rep * head_dim ||
        num_rep <= 0 || head_dim <= 0 || block_size <= 0 ||
        block_table_size <= 0 || layer_id < 0 || layer_id >= num_layers ||
        kv_head_id < 0 || kv_head_id >= num_kv_heads ||
        num_physical_blocks <= 0) {
        return Status::INVALID_ARGUMENT;
    }
    if (k_tile_elements < (size_t)head_dim * 16 ||
        output_acc_elements < (size_t)4 * head_dim) {
        return Status::OUT_OF_MEMORY;
    }

    const int seq_len = start_position + query_rows;
    const int logical_blocks_needed = (seq_len + block_size - 1) / block_size;
    if (logical_blocks_needed > block_table_size) {
        return Status::INVALID_ARGUMENT;
    }

    // 工作集固定为一个 Query Tile：4×16 个 FP32 score、4 组 m/l，外加
    // Workspace 中的 [4, head_dim] FP32 O。历史越长，只会增加扫描次数，
    // 不会增加临时内存。
    alignas(64) float score_tile[4 * 16];
    alignas(16) float running_max[4];
    alignas(16) float running_sum[4];
    alignas(64) float probabilities[4 * 16];

    for (int qh = 0; qh < num_rep; ++qh) {
        const fp16_t* q_head = q_chunk + (size_t)qh * head_dim;
        fp16_t* out_head = out_chunk + (size_t)qh * head_dim;

        for (int query_base = 0; query_base < query_rows; query_base += 4) {
            const int active_rows = std::min(4, query_rows - query_base);
            for (int r = 0; r < active_rows; ++r) {
                running_max[r] = -std::numeric_limits<float>::infinity();
                running_sum[r] = 0.0f;
                std::fill_n(
                    output_acc_workspace + (size_t)r * head_dim,
                    head_dim,
                    0.0f);
            }

            const fp16_t* q_tile =
                q_head + (size_t)query_base * q_row_stride;
            const int last_query_position =
                start_position + query_base + active_rows - 1;

            // 按逻辑页顺序查 Block Table。物理页可以完全不连续，但每次只把
            // 当前 16 个 K 转置到小 Tile；V 仍从对应物理页原位读取。
            for (int logical_block = 0;
                 logical_block < logical_blocks_needed;
                 ++logical_block) {
                const int physical_block = block_table[logical_block];
                if (physical_block < 0 || physical_block >= num_physical_blocks) {
                    return Status::INVALID_ARGUMENT;
                }

                const int block_token_begin = logical_block * block_size;
                if (block_token_begin > last_query_position) {
                    break;  // 整个物理页都位于当前 Query Tile 的因果边界之后。
                }
                const int valid_block_tokens =
                    std::min(block_size, seq_len - block_token_begin);
                const fp16_t* k_block = paged_kv_token_ptr(
                    k_pages, physical_block, layer_id, kv_head_id, 0,
                    num_layers, num_kv_heads, block_size, head_dim);
                const fp16_t* v_block = paged_kv_token_ptr(
                    v_pages, physical_block, layer_id, kv_head_id, 0,
                    num_layers, num_kv_heads, block_size, head_dim);

                for (int block_offset = 0;
                     block_offset < valid_block_tokens;
                     block_offset += 16) {
                    const int key_base = block_token_begin + block_offset;
                    if (key_base > last_query_position) break;
                    const int valid_keys =
                        std::min(16, valid_block_tokens - block_offset);

                    // Paged K 的原布局是 [token, head_dim]。QK^T 内层需要对同一
                    // d 连续读取 16 个 Key，因此先转置为 [head_dim, 16]。
                    for (int d = 0; d < head_dim; ++d) {
                        fp16_t* packed_k =
                            k_tile_workspace + (size_t)d * 16;
                        int lane = 0;
                        for (; lane < valid_keys; ++lane) {
                            packed_k[lane] = k_block[
                                (size_t)(block_offset + lane) * head_dim + d];
                        }
                        for (; lane < 16; ++lane) {
                            packed_k[lane] = (fp16_t)0;
                        }
                    }

                    attention_prefill_qk_tile_4x16_f32(
                        q_tile, q_row_stride, active_rows, k_tile_workspace,
                        head_dim, scale, score_tile);

                    int causal_keys_per_row[4] = {0, 0, 0, 0};
                    for (int r = 0; r < active_rows; ++r) {
                        const int query_position =
                            start_position + query_base + r;
                        const int causal_keys = std::min(
                            valid_keys, query_position - key_base + 1);
                        if (causal_keys <= 0) continue;
                        causal_keys_per_row[r] = causal_keys;

                        const float* score_row =
                            score_tile + (size_t)r * 16;
                        float tile_max = score_row[0];
                        for (int lane = 1; lane < causal_keys; ++lane) {
                            tile_max = std::max(tile_max, score_row[lane]);
                        }

                        // Online Softmax 合并公式：
                        //   m_new = max(m_old, max(score_tile))
                        //   alpha = exp(m_old - m_new)
                        //   l_new = alpha*l_old + sum(exp(score-m_new))
                        //   O_new = alpha*O_old + exp(score-m_new) * V
                        // 当最大值被新 Tile 刷新时，旧的 l 和 O 必须一起缩放。
                        const float new_max =
                            std::max(running_max[r], tile_max);
                        const float old_scale = std::isfinite(running_max[r])
                            ? std::exp(running_max[r] - new_max)
                            : 0.0f;
                        float* out_acc =
                            output_acc_workspace + (size_t)r * head_dim;
                        int d = 0;
                        const float32x4_t old_scale_v =
                            vdupq_n_f32(old_scale);
                        for (; d <= head_dim - 4; d += 4) {
                            vst1q_f32(
                                out_acc + d,
                                vmulq_f32(vld1q_f32(out_acc + d), old_scale_v));
                        }
                        for (; d < head_dim; ++d) out_acc[d] *= old_scale;

                        float tile_sum = 0.0f;
                        float* probability_row =
                            probabilities + (size_t)r * 16;
                        for (int lane = 0; lane < causal_keys; ++lane) {
                            const float p = std::exp(score_row[lane] - new_max);
                            probability_row[lane] = p;
                            tile_sum += p;
                        }

                        running_sum[r] =
                            old_scale * running_sum[r] + tile_sum;
                        running_max[r] = new_max;
                    }

                    // 先算完 4 行 probability，再让同一次 V Load 同时服务所有
                    // Query 行。这样 Multi-Query Prefill 不会像逐 Query Decode 那样
                    // 为每一行重新读取整块 V；每行不同的 causal_keys 负责跳过未来 lane。
                    for (int lane = 0; lane < valid_keys; ++lane) {
                        const fp16_t* value = v_block +
                            (size_t)(block_offset + lane) * head_dim;
                        int value_d = 0;
                        for (; value_d <= head_dim - 8; value_d += 8) {
                            const float16x8_t hv = vld1q_f16(value + value_d);
                            const float32x4_t value_lo =
                                vcvt_f32_f16(vget_low_f16(hv));
                            const float32x4_t value_hi =
                                vcvt_f32_f16(vget_high_f16(hv));
                            for (int r = 0; r < active_rows; ++r) {
                                if (lane >= causal_keys_per_row[r]) continue;
                                const float p = probabilities[(size_t)r * 16 + lane];
                                float* out_acc = output_acc_workspace +
                                    (size_t)r * head_dim + value_d;
                                vst1q_f32(
                                    out_acc,
                                    vfmaq_n_f32(vld1q_f32(out_acc), value_lo, p));
                                vst1q_f32(
                                    out_acc + 4,
                                    vfmaq_n_f32(
                                        vld1q_f32(out_acc + 4), value_hi, p));
                            }
                        }
                        for (; value_d < head_dim; ++value_d) {
                            const float scalar_value = (float)value[value_d];
                            for (int r = 0; r < active_rows; ++r) {
                                if (lane >= causal_keys_per_row[r]) continue;
                                output_acc_workspace[(size_t)r * head_dim + value_d] +=
                                    probabilities[(size_t)r * 16 + lane] * scalar_value;
                            }
                        }
                    }
                }
            }

            // 所有可见 KV Tile 扫描完毕后只做一次归一化并写回 FP16。
            for (int r = 0; r < active_rows; ++r) {
                if (!(running_sum[r] > 0.0f) ||
                    !std::isfinite(running_sum[r])) {
                    return Status::INVALID_ARGUMENT;
                }
                const float inverse_sum = 1.0f / running_sum[r];
                const float* out_acc =
                    output_acc_workspace + (size_t)r * head_dim;
                fp16_t* output = out_head +
                    (size_t)(query_base + r) * q_row_stride;
                int d = 0;
                for (; d <= head_dim - 8; d += 8) {
                    const float32x4_t lo = vmulq_n_f32(
                        vld1q_f32(out_acc + d), inverse_sum);
                    const float32x4_t hi = vmulq_n_f32(
                        vld1q_f32(out_acc + d + 4), inverse_sum);
                    vst1q_f16(
                        output + d,
                        vcombine_f16(vcvt_f16_f32(lo), vcvt_f16_f32(hi)));
                }
                for (; d < head_dim; ++d) {
                    output[d] = (fp16_t)(out_acc[d] * inverse_sum);
                }
            }
        }
    }

    return Status::SUCCESS;
}

Status attention_prefill_paged_f16_neon(
    const fp16_t* q_chunk,
    fp16_t* out_chunk,
    int query_rows,
    int start_position,
    int q_row_stride,
    int num_rep,
    int head_dim,
    float scale,
    const fp16_t* k_pages,
    const fp16_t* v_pages,
    const int* block_table,
    int block_table_size,
    int block_size,
    int layer_id,
    int kv_head_id,
    int num_layers,
    int num_kv_heads,
    int num_physical_blocks,
    fp16_t* score,
    size_t score_elements,
    fp16_t* k_tile_workspace,
    size_t k_tile_elements
) {
    if (!q_chunk || !out_chunk || !k_pages || !v_pages || !block_table ||
        !score || !k_tile_workspace || query_rows <= 1 || start_position < 0 ||
        q_row_stride < num_rep * head_dim || num_rep <= 0 || head_dim <= 0 ||
        block_size <= 0 || block_table_size <= 0 || num_layers <= layer_id ||
        num_kv_heads <= kv_head_id || num_physical_blocks <= 0) {
        return Status::INVALID_ARGUMENT;
    }

    const int seq_len = start_position + query_rows;
    const size_t required_score =
        (size_t)query_rows * num_rep * (size_t)seq_len;
    if (score_elements < required_score ||
        k_tile_elements < (size_t)head_dim * 16) {
        return Status::OUT_OF_MEMORY;
    }
    const int logical_blocks_needed = (seq_len + block_size - 1) / block_size;
    if (logical_blocks_needed > block_table_size) {
        return Status::INVALID_ARGUMENT;
    }

    // Score 的逻辑布局为 [query, q_head_in_group, seq_len]。按物理页读取 K，
    // 每次最多取 16 个 token，并转置为 [head_dim, 16] 后服务所有 Query Tile。
    for (int logical_block = 0; logical_block < logical_blocks_needed; ++logical_block) {
        const int physical_block = block_table[logical_block];
        if (physical_block < 0 || physical_block >= num_physical_blocks) {
            return Status::INVALID_ARGUMENT;
        }
        const int block_token_begin = logical_block * block_size;
        const int valid_block_tokens = std::min(block_size, seq_len - block_token_begin);
        const fp16_t* k_block = paged_kv_token_ptr(
            k_pages, physical_block, layer_id, kv_head_id, 0,
            num_layers, num_kv_heads, block_size, head_dim);

        for (int block_offset = 0; block_offset < valid_block_tokens;
             block_offset += 16) {
            const int valid_keys = std::min(16, valid_block_tokens - block_offset);
            for (int d = 0; d < head_dim; ++d) {
                fp16_t* packed_k = k_tile_workspace + (size_t)d * 16;
                int lane = 0;
                for (; lane < valid_keys; ++lane) {
                    packed_k[lane] =
                        k_block[(size_t)(block_offset + lane) * head_dim + d];
                }
                for (; lane < 16; ++lane) packed_k[lane] = (fp16_t)0;
            }

            const int key_base = block_token_begin + block_offset;
            for (int qh = 0; qh < num_rep; ++qh) {
                const fp16_t* q_head = q_chunk + (size_t)qh * head_dim;
                fp16_t* score_head = score + (size_t)qh * seq_len;
                for (int query_base = 0; query_base < query_rows; query_base += 4) {
                    const int active_rows = std::min(4, query_rows - query_base);
                    attention_prefill_qk_tile_4x16(
                        q_head + (size_t)query_base * q_row_stride,
                        q_row_stride, active_rows, k_tile_workspace, head_dim,
                        scale, score_head, num_rep * seq_len, query_base,
                        key_base, valid_keys, start_position);
                }
            }
        }
    }

    Status status = softmax_f16_inplace_neon(
        score, query_rows * num_rep, seq_len);
    if (status != Status::SUCCESS) return status;

    // Dense-Score 第一阶段：P×V 仍按 4 个 Query 为一块，但同一 V Load 同时
    // 服务 4 行，避免 Prefill 每一行分别重扫 V Cache。
    for (int qh = 0; qh < num_rep; ++qh) {
        fp16_t* out_head = out_chunk + (size_t)qh * head_dim;
        for (int query_base = 0; query_base < query_rows; query_base += 4) {
            const int active_rows = std::min(4, query_rows - query_base);
            int d = 0;
            for (; d <= head_dim - 8; d += 8) {
                float32x4_t acc_lo[4];
                float32x4_t acc_hi[4];
                for (int r = 0; r < 4; ++r) {
                    acc_lo[r] = vdupq_n_f32(0.0f);
                    acc_hi[r] = vdupq_n_f32(0.0f);
                }

                for (int logical_block = 0; logical_block < logical_blocks_needed;
                     ++logical_block) {
                    const int physical_block = block_table[logical_block];
                    const int token_begin = logical_block * block_size;
                    const int valid_tokens = std::min(block_size, seq_len - token_begin);
                    const fp16_t* v_block = paged_kv_token_ptr(
                        v_pages, physical_block, layer_id, kv_head_id, 0,
                        num_layers, num_kv_heads, block_size, head_dim);
                    for (int offset = 0; offset < valid_tokens; ++offset) {
                        const float16x8_t hv = vld1q_f16(
                            v_block + (size_t)offset * head_dim + d);
                        const float32x4_t v_lo = vcvt_f32_f16(vget_low_f16(hv));
                        const float32x4_t v_hi = vcvt_f32_f16(vget_high_f16(hv));
                        const int key = token_begin + offset;
                        for (int r = 0; r < active_rows; ++r) {
                            const fp16_t* probability = score +
                                ((size_t)(query_base + r) * num_rep + qh) * seq_len;
                            const float p = (float)probability[key];
                            acc_lo[r] = vfmaq_n_f32(acc_lo[r], v_lo, p);
                            acc_hi[r] = vfmaq_n_f32(acc_hi[r], v_hi, p);
                        }
                    }
                }

                for (int r = 0; r < active_rows; ++r) {
                    fp16_t* output = out_head +
                        (size_t)(query_base + r) * q_row_stride + d;
                    vst1q_f16(output, vcombine_f16(
                        vcvt_f16_f32(acc_lo[r]), vcvt_f16_f32(acc_hi[r])));
                }
            }

            // head_dim 通常为 128；此分支只处理极少见的非 8 对齐尾部。
            for (; d < head_dim; ++d) {
                float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                for (int logical_block = 0; logical_block < logical_blocks_needed;
                     ++logical_block) {
                    const int physical_block = block_table[logical_block];
                    const int token_begin = logical_block * block_size;
                    const int valid_tokens = std::min(block_size, seq_len - token_begin);
                    const fp16_t* v_block = paged_kv_token_ptr(
                        v_pages, physical_block, layer_id, kv_head_id, 0,
                        num_layers, num_kv_heads, block_size, head_dim);
                    for (int offset = 0; offset < valid_tokens; ++offset) {
                        const int key = token_begin + offset;
                        const float value = (float)v_block[
                            (size_t)offset * head_dim + d];
                        for (int r = 0; r < active_rows; ++r) {
                            const fp16_t* probability = score +
                                ((size_t)(query_base + r) * num_rep + qh) * seq_len;
                            acc[r] += (float)probability[key] * value;
                        }
                    }
                }
                for (int r = 0; r < active_rows; ++r) {
                    out_head[(size_t)(query_base + r) * q_row_stride + d] =
                        (fp16_t)acc[r];
                }
            }
        }
    }
    return Status::SUCCESS;
}

void compare_paged_attention_output(
    const fp16_t* paged,
    const fp16_t* gather,
    int n,
    int layer_id,
    int kv_head_id,
    int seq_len,
    float tolerance
) {
    if (!paged || !gather || n <= 0) {
        return;
    }

    float max_abs = 0.0f;
    float mean_abs = 0.0f;
    float max_rel = 0.0f;
    for (int i = 0; i < n; ++i) {
        float a = (float)paged[i];
        float b = (float)gather[i];
        float abs_diff = std::fabs(a - b);
        float denom = std::max(std::fabs(b), 1e-6f);
        max_abs = std::max(max_abs, abs_diff);
        mean_abs += abs_diff;
        max_rel = std::max(max_rel, abs_diff / denom);
    }
    mean_abs /= (float)n;

    std::cerr << "[PAGED_ATTN_COMPARE]"
              << " layer=" << layer_id
              << " kv_head=" << kv_head_id
              << " seq_len=" << seq_len
              << " max_abs=" << max_abs
              << " mean_abs=" << mean_abs
              << " max_rel=" << max_rel;
    if (max_abs > tolerance) {
        std::cerr << " warning=diff_exceeds_tolerance"
                  << " tolerance=" << tolerance;
        record_paged_attention_compare_warning();
    }
    std::cerr << std::endl;
}
} // namespace

void attention_decode_score_f16_neon_public(
    const fp16_t* q,
    const fp16_t* k_cache,
    fp16_t* score,
    int num_rep,
    int seq_len,
    int head_dim,
    float scale
) {
    attention_decode_score_f16_neon(
        q, k_cache, score, num_rep, seq_len, head_dim, scale);
}

void attention_decode_value_f16_neon_public(
    const fp16_t* score,
    const fp16_t* v_cache,
    fp16_t* out,
    int num_rep,
    int seq_len,
    int head_dim
) {
    attention_decode_value_f16_neon(
        score, v_cache, out, num_rep, seq_len, head_dim);
}

Status attention_decode_score_paged_f16_neon_public(
    const fp16_t* q,
    fp16_t* score,
    int num_rep,
    int seq_len,
    int head_dim,
    float scale,
    const fp16_t* raw_k_pages,
    const int* block_table,
    int block_table_size,
    int block_size,
    int layer_id,
    int kv_head,
    int num_layers,
    int num_kv_heads,
    int num_physical_blocks
) {
    return attention_decode_score_paged_f16_neon(
        q,
        score,
        num_rep,
        seq_len,
        head_dim,
        scale,
        raw_k_pages,
        block_table,
        block_table_size,
        block_size,
        layer_id,
        kv_head,
        num_layers,
        num_kv_heads,
        num_physical_blocks);
}

Status attention_decode_value_paged_f16_neon_public(
    const fp16_t* score,
    fp16_t* out,
    int num_rep,
    int seq_len,
    int head_dim,
    const fp16_t* raw_v_pages,
    const int* block_table,
    int block_table_size,
    int block_size,
    int layer_id,
    int kv_head,
    int num_layers,
    int num_kv_heads,
    int num_physical_blocks
) {
    return attention_decode_value_paged_f16_neon(
        score,
        out,
        num_rep,
        seq_len,
        head_dim,
        raw_v_pages,
        block_table,
        block_table_size,
        block_size,
        layer_id,
        kv_head,
        num_layers,
        num_kv_heads,
        num_physical_blocks);
}

Status attention_prefill_paged_f16_neon_public(
    const fp16_t* q_chunk,
    fp16_t* out_chunk,
    int query_rows,
    int start_position,
    int q_row_stride,
    int num_rep,
    int head_dim,
    float scale,
    const fp16_t* raw_k_pages,
    const fp16_t* raw_v_pages,
    const int* block_table,
    int block_table_size,
    int block_size,
    int layer_id,
    int kv_head,
    int num_layers,
    int num_kv_heads,
    int num_physical_blocks,
    fp16_t* score,
    size_t score_elements,
    fp16_t* k_tile_workspace,
    size_t k_tile_elements
) {
    return attention_prefill_paged_f16_neon(
        q_chunk, out_chunk, query_rows, start_position, q_row_stride,
        num_rep, head_dim, scale, raw_k_pages, raw_v_pages, block_table,
        block_table_size, block_size, layer_id, kv_head, num_layers,
        num_kv_heads, num_physical_blocks, score, score_elements,
        k_tile_workspace, k_tile_elements);
}

Status attention_prefill_paged_online_f16_neon_public(
    const fp16_t* q_chunk,
    fp16_t* out_chunk,
    int query_rows,
    int start_position,
    int q_row_stride,
    int num_rep,
    int head_dim,
    float scale,
    const fp16_t* raw_k_pages,
    const fp16_t* raw_v_pages,
    const int* block_table,
    int block_table_size,
    int block_size,
    int layer_id,
    int kv_head,
    int num_layers,
    int num_kv_heads,
    int num_physical_blocks,
    fp16_t* k_tile_workspace,
    size_t k_tile_elements,
    float* output_acc_workspace,
    size_t output_acc_elements
) {
    return attention_prefill_paged_online_f16_neon(
        q_chunk, out_chunk, query_rows, start_position, q_row_stride,
        num_rep, head_dim, scale, raw_k_pages, raw_v_pages, block_table,
        block_table_size, block_size, layer_id, kv_head, num_layers,
        num_kv_heads, num_physical_blocks, k_tile_workspace,
        k_tile_elements, output_acc_workspace, output_acc_elements);
}

Status attention_neon(
    const Tensor& hidden_states, // [1, hidden_dim]
    Tensor& attn_output,         // [1, hidden_dim]
    const Tensor& w_q, const Tensor& w_k, const Tensor& w_v, const Tensor& w_o,
    const Tensor& w_q_pack, const Tensor& w_k_pack, const Tensor& w_v_pack, const Tensor& w_o_pack,
    const float* q_bias, const float* k_bias, const float* v_bias,
    const float* cos_ptr, const float* sin_ptr, 
    KVCache& kv_cache,
    int layer_id,
    int current_pos,
    const AttentionConfig& config,
    Workspace& workspace
) {

    int num_tokens = hidden_states.shape[0];
    int hidden_dim = hidden_states.shape[1];

    int q_out_dim = config.num_q_heads * config.head_dim;
    int kv_out_dim = config.num_kv_heads * config.head_dim;

    // ==========================================
    // 🛡️ 防御 1：严格的 Shape 校验
    // ==========================================
    if (w_q.shape[1] != q_out_dim || w_k.shape[1] != kv_out_dim || w_v.shape[1] != kv_out_dim) {
        return Status::INVALID_ARGUMENT; 
    }
    // 严格绑定输入特征维度，防止配置错误
    if (w_o.shape[0] != q_out_dim || w_o.shape[1] != hidden_dim) {
        return Status::INVALID_ARGUMENT;
    }

    // ==========================================
    // 🛡️ 防御 2：极致优化的 Workspace 大小检查
    // ==========================================
    // 计算 Q, K, V 投影所需的空间
    size_t q_bytes = num_tokens * q_out_dim * sizeof(float);
    size_t k_bytes = num_tokens * kv_out_dim * sizeof(float);
    size_t v_bytes = num_tokens * kv_out_dim * sizeof(float);

    // 计算 GQA 复用组大小
    int num_rep = config.num_q_heads / config.num_kv_heads;
    
    // Score 缓冲区：只需要存当前 GQA 组的分数 (极限省内存！)
    size_t score_bytes = num_tokens * num_rep * kv_cache.get_max_seq_len() * sizeof(float);

    // Attention 融合后的输出缓存 (乘 w_o 之前)
    size_t attn_out_bytes = num_tokens * q_out_dim * sizeof(float);

    // 给底层的 GEMM 预留汇编指令 Pack 打包的缓存区
    size_t pack_bytes = 1024 * 1024; // 1MB，够跑 0.5B 模型了

    size_t required = q_bytes + k_bytes + v_bytes + score_bytes + attn_out_bytes + pack_bytes;

    if (workspace.size() < required) {
        return Status::OUT_OF_MEMORY;
    }

    // ==========================================
    // 0. Workspace 内存切分
    // ==========================================
    float* ws_base = static_cast<float*>(workspace.data()); 
    
    int q_size = config.num_q_heads * config.head_dim;
    int k_size = config.num_kv_heads * config.head_dim;
    int v_size = config.num_kv_heads * config.head_dim;
    int max_seq_len = kv_cache.get_max_seq_len();
    
    float* q_proj_ptr = ws_base;                ws_base += q_size;
    float* k_proj_ptr = ws_base;                ws_base += k_size;
    float* v_proj_ptr = ws_base;                ws_base += v_size;
    
    // 💡 修复问题2：Score 大小改为 num_rep * max_seq_len，满足成组计算
    float* score_ptr  = ws_base;                ws_base += num_rep * max_seq_len; 
    float* attn_out_ptr = ws_base;              ws_base += q_size; 
    float* matmul_ws_ptr = ws_base; 

    // 💡 修复问题1：全面回归纯净的 2D 形状
    Tensor Q_proj({1, q_size}, q_proj_ptr);
    Tensor K_proj({1, k_size}, k_proj_ptr);
    Tensor V_proj({1, v_size}, v_proj_ptr);
    Tensor Attn_Out_Buf({1, q_size}, attn_out_ptr);


    // w_q 必须是 [hidden_dim, q_size]
    assert(w_q.shape[1] == q_size);
    assert(w_k.shape[1] == k_size);
    assert(w_v.shape[1] == v_size);


    // ==========================================
    // 1 & 2. QKV 投影与 RoPE
    // ==========================================
    if (num_tokens == 1 && w_q_pack.data && w_k_pack.data && w_v_pack.data) {
        Status status = linear_decode_prepacked_neon(
            hidden_states.ptr<float>(), w_q_pack.ptr<float>(), q_proj_ptr,
            hidden_dim, q_size, q_bias
        );
        if (status != Status::SUCCESS) return status;

        status = linear_decode_prepacked_neon(
            hidden_states.ptr<float>(), w_k_pack.ptr<float>(), k_proj_ptr,
            hidden_dim, k_size, k_bias
        );
        if (status != Status::SUCCESS) return status;

        status = linear_decode_prepacked_neon(
            hidden_states.ptr<float>(), w_v_pack.ptr<float>(), v_proj_ptr,
            hidden_dim, v_size, v_bias
        );
        if (status != Status::SUCCESS) return status;
    } else {
        matmul_neon(hidden_states, w_q, Q_proj, matmul_ws_ptr, false, q_bias);
        matmul_neon(hidden_states, w_k, K_proj, matmul_ws_ptr, false, k_bias);
        matmul_neon(hidden_states, w_v, V_proj, matmul_ws_ptr, false, v_bias);
    }

    // rope_neon(q_proj_ptr, cos_ptr, sin_ptr, q_size);
    // rope_neon(k_proj_ptr, cos_ptr, sin_ptr, k_size);
    // 遍历每一个 Q Head，分别应用长度为 head_dim 的 ROPE
    for(int h = 0; h < config.num_q_heads; ++h){
        rope_neon(q_proj_ptr + h * config.head_dim, cos_ptr, sin_ptr, config.head_dim);
    }

    // 遍历每一个 Kv Head，分别应用长度为 head dim 的 ROPE
    for (int h = 0; h < config.num_kv_heads; ++h){
        rope_neon(k_proj_ptr + h * config.head_dim, cos_ptr, sin_ptr, config.head_dim);
    }

    // ==========================================
    // 3. 更新 KV Cache
    // ==========================================
    kv_cache.update(
        layer_id,
        current_pos,
        reinterpret_cast<const fp16_t*>(k_proj_ptr),
        reinterpret_cast<const fp16_t*>(v_proj_ptr));

    // ==========================================
    // 4. GQA 核心计算循环 (💡 修复问题3：按 KV Head 循环！)
    // ==========================================
    float scale = 1.0f / std::sqrt((float)config.head_dim);
    int current_seq_len = current_pos + 1; 

    for (int kv_head = 0; kv_head < config.num_kv_heads; ++kv_head) {
        
        // --- 准备 K 和 V (完全纯净的 2D Tensor) ---
        float* k_cache_ptr = reinterpret_cast<float*>(kv_cache.get_k_head_ptr(layer_id, kv_head));
        float* v_cache_ptr = reinterpret_cast<float*>(kv_cache.get_v_head_ptr(layer_id, kv_head));
        
        // 💡 修复问题1：直接使用 [seq_len, head_dim] 2D 形状
        Tensor K_cache({current_seq_len, config.head_dim}, k_cache_ptr);
        Tensor V_cache({current_seq_len, config.head_dim}, v_cache_ptr);

        // --- 准备成组的 Q (Grouped Q) ---
        // 提取与此 kv_head 绑定的 num_rep 个连续 Q head
        float* q_group_ptr = q_proj_ptr + kv_head * num_rep * config.head_dim;
        // 把它们看作一个矩阵: [num_rep, head_dim]
        Tensor Q_group({num_rep, config.head_dim}, q_group_ptr);

        // --- 准备 Score ---
        // Score 尺寸为 [num_rep, current_seq_len]
        Tensor Score({num_rep, current_seq_len}, score_ptr);

        // A. 极致性能：一次计算 Score = Q_group * K_cache^T
        // 因为 K_cache 是复用的，底层会自动 pack 唯一一次 K，然后服务于 num_rep 行的 Q！
        matmul_neon(Q_group, K_cache, Score, matmul_ws_ptr, true, nullptr);

        // B. Score 缩放 (按总元素个数循环)
        int total_score_elements = num_rep * current_seq_len;
        int i = 0;
        float32x4_t v_scale = vdupq_n_f32(scale);
        for (; i <= total_score_elements - 4; i += 4) {
            float32x4_t v_score = vld1q_f32(score_ptr + i);
            v_score = vmulq_f32(v_score, v_scale);
            vst1q_f32(score_ptr + i, v_score);
        }
        for (; i < total_score_elements; i++) {
            score_ptr[i] *= scale;
        }

        // C. Softmax
        // 你写的 softmax_neon 内部是按最后维度 (seq_len) 循环的
        // 所以传 [num_rep, seq_len] 进去，它会自动按行对 num_rep 组各自做 softmax，完美兼容！
        softmax_neon(Score, Score);

        // D. 极致性能：一次计算 Output = Score * V_cache
        // 同理，V 只被 pack 唯一一次
        float* out_group_ptr = attn_out_ptr + kv_head * num_rep * config.head_dim;
        Tensor Out_group({num_rep, config.head_dim}, out_group_ptr);
        
        matmul_neon(Score, V_cache, Out_group, matmul_ws_ptr, false, nullptr);
    }

    // ==========================================
    // 5. 最终输出投影
    // ==========================================
    if (num_tokens == 1 && w_o_pack.data) {
        Status status = linear_decode_prepacked_neon(
            attn_out_ptr, w_o_pack.ptr<float>(), attn_output.ptr<float>(),
            q_size, hidden_dim, nullptr
        );
        if (status != Status::SUCCESS) return status;
    } else {
        matmul_neon(Attn_Out_Buf, w_o, attn_output, matmul_ws_ptr, false, nullptr);
    }

    return Status::SUCCESS;
}

Status attention_neon(
    const Tensor& hidden_states,
    Tensor& attn_output,
    const Tensor& w_q, const Tensor& w_k, const Tensor& w_v, const Tensor& w_o,
    const float* q_bias, const float* k_bias, const float* v_bias,
    const float* cos_ptr, const float* sin_ptr,
    KVCache& kv_cache,
    int layer_id,
    int current_pos,
    const AttentionConfig& config,
    Workspace& workspace
) {
    Tensor empty;
    return attention_neon(
        hidden_states, attn_output,
        w_q, w_k, w_v, w_o,
        empty, empty, empty, empty,
        q_bias, k_bias, v_bias,
        cos_ptr, sin_ptr,
        kv_cache, layer_id, current_pos, config, workspace
    );
}

Status attention_f16_gptq_neon(
    const Tensor& hidden_states,
    Tensor& attn_output,
    const GPTQInt8Weight& q_proj,
    const GPTQInt8Weight& k_proj,
    const GPTQInt8Weight& v_proj,
    const GPTQInt8Weight& o_proj,
    const fp16_t* q_bias,
    const fp16_t* k_bias,
    const fp16_t* v_bias,
    const fp16_t* cos_ptr,
    const fp16_t* sin_ptr,
    KVCache& kv_cache,
    int layer_id,
    int current_pos,
    const AttentionConfig& config,
    Workspace& workspace
) {
    bool debug_numeric = debug_numeric_enabled();
    if (hidden_states.dtype != DataType::FP16 || attn_output.dtype != DataType::FP16) {
        return Status::INVALID_ARGUMENT;
    }

    int hidden_dim = hidden_states.shape[1];
    int q_size = config.num_q_heads * config.head_dim;
    int kv_size = config.num_kv_heads * config.head_dim;
    int num_rep = config.num_q_heads / config.num_kv_heads;
    int max_seq_len = kv_cache.get_max_seq_len();

    size_t q_bytes = align_size((size_t)q_size * sizeof(fp16_t));
    size_t k_bytes = align_size((size_t)kv_size * sizeof(fp16_t));
    size_t v_bytes = align_size((size_t)kv_size * sizeof(fp16_t));
    size_t score_bytes = align_size((size_t)num_rep * max_seq_len * sizeof(fp16_t));
    size_t attn_bytes = align_size((size_t)q_size * sizeof(fp16_t));
    size_t required = q_bytes + k_bytes + v_bytes + score_bytes + attn_bytes + 5 * 64;
    if (workspace.size() < required) {
        return Status::OUT_OF_MEMORY;
    }

    char* base = static_cast<char*>(workspace.data());
    base = align_ptr(base); fp16_t* q_ptr = reinterpret_cast<fp16_t*>(base); base += q_bytes;
    base = align_ptr(base); fp16_t* k_ptr = reinterpret_cast<fp16_t*>(base); base += k_bytes;
    base = align_ptr(base); fp16_t* v_ptr = reinterpret_cast<fp16_t*>(base); base += v_bytes;
    base = align_ptr(base); fp16_t* score_ptr = reinterpret_cast<fp16_t*>(base); base += score_bytes;
    base = align_ptr(base); fp16_t* attn_out_ptr = reinterpret_cast<fp16_t*>(base); base += attn_bytes;

    Status status = linear_gptq_int8_decode_neon(
        hidden_states.ptr<fp16_t>(), q_proj, q_ptr, q_bias, nullptr, 0);
    if (status != Status::SUCCESS) return status;
    status = linear_gptq_int8_decode_neon(
        hidden_states.ptr<fp16_t>(), k_proj, k_ptr, k_bias, nullptr, 0);
    if (status != Status::SUCCESS) return status;
    status = linear_gptq_int8_decode_neon(
        hidden_states.ptr<fp16_t>(), v_proj, v_ptr, v_bias, nullptr, 0);
    if (status != Status::SUCCESS) return status;
    if (debug_numeric) {
        dump_f16_stats("attn_norm_in", layer_id, hidden_states.ptr<fp16_t>(), hidden_dim);
        dump_f16_stats("q_decode", layer_id, q_ptr, q_size);
        dump_f16_stats("k_decode", layer_id, k_ptr, kv_size);
        dump_f16_stats("v_decode", layer_id, v_ptr, kv_size);
    }

    for (int h = 0; h < config.num_q_heads; ++h) {
        rope_f16_neon(q_ptr + h * config.head_dim, cos_ptr, sin_ptr, config.head_dim);
    }
    for (int h = 0; h < config.num_kv_heads; ++h) {
        rope_f16_neon(k_ptr + h * config.head_dim, cos_ptr, sin_ptr, config.head_dim);
    }
    if (debug_numeric) {
        dump_f16_stats("q_rope", layer_id, q_ptr, q_size);
        dump_f16_stats("k_rope", layer_id, k_ptr, kv_size);
    }

    kv_cache.update(layer_id, current_pos, k_ptr, v_ptr);

    int current_seq_len = current_pos + 1;
    float scale = 1.0f / std::sqrt((float)config.head_dim);
    int num_tokens = hidden_states.shape.empty() ? 0 : hidden_states.shape[0];

    PagedKVView paged_view;
    bool paged_attention_requested = kv_cache.is_paged();
    bool debug_paged_attention = attention_env_flag("LLM_DEBUG_PAGED_ATTENTION");
    bool strict_paged_attention = attention_env_flag("LLM_PAGED_ATTENTION_STRICT");
    bool compare_paged_attention = attention_env_flag("LLM_PAGED_ATTENTION_COMPARE");
    bool use_paged_attention = false;
    const fp16_t* raw_k_pages = nullptr;
    const fp16_t* raw_v_pages = nullptr;
    const int* paged_block_table = nullptr;
    int paged_block_table_size = 0;
    const char* paged_fallback_reason = nullptr;

    if (paged_attention_requested) {
        if (num_tokens != 1) {
            paged_fallback_reason = "not_decode";
        } else if (!kv_cache.is_paged()) {
            paged_fallback_reason = "not_paged_kv";
        } else if (!kv_cache.get_active_paged_view(&paged_view)) {
            paged_fallback_reason = "no_active_sequence";
        } else if (!paged_view.block_table) {
            paged_fallback_reason = "missing_block_table";
        } else if (paged_view.seq_len < current_seq_len) {
            paged_fallback_reason = "seq_len_too_short";
        } else if (paged_view.head_dim != config.head_dim ||
                   paged_view.num_kv_heads != config.num_kv_heads ||
                   paged_view.block_size <= 0 ||
                   paged_view.num_layers <= layer_id) {
            paged_fallback_reason = "view_config_mismatch";
        } else {
            raw_k_pages = kv_cache.raw_k_pages();
            raw_v_pages = kv_cache.raw_v_pages();
            if (!raw_k_pages || !raw_v_pages) {
                paged_fallback_reason = "missing_raw_pages";
            } else {
                paged_block_table = paged_view.block_table->data();
                paged_block_table_size = static_cast<int>(paged_view.block_table->size());
                use_paged_attention = true;
            }
        }
    }

    if (debug_paged_attention && paged_attention_requested) {
        static int paged_attention_log_count = 0;
        if (paged_attention_log_count < 64) {
            if (use_paged_attention) {
                int logical_blocks = (current_seq_len + paged_view.block_size - 1) /
                                     paged_view.block_size;
                std::cerr << "[PAGED_ATTN] enabled"
                          << " layer=" << layer_id
                          << " pos=" << current_pos
                          << " seq_len=" << current_seq_len
                          << " block_size=" << paged_view.block_size
                          << " logical_blocks=" << logical_blocks
                          << " physical_blocks=" << paged_view.num_physical_blocks
                          << std::endl;
            } else {
                std::cerr << "[PAGED_ATTN] fallback"
                          << " layer=" << layer_id
                          << " reason=" << (paged_fallback_reason ? paged_fallback_reason : "disabled")
                          << std::endl;
            }
            paged_attention_log_count++;
        }
    }
    if (paged_attention_requested && !use_paged_attention) {
        record_paged_attention_fallback();
    }

    for (int kv_head = 0; kv_head < config.num_kv_heads; ++kv_head) {
        fp16_t* q_group_ptr = q_ptr + kv_head * num_rep * config.head_dim;
        fp16_t* out_group_ptr = attn_out_ptr + kv_head * num_rep * config.head_dim;
        int total_score = num_rep * current_seq_len;

        Tensor Score({num_rep, current_seq_len}, score_ptr, DataType::FP16);
        bool used_paged_group = false;

        if (use_paged_attention) {
            Status paged_status = attention_decode_score_paged_f16_neon(
                q_group_ptr,
                score_ptr,
                num_rep,
                current_seq_len,
                config.head_dim,
                scale,
                raw_k_pages,
                paged_block_table,
                paged_block_table_size,
                paged_view.block_size,
                layer_id,
                kv_head,
                paged_view.num_layers,
                paged_view.num_kv_heads,
                paged_view.num_physical_blocks);
            if (paged_status == Status::SUCCESS) {
                if (debug_numeric) {
                    dump_f16_stats("score_scaled", layer_id, score_ptr, total_score);
                }
                paged_status = softmax_f16_neon(Score, Score);
            }
            if (paged_status == Status::SUCCESS) {
                if (debug_numeric) {
                    dump_f16_stats("score_softmax", layer_id, score_ptr, total_score);
                }
                paged_status = attention_decode_value_paged_f16_neon(
                    score_ptr,
                    out_group_ptr,
                    num_rep,
                    current_seq_len,
                    config.head_dim,
                    raw_v_pages,
                    paged_block_table,
                    paged_block_table_size,
                    paged_view.block_size,
                    layer_id,
                    kv_head,
                    paged_view.num_layers,
                    paged_view.num_kv_heads,
                    paged_view.num_physical_blocks);
            }

            if (paged_status == Status::SUCCESS) {
                used_paged_group = true;
                record_paged_attention_call();
                if (compare_paged_attention) {
                    int group_elements = num_rep * config.head_dim;
                    std::vector<fp16_t> paged_out((size_t)group_elements);
                    std::vector<fp16_t> gather_out((size_t)group_elements);
                    std::memcpy(
                        paged_out.data(),
                        out_group_ptr,
                        (size_t)group_elements * sizeof(fp16_t));

                    fp16_t* k_cache_ptr = kv_cache.get_k_head_ptr(layer_id, kv_head);
                    fp16_t* v_cache_ptr = kv_cache.get_v_head_ptr(layer_id, kv_head);
                    attention_decode_score_f16_neon(
                        q_group_ptr,
                        k_cache_ptr,
                        score_ptr,
                        num_rep,
                        current_seq_len,
                        config.head_dim,
                        scale);
                    status = softmax_f16_neon(Score, Score);
                    if (status != Status::SUCCESS) return status;
                    attention_decode_value_f16_neon(
                        score_ptr,
                        v_cache_ptr,
                        gather_out.data(),
                        num_rep,
                        current_seq_len,
                        config.head_dim);
                    compare_paged_attention_output(
                        paged_out.data(),
                        gather_out.data(),
                        group_elements,
                        layer_id,
                        kv_head,
                        current_seq_len,
                        attention_env_float("LLM_PAGED_ATTENTION_COMPARE_TOL", 1e-2f));
                    std::memcpy(
                        out_group_ptr,
                        paged_out.data(),
                        (size_t)group_elements * sizeof(fp16_t));
                }
            } else {
                record_paged_attention_fallback();
                if (debug_paged_attention) {
                    std::cerr << "[PAGED_ATTN] fallback"
                              << " layer=" << layer_id
                              << " kv_head=" << kv_head
                              << " reason=kernel_error"
                              << " status=" << StatusToString(paged_status)
                              << std::endl;
                }
                if (strict_paged_attention) {
                    return paged_status;
                }
            }
        }

        if (!used_paged_group) {
            fp16_t* k_cache_ptr = kv_cache.get_k_head_ptr(layer_id, kv_head);
            fp16_t* v_cache_ptr = kv_cache.get_v_head_ptr(layer_id, kv_head);
            attention_decode_score_f16_neon(
                q_group_ptr,
                k_cache_ptr,
                score_ptr,
                num_rep,
                current_seq_len,
                config.head_dim,
                scale);
            if (debug_numeric) {
                dump_f16_stats("score_scaled", layer_id, score_ptr, total_score);
            }

            status = softmax_f16_neon(Score, Score);
            if (status != Status::SUCCESS) return status;
            if (debug_numeric) {
                dump_f16_stats("score_softmax", layer_id, score_ptr, total_score);
            }

            attention_decode_value_f16_neon(
                score_ptr,
                v_cache_ptr,
                out_group_ptr,
                num_rep,
                current_seq_len,
                config.head_dim);
        }
        if (debug_numeric) {
            dump_f16_stats("attn_group_out", layer_id, out_group_ptr, num_rep * config.head_dim);
        }
    }

    status = linear_gptq_int8_decode_neon(
        attn_out_ptr, o_proj, attn_output.ptr<fp16_t>(), nullptr, nullptr, 0);
    if (debug_numeric) {
        dump_f16_stats("o_proj_out", layer_id, attn_output.ptr<fp16_t>(), hidden_dim);
    }
    return status;
}

Status attention_f16_gptq_prefill_neon(
    const Tensor& hidden_states,
    Tensor& attn_output,
    const GPTQInt8Weight& q_proj,
    const GPTQInt8Weight& k_proj,
    const GPTQInt8Weight& v_proj,
    const GPTQInt8Weight& o_proj,
    const fp16_t* q_bias,
    const fp16_t* k_bias,
    const fp16_t* v_bias,
    const fp16_t* cos_base,
    const fp16_t* sin_base,
    KVCache& kv_cache,
    int layer_id,
    int start_pos,
    const AttentionConfig& config,
    Workspace& workspace
) {
    bool debug_numeric = debug_numeric_enabled();
    if (hidden_states.dtype != DataType::FP16 || attn_output.dtype != DataType::FP16) {
        return Status::INVALID_ARGUMENT;
    }
    if (hidden_states.shape.size() < 2 || attn_output.shape.size() < 2) {
        return Status::INVALID_ARGUMENT;
    }

    const int rows = hidden_states.shape[0];
    const int hidden_dim = hidden_states.shape[1];
    const int q_size = config.num_q_heads * config.head_dim;
    const int kv_size = config.num_kv_heads * config.head_dim;
    const int num_rep = config.num_q_heads / config.num_kv_heads;
    const int max_seq_len = kv_cache.get_max_seq_len();
    if (rows <= 0 ||
        attn_output.shape[0] != rows ||
        attn_output.shape[1] != hidden_dim ||
        start_pos < 0 ||
        start_pos + rows > max_seq_len) {
        return Status::INVALID_ARGUMENT;
    }

    if (rows == 1) {
        const fp16_t* cos_ptr = cos_base + (size_t)start_pos * config.head_dim;
        const fp16_t* sin_ptr = sin_base + (size_t)start_pos * config.head_dim;
        return attention_f16_gptq_neon(
            hidden_states, attn_output,
            q_proj, k_proj, v_proj, o_proj,
            q_bias, k_bias, v_bias,
            cos_ptr, sin_ptr,
            kv_cache, layer_id, start_pos,
            config, workspace);
    }

    size_t q_bytes = align_size((size_t)rows * q_size * sizeof(fp16_t));
    size_t k_bytes = align_size((size_t)rows * kv_size * sizeof(fp16_t));
    size_t v_bytes = align_size((size_t)rows * kv_size * sizeof(fp16_t));
    size_t score_bytes = align_size((size_t)num_rep * max_seq_len * sizeof(fp16_t));
    size_t attn_bytes = align_size((size_t)rows * q_size * sizeof(fp16_t));
    size_t required = q_bytes + k_bytes + v_bytes + score_bytes + attn_bytes + 5 * 64;
    if (workspace.size() < required) {
        return Status::OUT_OF_MEMORY;
    }

    char* base = static_cast<char*>(workspace.data());
    base = align_ptr(base); fp16_t* q_ptr = reinterpret_cast<fp16_t*>(base); base += q_bytes;
    base = align_ptr(base); fp16_t* k_ptr = reinterpret_cast<fp16_t*>(base); base += k_bytes;
    base = align_ptr(base); fp16_t* v_ptr = reinterpret_cast<fp16_t*>(base); base += v_bytes;
    base = align_ptr(base); fp16_t* score_ptr = reinterpret_cast<fp16_t*>(base); base += score_bytes;
    base = align_ptr(base); fp16_t* attn_out_ptr = reinterpret_cast<fp16_t*>(base); base += attn_bytes;

    Status status = linear_gptq_int8_batch_neon(
        hidden_states.ptr<fp16_t>(), rows, q_proj, q_ptr, q_bias, nullptr, 0);
    if (status != Status::SUCCESS) return status;
    status = linear_gptq_int8_batch_neon(
        hidden_states.ptr<fp16_t>(), rows, k_proj, k_ptr, k_bias, nullptr, 0);
    if (status != Status::SUCCESS) return status;
    status = linear_gptq_int8_batch_neon(
        hidden_states.ptr<fp16_t>(), rows, v_proj, v_ptr, v_bias, nullptr, 0);
    if (status != Status::SUCCESS) return status;

    for (int t = 0; t < rows; ++t) {
        int pos = start_pos + t;
        const fp16_t* cos_ptr = cos_base + (size_t)pos * config.head_dim;
        const fp16_t* sin_ptr = sin_base + (size_t)pos * config.head_dim;
        fp16_t* q_row = q_ptr + (size_t)t * q_size;
        fp16_t* k_row = k_ptr + (size_t)t * kv_size;

        for (int h = 0; h < config.num_q_heads; ++h) {
            rope_f16_neon(q_row + h * config.head_dim, cos_ptr, sin_ptr, config.head_dim);
        }
        for (int h = 0; h < config.num_kv_heads; ++h) {
            rope_f16_neon(k_row + h * config.head_dim, cos_ptr, sin_ptr, config.head_dim);
        }

        kv_cache.update(
            layer_id,
            pos,
            k_row,
            v_ptr + (size_t)t * kv_size);
    }

    float scale = 1.0f / std::sqrt((float)config.head_dim);
    for (int t = 0; t < rows; ++t) {
        int pos = start_pos + t;
        int current_seq_len = pos + 1;
        fp16_t* q_row = q_ptr + (size_t)t * q_size;
        fp16_t* out_row = attn_out_ptr + (size_t)t * q_size;

        for (int kv_head = 0; kv_head < config.num_kv_heads; ++kv_head) {
            fp16_t* k_cache_ptr = kv_cache.get_k_head_ptr(layer_id, kv_head);
            fp16_t* v_cache_ptr = kv_cache.get_v_head_ptr(layer_id, kv_head);
            fp16_t* q_group_ptr = q_row + kv_head * num_rep * config.head_dim;
            fp16_t* out_group_ptr = out_row + kv_head * num_rep * config.head_dim;

            Tensor Score({num_rep, current_seq_len}, score_ptr, DataType::FP16);
            attention_decode_score_f16_neon(
                q_group_ptr,
                k_cache_ptr,
                score_ptr,
                num_rep,
                current_seq_len,
                config.head_dim,
                scale);

            status = softmax_f16_neon(Score, Score);
            if (status != Status::SUCCESS) return status;

            attention_decode_value_f16_neon(
                score_ptr,
                v_cache_ptr,
                out_group_ptr,
                num_rep,
                current_seq_len,
                config.head_dim);
        }
    }

    if (debug_numeric) {
        dump_f16_stats("prefill_attn_out_pre_o", layer_id, attn_out_ptr, rows * q_size);
    }

    status = linear_gptq_int8_batch_neon(
        attn_out_ptr, rows, o_proj, attn_output.ptr<fp16_t>(), nullptr, nullptr, 0);
    if (debug_numeric) {
        dump_f16_stats("prefill_o_proj_out", layer_id, attn_output.ptr<fp16_t>(), rows * hidden_dim);
    }
    return status;
}

} // namespace arm_neon
} // namespace llm_engine
