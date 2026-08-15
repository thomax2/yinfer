#include "backends/cpu/arm_neon/neon_ops.h"
#include "backends/cpu/arm_neon/kernel_common.h"
#include "backends/cpu/arm_neon/pack.h"
#include "llm_engine/runtime/thread_pool.h"
#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>
#include <utility>
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

inline float32x4_t exp_neon_f32(float32x4_t x) {
    x = vmaxq_f32(x, vdupq_n_f32(-87.3365f));
    x = vminq_f32(x, vdupq_n_f32(88.0296f));

    float32x4_t y = vmulq_f32(x, vdupq_n_f32(1.44269504f));
    int32x4_t n = vcvtnq_s32_f32(y);
    float32x4_t fn = vcvtq_f32_s32(n);
    float32x4_t r = vmlsq_f32(x, fn, vdupq_n_f32(0.69314718f));

    float32x4_t poly = vdupq_n_f32(0.00833333f);
    poly = vfmaq_f32(vdupq_n_f32(0.04166667f), poly, r);
    poly = vfmaq_f32(vdupq_n_f32(0.16666667f), poly, r);
    poly = vfmaq_f32(vdupq_n_f32(0.5f), poly, r);
    poly = vfmaq_f32(vdupq_n_f32(1.0f), poly, r);
    poly = vfmaq_f32(vdupq_n_f32(1.0f), poly, r);

    int32x4_t exp_int = vshlq_n_s32(vaddq_s32(n, vdupq_n_s32(127)), 23);
    return vmulq_f32(poly, vreinterpretq_f32_s32(exp_int));
}

inline float32x4_t sigmoid_neon_f32(float32x4_t x) {
    float32x4_t one = vdupq_n_f32(1.0f);
    return vdivq_f32(one, vaddq_f32(one, exp_neon_f32(vnegq_f32(x))));
}
} // namespace

// =============================================================================
// Range kernel：计算 [panel_begin, panel_end) 范围内的输出通道
// 不申请内存、不使用线程池、不访问 g_memory_pool。
// =============================================================================
Status linear_decode_prepacked_range_neon(
    const float* x,
    const float* w_pack,
    float* y,
    int K,
    int N,
    int panel_begin,
    int panel_end,
    const float* bias
) {
    if (!x || !w_pack || !y || K <= 0 || N <= 0) {
        return Status::INVALID_ARGUMENT;
    }
    if (panel_begin < 0 || panel_end < panel_begin) {
        return Status::INVALID_ARGUMENT;
    }

    int np = (N + NR - 1) / NR;
    if (panel_end > np) panel_end = np;

    for (int panel = panel_begin; panel < panel_end; ++panel) {
        const float* wp = w_pack + (size_t)panel * K * NR;

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

// =============================================================================
// Serial wrapper：保持旧接口，内部委托给 range kernel
// =============================================================================
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
    return linear_decode_prepacked_range_neon(
        x, w_pack, y, K, N, 0, np, bias
    );
}

// =============================================================================
// Parallel wrapper：把 panel 范围切给线程池
// 每个 worker 写 y 的不同输出通道区间，无锁。
// =============================================================================
Status linear_decode_prepacked_parallel_neon(
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

    // 线程池不可用，或者输出维度太小，不值得并行
    if (g_thread_pool == nullptr ||
        g_thread_pool->num_threads() <= 1 ||
        N < 8192) {
        return linear_decode_prepacked_neon(x, w_pack, y, K, N, bias);
    }

    int np = (N + NR - 1) / NR;
    constexpr int grain = 16; // 16 panels = 192 输出通道

    std::atomic<int> err_flag(0);
    g_thread_pool->parallel_for(0, np, grain, [&](int pb, int pe) {
        Status s = linear_decode_prepacked_range_neon(
            x, w_pack, y, K, N, pb, pe, bias
        );
        if (s != Status::SUCCESS) {
            err_flag.store(static_cast<int>(s), std::memory_order_relaxed);
        }
    });

    int err = err_flag.load(std::memory_order_relaxed);
    return err == 0 ? Status::SUCCESS : static_cast<Status>(err);
}

// =============================================================================
// LM Head 专用：fused argmax range kernel
// 在 [panel_begin, panel_end) 内计算 logit，同时维护 local max。
// 不写完整 y。
// =============================================================================
static ArgmaxResult linear_decode_prepacked_argmax_range_neon(
    const float* x,
    const float* w_pack,
    int K,
    int N,
    int panel_begin,
    int panel_end
) {
    ArgmaxResult result;
    result.index = -1;
    result.value = -std::numeric_limits<float>::infinity();

    int np = (N + NR - 1) / NR;
    if (panel_end > np) panel_end = np;
    if (panel_begin < 0) panel_begin = 0;

    for (int panel = panel_begin; panel < panel_end; ++panel) {
        const float* wp = w_pack + (size_t)panel * K * NR;

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

        float temp[NR];
        vst1q_f32(temp, acc0);
        vst1q_f32(temp + 4, acc1);
        vst1q_f32(temp + 8, acc2);

        // 只在有效通道上做 argmax；padding 不参与。
        // 严格大于才更新，保证两值相等时 index 更小，结果更确定。
        for (int i = 0; i < actual_n; ++i) {
            float v = temp[i];
            if (v > result.value) {
                result.value = v;
                result.index = col + i;
            }
        }
    }

    return result;
}

ArgmaxResult linear_decode_prepacked_argmax_parallel_neon(
    const float* x,
    const float* w_pack,
    int K,
    int N
) {
    ArgmaxResult result;
    result.index = -1;
    result.value = -std::numeric_limits<float>::infinity();

    if (!x || !w_pack || K <= 0 || N <= 0) {
        return result;
    }

    int np = (N + NR - 1) / NR;

    // 串行 fallback：线程池不可用、单线程、或 N 太小
    if (g_thread_pool == nullptr ||
        g_thread_pool->num_threads() <= 1 ||
        N < 4096) {
        return linear_decode_prepacked_argmax_range_neon(
            x, w_pack, K, N, 0, np
        );
    }

    int n_threads = g_thread_pool->num_threads();
    // 每个 worker 一个独立的 result slot；按 chunk 顺序写
    std::vector<ArgmaxResult> partials;
    partials.reserve(n_threads);

    constexpr int grain = 64; // 64 panels = 768 输出通道

    // parallel_for 的切块策略和 partials 的 slot 数必须一致。
    // 这里简单地：先在主线程切块，然后用 parallel_for 在每个 slot 上跑。
    int total = np;
    int max_chunks = (total + grain - 1) / grain;
    int n_chunks = std::min(n_threads, max_chunks);
    if (n_chunks <= 1) {
        return linear_decode_prepacked_argmax_range_neon(
            x, w_pack, K, N, 0, np
        );
    }

    partials.assign(n_chunks, ArgmaxResult{-1, -std::numeric_limits<float>::infinity()});

    int base = total / n_chunks;
    int rem = total % n_chunks;

    std::vector<std::pair<int,int>> ranges;
    ranges.reserve(n_chunks);
    int cursor = 0;
    for (int i = 0; i < n_chunks; ++i) {
        int len = base + (i < rem ? 1 : 0);
        ranges.push_back({cursor, cursor + len});
        cursor += len;
    }

    // 用 parallel_for 跑 [0, n_chunks)，每个 task 内部会跑一段 idx 区间。
    // 我们需要：每个任务对应一个或多个 slot，slot 之间互不重叠。
    // 简化做法：让 parallel_for 在 [0, n_chunks) 上以 grain=1 切，自然每个 slot 一个 task。
    g_thread_pool->parallel_for(0, n_chunks, 1, [&](int sb, int se) {
        for (int s = sb; s < se; ++s) {
            int pb = ranges[s].first;
            int pe = ranges[s].second;
            partials[s] = linear_decode_prepacked_argmax_range_neon(
                x, w_pack, K, N, pb, pe
            );
        }
    });

    // 主线程归并：严格大于才更新，保证 index 取更小的那个
    for (const auto& p : partials) {
        if (p.index < 0) continue;
        if (p.value > result.value ||
            (p.value == result.value && (result.index < 0 || p.index < result.index))) {
            result.value = p.value;
            result.index = p.index;
        }
    }

    return result;
}

// =============================================================================
// LM Head 专用：fused 串行 argmax wrapper（debug / 一致性对照用）。
// =============================================================================
ArgmaxResult linear_decode_prepacked_argmax_serial_neon(
    const float* x,
    const float* w_pack,
    int K,
    int N
) {
    ArgmaxResult result;
    result.index = -1;
    result.value = -std::numeric_limits<float>::infinity();

    if (!x || !w_pack || K <= 0 || N <= 0) {
        return result;
    }

    int np = (N + NR - 1) / NR;
    return linear_decode_prepacked_argmax_range_neon(
        x, w_pack, K, N, 0, np
    );
}

static inline int gptq_group_for_k(const GPTQInt8Weight& w, int k) {
    if (w.has_g_idx && w.g_idx.data) {
        return w.g_idx.ptr<int32_t>()[k];
    }
    return k / w.group_size;
}

static inline void gptq_apply_group_accum_f16(
    float32x4_t& acc0,
    float32x4_t& acc1,
    float32x4_t& acc2,
    float32x4_t& acc3,
    float32x4_t sum0,
    float32x4_t sum1,
    float32x4_t sum2,
    float32x4_t sum3,
    const fp16_t* scale,
    const int8_t* zero,
    float xsum
) {
    float16x8_t sv0 = vld1q_f16(scale);
    float16x8_t sv1 = vld1q_f16(scale + 8);
    float32x4_t s0 = vcvt_f32_f16(vget_low_f16(sv0));
    float32x4_t s1 = vcvt_f32_f16(vget_high_f16(sv0));
    float32x4_t s2 = vcvt_f32_f16(vget_low_f16(sv1));
    float32x4_t s3 = vcvt_f32_f16(vget_high_f16(sv1));

    if (zero) {
        int8x16_t zv = vld1q_s8(zero);
        int16x8_t z16_0 = vmovl_s8(vget_low_s8(zv));
        int16x8_t z16_1 = vmovl_s8(vget_high_s8(zv));
        float32x4_t xsumv = vdupq_n_f32(xsum);

        sum0 = vmlsq_f32(sum0, vcvtq_f32_s32(vmovl_s16(vget_low_s16(z16_0))), xsumv);
        sum1 = vmlsq_f32(sum1, vcvtq_f32_s32(vmovl_s16(vget_high_s16(z16_0))), xsumv);
        sum2 = vmlsq_f32(sum2, vcvtq_f32_s32(vmovl_s16(vget_low_s16(z16_1))), xsumv);
        sum3 = vmlsq_f32(sum3, vcvtq_f32_s32(vmovl_s16(vget_high_s16(z16_1))), xsumv);
    }

    acc0 = vfmaq_f32(acc0, sum0, s0);
    acc1 = vfmaq_f32(acc1, sum1, s1);
    acc2 = vfmaq_f32(acc2, sum2, s2);
    acc3 = vfmaq_f32(acc3, sum3, s3);
}

constexpr int GPTQ_PARALLEL_N_THRESHOLD = 4096;
constexpr int64_t GPTQ_PARALLEL_WORK_THRESHOLD = 2LL * 1024 * 1024;
constexpr int GPTQ_FUSED_PARALLEL_N_THRESHOLD = 1024;
constexpr int GPTQ_PARALLEL_GRAIN_PANELS = 16;
constexpr int GPTQ_LINEAR_PARALLEL_GRAIN_PANELS = 8;

static inline void gptq_store_panel_f16(
    fp16_t* y,
    const fp16_t* bias,
    int N,
    int panel,
    float32x4_t acc0,
    float32x4_t acc1,
    float32x4_t acc2,
    float32x4_t acc3
) {
    int col = panel * NR_F16;
    int actual_n = std::min(NR_F16, N - col);

    if (bias && actual_n == NR_F16) {
        float16x8_t bv0 = vld1q_f16(bias + col);
        float16x8_t bv1 = vld1q_f16(bias + col + 8);
        acc0 = vaddq_f32(acc0, vcvt_f32_f16(vget_low_f16(bv0)));
        acc1 = vaddq_f32(acc1, vcvt_f32_f16(vget_high_f16(bv0)));
        acc2 = vaddq_f32(acc2, vcvt_f32_f16(vget_low_f16(bv1)));
        acc3 = vaddq_f32(acc3, vcvt_f32_f16(vget_high_f16(bv1)));
    }

    float16x8_t out0 = vcombine_f16(vcvt_f16_f32(acc0), vcvt_f16_f32(acc1));
    float16x8_t out1 = vcombine_f16(vcvt_f16_f32(acc2), vcvt_f16_f32(acc3));
    if (actual_n == NR_F16) {
        vst1q_f16(y + col, out0);
        vst1q_f16(y + col + 8, out1);
    } else {
        float tmp[NR_F16];
        vst1q_f32(tmp, acc0);
        vst1q_f32(tmp + 4, acc1);
        vst1q_f32(tmp + 8, acc2);
        vst1q_f32(tmp + 12, acc3);
        for (int lane = 0; lane < actual_n; ++lane) {
            float v = tmp[lane];
            if (bias) v += (float)bias[col + lane];
            y[col + lane] = (fp16_t)v;
        }
    }
}

static Status linear_gptq_int8_decode_range_neon(
    const fp16_t* x,
    const GPTQInt8Weight& w,
    fp16_t* y,
    const fp16_t* bias,
    int panel_begin,
    int panel_end
) {
    if (!x || !y || !w.qweight_pack.data || !w.scales_pack.data || w.K <= 0 || w.N <= 0) {
        return Status::INVALID_ARGUMENT;
    }
    if (panel_begin < 0 || panel_end < panel_begin) {
        return Status::INVALID_ARGUMENT;
    }

    const int K_pad = ((w.K + 7) / 8) * 8;
    const int np = (w.N + NR_F16 - 1) / NR_F16;
    if (panel_end > np) panel_end = np;
    const int8_t* qpack = w.qweight_pack.ptr<int8_t>();
    const fp16_t* spack = w.scales_pack.ptr<fp16_t>();
    const int8_t* zpack = w.zeros_pack.data ? w.zeros_pack.ptr<int8_t>() : nullptr;

    if (!w.has_g_idx) {
        int panel = panel_begin;
        for (; panel + 1 < panel_end; panel += 2) {
            float32x4_t p0_acc0 = vdupq_n_f32(0.0f);
            float32x4_t p0_acc1 = vdupq_n_f32(0.0f);
            float32x4_t p0_acc2 = vdupq_n_f32(0.0f);
            float32x4_t p0_acc3 = vdupq_n_f32(0.0f);
            float32x4_t p1_acc0 = vdupq_n_f32(0.0f);
            float32x4_t p1_acc1 = vdupq_n_f32(0.0f);
            float32x4_t p1_acc2 = vdupq_n_f32(0.0f);
            float32x4_t p1_acc3 = vdupq_n_f32(0.0f);

            int panel1 = panel + 1;
            for (int group = 0; group < w.num_groups; ++group) {
                int k_begin = group * w.group_size;
                int k_end = std::min(k_begin + w.group_size, w.K);
                if (k_begin >= k_end) continue;

                float xsum = 0.0f;
                float32x4_t p0_sum0 = vdupq_n_f32(0.0f);
                float32x4_t p0_sum1 = vdupq_n_f32(0.0f);
                float32x4_t p0_sum2 = vdupq_n_f32(0.0f);
                float32x4_t p0_sum3 = vdupq_n_f32(0.0f);
                float32x4_t p1_sum0 = vdupq_n_f32(0.0f);
                float32x4_t p1_sum1 = vdupq_n_f32(0.0f);
                float32x4_t p1_sum2 = vdupq_n_f32(0.0f);
                float32x4_t p1_sum3 = vdupq_n_f32(0.0f);

                for (int k = k_begin; k < k_end; ++k) {
                    float xv_scalar = (float)x[k];
                    xsum += xv_scalar;
                    float32x4_t xv = vdupq_n_f32(xv_scalar);

                    {
                        const int8_t* q = qpack + ((size_t)panel * K_pad + k) * NR_F16;
                        int8x16_t qv = vld1q_s8(q);
                        int16x8_t d0 = vmovl_s8(vget_low_s8(qv));
                        int16x8_t d1 = vmovl_s8(vget_high_s8(qv));

                        p0_sum0 = vfmaq_f32(p0_sum0, vcvtq_f32_s32(vmovl_s16(vget_low_s16(d0))), xv);
                        p0_sum1 = vfmaq_f32(p0_sum1, vcvtq_f32_s32(vmovl_s16(vget_high_s16(d0))), xv);
                        p0_sum2 = vfmaq_f32(p0_sum2, vcvtq_f32_s32(vmovl_s16(vget_low_s16(d1))), xv);
                        p0_sum3 = vfmaq_f32(p0_sum3, vcvtq_f32_s32(vmovl_s16(vget_high_s16(d1))), xv);
                    }

                    {
                        const int8_t* q = qpack + ((size_t)panel1 * K_pad + k) * NR_F16;
                        int8x16_t qv = vld1q_s8(q);
                        int16x8_t d0 = vmovl_s8(vget_low_s8(qv));
                        int16x8_t d1 = vmovl_s8(vget_high_s8(qv));

                        p1_sum0 = vfmaq_f32(p1_sum0, vcvtq_f32_s32(vmovl_s16(vget_low_s16(d0))), xv);
                        p1_sum1 = vfmaq_f32(p1_sum1, vcvtq_f32_s32(vmovl_s16(vget_high_s16(d0))), xv);
                        p1_sum2 = vfmaq_f32(p1_sum2, vcvtq_f32_s32(vmovl_s16(vget_low_s16(d1))), xv);
                        p1_sum3 = vfmaq_f32(p1_sum3, vcvtq_f32_s32(vmovl_s16(vget_high_s16(d1))), xv);
                    }
                }

                const fp16_t* p0_s = spack + ((size_t)panel * w.num_groups + group) * NR_F16;
                const int8_t* p0_z = zpack ? zpack + ((size_t)panel * w.num_groups + group) * NR_F16 : nullptr;
                gptq_apply_group_accum_f16(
                    p0_acc0, p0_acc1, p0_acc2, p0_acc3,
                    p0_sum0, p0_sum1, p0_sum2, p0_sum3,
                    p0_s, p0_z, xsum);

                const fp16_t* p1_s = spack + ((size_t)panel1 * w.num_groups + group) * NR_F16;
                const int8_t* p1_z = zpack ? zpack + ((size_t)panel1 * w.num_groups + group) * NR_F16 : nullptr;
                gptq_apply_group_accum_f16(
                    p1_acc0, p1_acc1, p1_acc2, p1_acc3,
                    p1_sum0, p1_sum1, p1_sum2, p1_sum3,
                    p1_s, p1_z, xsum);
            }

            gptq_store_panel_f16(y, bias, w.N, panel, p0_acc0, p0_acc1, p0_acc2, p0_acc3);
            gptq_store_panel_f16(y, bias, w.N, panel1, p1_acc0, p1_acc1, p1_acc2, p1_acc3);
        }

        for (; panel < panel_end; ++panel) {
            float32x4_t acc0 = vdupq_n_f32(0.0f);
            float32x4_t acc1 = vdupq_n_f32(0.0f);
            float32x4_t acc2 = vdupq_n_f32(0.0f);
            float32x4_t acc3 = vdupq_n_f32(0.0f);

            for (int group = 0; group < w.num_groups; ++group) {
                int k_begin = group * w.group_size;
                int k_end = std::min(k_begin + w.group_size, w.K);
                if (k_begin >= k_end) continue;

                float xsum = 0.0f;
                float32x4_t sum0 = vdupq_n_f32(0.0f);
                float32x4_t sum1 = vdupq_n_f32(0.0f);
                float32x4_t sum2 = vdupq_n_f32(0.0f);
                float32x4_t sum3 = vdupq_n_f32(0.0f);

                for (int k = k_begin; k < k_end; ++k) {
                    float xv_scalar = (float)x[k];
                    xsum += xv_scalar;
                    float32x4_t xv = vdupq_n_f32(xv_scalar);

                    const int8_t* q = qpack + ((size_t)panel * K_pad + k) * NR_F16;
                    int8x16_t qv = vld1q_s8(q);
                    int16x8_t d0 = vmovl_s8(vget_low_s8(qv));
                    int16x8_t d1 = vmovl_s8(vget_high_s8(qv));

                    sum0 = vfmaq_f32(sum0, vcvtq_f32_s32(vmovl_s16(vget_low_s16(d0))), xv);
                    sum1 = vfmaq_f32(sum1, vcvtq_f32_s32(vmovl_s16(vget_high_s16(d0))), xv);
                    sum2 = vfmaq_f32(sum2, vcvtq_f32_s32(vmovl_s16(vget_low_s16(d1))), xv);
                    sum3 = vfmaq_f32(sum3, vcvtq_f32_s32(vmovl_s16(vget_high_s16(d1))), xv);
                }

                const fp16_t* s = spack + ((size_t)panel * w.num_groups + group) * NR_F16;
                const int8_t* z = zpack ? zpack + ((size_t)panel * w.num_groups + group) * NR_F16 : nullptr;
                gptq_apply_group_accum_f16(
                    acc0, acc1, acc2, acc3,
                    sum0, sum1, sum2, sum3,
                    s, z, xsum);
            }

            gptq_store_panel_f16(y, bias, w.N, panel, acc0, acc1, acc2, acc3);
        }

        return Status::SUCCESS;
    }

    int panel = panel_begin;
    for (; panel + 1 < panel_end; panel += 2) {
        float32x4_t p0_acc0 = vdupq_n_f32(0.0f);
        float32x4_t p0_acc1 = vdupq_n_f32(0.0f);
        float32x4_t p0_acc2 = vdupq_n_f32(0.0f);
        float32x4_t p0_acc3 = vdupq_n_f32(0.0f);
        float32x4_t p1_acc0 = vdupq_n_f32(0.0f);
        float32x4_t p1_acc1 = vdupq_n_f32(0.0f);
        float32x4_t p1_acc2 = vdupq_n_f32(0.0f);
        float32x4_t p1_acc3 = vdupq_n_f32(0.0f);

        for (int k = 0; k < w.K; ++k) {
            int group = gptq_group_for_k(w, k);
            float32x4_t xv = vdupq_n_f32((float)x[k]);

            {
                const int8_t* q = qpack + ((size_t)panel * K_pad + k) * NR_F16;
                const fp16_t* s = spack + ((size_t)panel * w.num_groups + group) * NR_F16;
                const int8_t* z = zpack ? zpack + ((size_t)panel * w.num_groups + group) * NR_F16 : nullptr;

                int8x16_t qv = vld1q_s8(q);
                int16x8_t d0 = vmovl_s8(vget_low_s8(qv));
                int16x8_t d1 = vmovl_s8(vget_high_s8(qv));
                if (z) {
                    int8x16_t zv = vld1q_s8(z);
                    d0 = vsubq_s16(d0, vmovl_s8(vget_low_s8(zv)));
                    d1 = vsubq_s16(d1, vmovl_s8(vget_high_s8(zv)));
                }

                float16x8_t sv0 = vld1q_f16(s);
                float16x8_t sv1 = vld1q_f16(s + 8);
                p0_acc0 = vfmaq_f32(p0_acc0,
                                     vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(d0))),
                                               vcvt_f32_f16(vget_low_f16(sv0))),
                                     xv);
                p0_acc1 = vfmaq_f32(p0_acc1,
                                     vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(d0))),
                                               vcvt_f32_f16(vget_high_f16(sv0))),
                                     xv);
                p0_acc2 = vfmaq_f32(p0_acc2,
                                     vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(d1))),
                                               vcvt_f32_f16(vget_low_f16(sv1))),
                                     xv);
                p0_acc3 = vfmaq_f32(p0_acc3,
                                     vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(d1))),
                                               vcvt_f32_f16(vget_high_f16(sv1))),
                                     xv);
            }

            {
                int panel1 = panel + 1;
                const int8_t* q = qpack + ((size_t)panel1 * K_pad + k) * NR_F16;
                const fp16_t* s = spack + ((size_t)panel1 * w.num_groups + group) * NR_F16;
                const int8_t* z = zpack ? zpack + ((size_t)panel1 * w.num_groups + group) * NR_F16 : nullptr;

                int8x16_t qv = vld1q_s8(q);
                int16x8_t d0 = vmovl_s8(vget_low_s8(qv));
                int16x8_t d1 = vmovl_s8(vget_high_s8(qv));
                if (z) {
                    int8x16_t zv = vld1q_s8(z);
                    d0 = vsubq_s16(d0, vmovl_s8(vget_low_s8(zv)));
                    d1 = vsubq_s16(d1, vmovl_s8(vget_high_s8(zv)));
                }

                float16x8_t sv0 = vld1q_f16(s);
                float16x8_t sv1 = vld1q_f16(s + 8);
                p1_acc0 = vfmaq_f32(p1_acc0,
                                     vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(d0))),
                                               vcvt_f32_f16(vget_low_f16(sv0))),
                                     xv);
                p1_acc1 = vfmaq_f32(p1_acc1,
                                     vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(d0))),
                                               vcvt_f32_f16(vget_high_f16(sv0))),
                                     xv);
                p1_acc2 = vfmaq_f32(p1_acc2,
                                     vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(d1))),
                                               vcvt_f32_f16(vget_low_f16(sv1))),
                                     xv);
                p1_acc3 = vfmaq_f32(p1_acc3,
                                     vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(d1))),
                                               vcvt_f32_f16(vget_high_f16(sv1))),
                                     xv);
            }
        }

        gptq_store_panel_f16(y, bias, w.N, panel, p0_acc0, p0_acc1, p0_acc2, p0_acc3);
        gptq_store_panel_f16(y, bias, w.N, panel + 1, p1_acc0, p1_acc1, p1_acc2, p1_acc3);
    }

    for (; panel < panel_end; ++panel) {
        float32x4_t acc0 = vdupq_n_f32(0.0f);
        float32x4_t acc1 = vdupq_n_f32(0.0f);
        float32x4_t acc2 = vdupq_n_f32(0.0f);
        float32x4_t acc3 = vdupq_n_f32(0.0f);

        for (int k = 0; k < w.K; ++k) {
            int group = gptq_group_for_k(w, k);
            const int8_t* q = qpack + ((size_t)panel * K_pad + k) * NR_F16;
            const fp16_t* s = spack + ((size_t)panel * w.num_groups + group) * NR_F16;
            const int8_t* z = zpack ? zpack + ((size_t)panel * w.num_groups + group) * NR_F16 : nullptr;

            // q/zero are int8, but q-zero may need 9 bits. Widen before
            // subtracting, then widen again to FP32 for stable accumulation.
            int8x16_t qv = vld1q_s8(q);
            int16x8_t d0 = vmovl_s8(vget_low_s8(qv));
            int16x8_t d1 = vmovl_s8(vget_high_s8(qv));
            if (z) {
                int8x16_t zv = vld1q_s8(z);
                d0 = vsubq_s16(d0, vmovl_s8(vget_low_s8(zv)));
                d1 = vsubq_s16(d1, vmovl_s8(vget_high_s8(zv)));
            }

            float16x8_t sv0 = vld1q_f16(s);
            float16x8_t sv1 = vld1q_f16(s + 8);
            float32x4_t xv = vdupq_n_f32((float)x[k]);

            float32x4_t q0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(d0)));
            float32x4_t q1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(d0)));
            float32x4_t q2 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(d1)));
            float32x4_t q3 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(d1)));

            float32x4_t s0 = vcvt_f32_f16(vget_low_f16(sv0));
            float32x4_t s1 = vcvt_f32_f16(vget_high_f16(sv0));
            float32x4_t s2 = vcvt_f32_f16(vget_low_f16(sv1));
            float32x4_t s3 = vcvt_f32_f16(vget_high_f16(sv1));

            acc0 = vfmaq_f32(acc0, vmulq_f32(q0, s0), xv);
            acc1 = vfmaq_f32(acc1, vmulq_f32(q1, s1), xv);
            acc2 = vfmaq_f32(acc2, vmulq_f32(q2, s2), xv);
            acc3 = vfmaq_f32(acc3, vmulq_f32(q3, s3), xv);
        }

        gptq_store_panel_f16(y, bias, w.N, panel, acc0, acc1, acc2, acc3);
    }

    return Status::SUCCESS;
}

Status linear_gptq_int8_decode_neon(
    const fp16_t* x,
    const GPTQInt8Weight& w,
    fp16_t* y,
    const fp16_t* bias,
    void*,
    size_t
) {
    if (!x || !y || !w.qweight_pack.data || !w.scales_pack.data || w.K <= 0 || w.N <= 0) {
        return Status::INVALID_ARGUMENT;
    }

    int np = (w.N + NR_F16 - 1) / NR_F16;
    int64_t work = (int64_t)w.K * w.N;
    if (g_thread_pool == nullptr ||
        g_thread_pool->num_threads() <= 1 ||
        (w.N < GPTQ_PARALLEL_N_THRESHOLD && work < GPTQ_PARALLEL_WORK_THRESHOLD)) {
        return linear_gptq_int8_decode_range_neon(x, w, y, bias, 0, np);
    }

    std::atomic<int> err_flag(0);
    g_thread_pool->parallel_for(0, np, GPTQ_LINEAR_PARALLEL_GRAIN_PANELS, [&](int pb, int pe) {
        Status s = linear_gptq_int8_decode_range_neon(x, w, y, bias, pb, pe);
        if (s != Status::SUCCESS) {
            err_flag.store(static_cast<int>(s), std::memory_order_relaxed);
        }
    });

    int err = err_flag.load(std::memory_order_relaxed);
    return err == 0 ? Status::SUCCESS : static_cast<Status>(err);
}

namespace {

constexpr int GPTQ_BATCH_MAX_ROWS = 8;
constexpr int GPTQ_BATCH_PANEL_GRAIN = 8;
constexpr int GPTQ_BATCH_ARGMAX_GRAIN = 16;

std::atomic<uint64_t> g_batch_kernel_calls{0};
std::atomic<uint64_t> g_batch_rows_total{0};
std::atomic<uint64_t> g_batch_output_panel_tasks{0};
std::atomic<uint64_t> g_batch_row_gemv_fallbacks{0};
std::atomic<uint64_t> g_batch_weight_vector_loads{0};
std::atomic<uint64_t> g_batch_dequant_vector_ops{0};
std::atomic<uint64_t> g_batch_argmax_calls{0};
std::atomic<uint64_t> g_batch_argmax_rows{0};
std::atomic<uint64_t> g_batch_full_logits_elements_written{0};
std::atomic<uint64_t> g_batch_compare_mismatches{0};

bool gptq_batch_env_flag(const char* name) {
    const char* value = std::getenv(name);
    if (!value) return false;
    std::string text(value);
    return text == "1" || text == "true" || text == "TRUE" ||
           text == "on" || text == "ON";
}

float gptq_batch_compare_tolerance() {
    const char* value = std::getenv("LLM_GPTQ_BATCH_COMPARE_TOL");
    if (!value || !*value) return 0.02f;
    char* end = nullptr;
    float result = std::strtof(value, &end);
    return end == value || result <= 0.0f ? 0.02f : result;
}

template <int ActiveRows, int Channels>
inline void gptq_batch_compute_subpanel(
    const fp16_t* x,
    const GPTQInt8Weight& w,
    int panel,
    int channel_offset,
    float* panel_out
) {
    static_assert(Channels == 8 || Channels == 16, "unsupported output tile");
    constexpr int Vecs = Channels / 4;
    for (int row = 0; row < ActiveRows; ++row) {
        for (int vec = 0; vec < Vecs; ++vec) {
            vst1q_f32(
                panel_out + (size_t)row * NR_F16 + channel_offset + vec * 4,
                vdupq_n_f32(0.0f));
        }
    }

    const int K_pad = ((w.K + 7) / 8) * 8;
    const int8_t* qpack = w.qweight_pack.ptr<int8_t>();
    const fp16_t* spack = w.scales_pack.ptr<fp16_t>();
    const int8_t* zpack = w.zeros_pack.data ? w.zeros_pack.ptr<int8_t>() : nullptr;

    for (int group = 0; group < w.num_groups; ++group) {
        float32x4_t group_sum[ActiveRows][Vecs];
        float xsum[ActiveRows];
        int k_begin = group * w.group_size;
        int k_end = std::min(k_begin + w.group_size, w.K);
        for (int row = 0; row < ActiveRows; ++row) {
            xsum[row] = 0.0f;
            for (int k_index = k_begin; k_index < k_end; ++k_index) {
                xsum[row] += (float)x[(size_t)row * w.K + k_index];
            }
            for (int vec = 0; vec < Vecs; ++vec) {
                group_sum[row][vec] = vdupq_n_f32(0.0f);
            }
        }
        for (int k_index = k_begin; k_index < k_end; ++k_index) {
            const int8_t* q_ptr =
                qpack + ((size_t)panel * K_pad + k_index) * NR_F16 + channel_offset;
            float32x4_t qvalue[Vecs];
            if constexpr (Channels == 16) {
                int8x16_t q8 = vld1q_s8(q_ptr);
                int16x8_t q0 = vmovl_s8(vget_low_s8(q8));
                int16x8_t q1 = vmovl_s8(vget_high_s8(q8));
                qvalue[0] = vcvtq_f32_s32(vmovl_s16(vget_low_s16(q0)));
                qvalue[1] = vcvtq_f32_s32(vmovl_s16(vget_high_s16(q0)));
                qvalue[2] = vcvtq_f32_s32(vmovl_s16(vget_low_s16(q1)));
                qvalue[3] = vcvtq_f32_s32(vmovl_s16(vget_high_s16(q1)));
            } else {
                int16x8_t q8 = vmovl_s8(vld1_s8(q_ptr));
                qvalue[0] = vcvtq_f32_s32(vmovl_s16(vget_low_s16(q8)));
                qvalue[1] = vcvtq_f32_s32(vmovl_s16(vget_high_s16(q8)));
            }
            for (int row = 0; row < ActiveRows; ++row) {
                float activation = (float)x[(size_t)row * w.K + k_index];
                for (int vec = 0; vec < Vecs; ++vec) {
                    group_sum[row][vec] =
                        vfmaq_n_f32(group_sum[row][vec], qvalue[vec], activation);
                }
            }
        }

        const fp16_t* scale_ptr =
            spack + ((size_t)panel * w.num_groups + group) * NR_F16 + channel_offset;
        const int8_t* zero_ptr = zpack
            ? zpack + ((size_t)panel * w.num_groups + group) * NR_F16 + channel_offset
            : nullptr;
        float32x4_t scale[Vecs];
        float32x4_t zero[Vecs];
        if constexpr (Channels == 16) {
            float16x8_t s0 = vld1q_f16(scale_ptr);
            float16x8_t s1 = vld1q_f16(scale_ptr + 8);
            scale[0] = vcvt_f32_f16(vget_low_f16(s0));
            scale[1] = vcvt_f32_f16(vget_high_f16(s0));
            scale[2] = vcvt_f32_f16(vget_low_f16(s1));
            scale[3] = vcvt_f32_f16(vget_high_f16(s1));
            if (zero_ptr) {
                int8x16_t z8 = vld1q_s8(zero_ptr);
                int16x8_t z0 = vmovl_s8(vget_low_s8(z8));
                int16x8_t z1 = vmovl_s8(vget_high_s8(z8));
                zero[0] = vcvtq_f32_s32(vmovl_s16(vget_low_s16(z0)));
                zero[1] = vcvtq_f32_s32(vmovl_s16(vget_high_s16(z0)));
                zero[2] = vcvtq_f32_s32(vmovl_s16(vget_low_s16(z1)));
                zero[3] = vcvtq_f32_s32(vmovl_s16(vget_high_s16(z1)));
            }
        } else {
            float16x8_t s = vld1q_f16(scale_ptr);
            scale[0] = vcvt_f32_f16(vget_low_f16(s));
            scale[1] = vcvt_f32_f16(vget_high_f16(s));
            if (zero_ptr) {
                int16x8_t z = vmovl_s8(vld1_s8(zero_ptr));
                zero[0] = vcvtq_f32_s32(vmovl_s16(vget_low_s16(z)));
                zero[1] = vcvtq_f32_s32(vmovl_s16(vget_high_s16(z)));
            }
        }
        if (!zero_ptr) {
            for (int vec = 0; vec < Vecs; ++vec) zero[vec] = vdupq_n_f32(0.0f);
        }
        for (int row = 0; row < ActiveRows; ++row) {
            float32x4_t xsum_vector = vdupq_n_f32(xsum[row]);
            for (int vec = 0; vec < Vecs; ++vec) {
                float* output =
                    panel_out + (size_t)row * NR_F16 + channel_offset + vec * 4;
                float32x4_t corrected =
                    vmlsq_f32(group_sum[row][vec], zero[vec], xsum_vector);
                float32x4_t total = vld1q_f32(output);
                vst1q_f32(output, vfmaq_f32(total, corrected, scale[vec]));
            }
        }
    }
}

inline Status gptq_batch_compute_panel(
    const fp16_t* x,
    int rows,
    const GPTQInt8Weight& w,
    int panel,
    float* panel_out
) {
    if (w.has_g_idx) return Status::INVALID_ARGUMENT;
    switch (rows) {
        case 1:
            gptq_batch_compute_subpanel<1, 16>(x, w, panel, 0, panel_out);
            break;
        case 2:
            gptq_batch_compute_subpanel<2, 16>(x, w, panel, 0, panel_out);
            break;
        case 3:
            gptq_batch_compute_subpanel<3, 16>(x, w, panel, 0, panel_out);
            break;
        case 4:
            gptq_batch_compute_subpanel<4, 16>(x, w, panel, 0, panel_out);
            break;
        case 5:
            gptq_batch_compute_subpanel<5, 8>(x, w, panel, 0, panel_out);
            gptq_batch_compute_subpanel<5, 8>(x, w, panel, 8, panel_out);
            break;
        case 6:
            gptq_batch_compute_subpanel<6, 8>(x, w, panel, 0, panel_out);
            gptq_batch_compute_subpanel<6, 8>(x, w, panel, 8, panel_out);
            break;
        case 7:
            gptq_batch_compute_subpanel<7, 8>(x, w, panel, 0, panel_out);
            gptq_batch_compute_subpanel<7, 8>(x, w, panel, 8, panel_out);
            break;
        case 8:
            gptq_batch_compute_subpanel<8, 8>(x, w, panel, 0, panel_out);
            gptq_batch_compute_subpanel<8, 8>(x, w, panel, 8, panel_out);
            break;
        default:
            return Status::INVALID_ARGUMENT;
    }
    return Status::SUCCESS;
}

inline void gptq_batch_accumulate_packed_rows_8x8(
    float32x4_t (&group_sum)[GPTQ_BATCH_MAX_ROWS][2],
    const float32x4_t (&qvalue)[2],
    float32x4_t activation_lo,
    float32x4_t activation_hi
) {
    // qvalue 是连续 8 个输出通道；activation_lo/hi 是连续 8 行输入。
    // lane 编号全部为编译期立即数，固定更新 8x8 输出 tile。
    for (int vec = 0; vec < 2; ++vec) {
        group_sum[0][vec] = vfmaq_laneq_f32(
            group_sum[0][vec], qvalue[vec], activation_lo, 0);
        group_sum[1][vec] = vfmaq_laneq_f32(
            group_sum[1][vec], qvalue[vec], activation_lo, 1);
        group_sum[2][vec] = vfmaq_laneq_f32(
            group_sum[2][vec], qvalue[vec], activation_lo, 2);
        group_sum[3][vec] = vfmaq_laneq_f32(
            group_sum[3][vec], qvalue[vec], activation_lo, 3);
        group_sum[4][vec] = vfmaq_laneq_f32(
            group_sum[4][vec], qvalue[vec], activation_hi, 0);
        group_sum[5][vec] = vfmaq_laneq_f32(
            group_sum[5][vec], qvalue[vec], activation_hi, 1);
        group_sum[6][vec] = vfmaq_laneq_f32(
            group_sum[6][vec], qvalue[vec], activation_hi, 2);
        group_sum[7][vec] = vfmaq_laneq_f32(
            group_sum[7][vec], qvalue[vec], activation_hi, 3);
    }
}

inline void gptq_batch_compute_packed_a_tile_8x8(
    const fp16_t* packed_x,
    const GPTQInt8Weight& w,
    int panel,
    int channel_offset,
    float* panel_out
) {
    constexpr int Vecs = 2;
    for (int row = 0; row < GPTQ_BATCH_MAX_ROWS; ++row) {
        for (int vec = 0; vec < Vecs; ++vec) {
            vst1q_f32(
                panel_out + (size_t)row * NR_F16 + channel_offset + vec * 4,
                vdupq_n_f32(0.0f));
        }
    }

    const int K_pad = ((w.K + 7) / 8) * 8;
    const int8_t* qpack = w.qweight_pack.ptr<int8_t>();
    const fp16_t* spack = w.scales_pack.ptr<fp16_t>();
    const int8_t* zpack = w.zeros_pack.data ? w.zeros_pack.ptr<int8_t>() : nullptr;

    for (int group = 0; group < w.num_groups; ++group) {
        float32x4_t group_sum[GPTQ_BATCH_MAX_ROWS][Vecs];
        for (int row = 0; row < GPTQ_BATCH_MAX_ROWS; ++row) {
            for (int vec = 0; vec < Vecs; ++vec) {
                group_sum[row][vec] = vdupq_n_f32(0.0f);
            }
        }
        float32x4_t xsum_lo = vdupq_n_f32(0.0f);
        float32x4_t xsum_hi = vdupq_n_f32(0.0f);

        const int k_begin = group * w.group_size;
        const int k_end = std::min(k_begin + w.group_size, w.K);
        for (int k_index wo= k_begin; k_index < k_end; ++k_index) {
            // packed_x 的布局为 [row_tile, K, 8]，一次连续 Load 得到同一 k
            // 对应的 8 行激活，替代原内核的多次跨行标量读取。
            float16x8_t activation_f16 = vld1q_f16(packed_x + (size_t)k_index * 8);
            float32x4_t activation_lo =
                vcvt_f32_f16(vget_low_f16(activation_f16));
            float32x4_t activation_hi =
                vcvt_f32_f16(vget_high_f16(activation_f16));
            xsum_lo = vaddq_f32(xsum_lo, activation_lo);
            xsum_hi = vaddq_f32(xsum_hi, activation_hi);

            const int8_t* q_ptr =
                qpack + ((size_t)panel * K_pad + k_index) * NR_F16 + channel_offset;
            float32x4_t qvalue[Vecs];
            int16x8_t q8 = vmovl_s8(vld1_s8(q_ptr));
            qvalue[0] = vcvtq_f32_s32(vmovl_s16(vget_low_s16(q8)));
            qvalue[1] = vcvtq_f32_s32(vmovl_s16(vget_high_s16(q8)));
            gptq_batch_accumulate_packed_rows_8x8(
                group_sum, qvalue, activation_lo, activation_hi);
        }

        alignas(16) float xsum[8];
        vst1q_f32(xsum, xsum_lo);
        vst1q_f32(xsum + 4, xsum_hi);
        const fp16_t* scale_ptr =
            spack + ((size_t)panel * w.num_groups + group) * NR_F16 + channel_offset;
        const int8_t* zero_ptr = zpack
            ? zpack + ((size_t)panel * w.num_groups + group) * NR_F16 + channel_offset
            : nullptr;
        float32x4_t scale[Vecs];
        float32x4_t zero[Vecs];
        float16x8_t s = vld1q_f16(scale_ptr);
        scale[0] = vcvt_f32_f16(vget_low_f16(s));
        scale[1] = vcvt_f32_f16(vget_high_f16(s));
        if (zero_ptr) {
            int16x8_t z = vmovl_s8(vld1_s8(zero_ptr));
            zero[0] = vcvtq_f32_s32(vmovl_s16(vget_low_s16(z)));
            zero[1] = vcvtq_f32_s32(vmovl_s16(vget_high_s16(z)));
        }
        if (!zero_ptr) {
            for (int vec = 0; vec < Vecs; ++vec) zero[vec] = vdupq_n_f32(0.0f);
        }
        for (int row = 0; row < GPTQ_BATCH_MAX_ROWS; ++row) {
            float32x4_t xsum_vector = vdupq_n_f32(xsum[row]);
            for (int vec = 0; vec < Vecs; ++vec) {
                float* output =
                    panel_out + (size_t)row * NR_F16 + channel_offset + vec * 4;
                float32x4_t corrected =
                    vmlsq_f32(group_sum[row][vec], zero[vec], xsum_vector);
                float32x4_t total = vld1q_f32(output);
                vst1q_f32(output, vfmaq_f32(total, corrected, scale[vec]));
            }
        }
    }
}

// Packed A 已把不足 8 行的最后一个 tile 补零，因此 Packed 路径只需要
// 一个固定 8x16 微内核，不再根据实际有效行数生成 1～8 行的动态分支。
// 8x16 在内部拆成两个 8x8 subpanel，是为了控制 NEON 寄存器压力；
// 这是固定微内核的实现细节，不是对输入行数再次分块。
inline void gptq_batch_compute_packed_a_tile_8x16(
    const fp16_t* packed_x,
    const GPTQInt8Weight& w,
    int panel,
    float* panel_out
) {
    gptq_batch_compute_packed_a_tile_8x8(
        packed_x, w, panel, 0, panel_out);
    gptq_batch_compute_packed_a_tile_8x8(
        packed_x, w, panel, 8, panel_out);
}

Status linear_gptq_int8_batch_packed_a_panel_range_neon(
    const fp16_t* packed_x,
    int rows,
    const GPTQInt8Weight& w,
    fp16_t* y,
    const fp16_t* bias,
    int panel_begin,
    int panel_end
) {
    alignas(64) float panel_out[GPTQ_BATCH_MAX_ROWS * NR_F16];
    const int mp = (rows + GPTQ_BATCH_MAX_ROWS - 1) / GPTQ_BATCH_MAX_ROWS;

    // Weight-stationary tile 顺序：j 是 16 列的 Packed B panel，i 是
    // 8 行的 Packed A tile。线程池把互不重叠的 Panel 区间分给 worker；
    // worker 固定一个权重 Panel 后连续服务全部行 tile，提升权重缓存复用。
    for (int j = panel_begin; j < panel_end; ++j) {
        const int col_begin = j * NR_F16;
        const int actual_n = std::min(NR_F16, w.N - col_begin);

        for (int i = 0; i < mp; ++i) {
            const int row_begin = i * GPTQ_BATCH_MAX_ROWS;
            const int actual_m =
                std::min(GPTQ_BATCH_MAX_ROWS, rows - row_begin);
            const fp16_t* packed_tile =
                packed_x + (size_t)i * w.K * GPTQ_BATCH_MAX_ROWS;

            // 即使最后一个 row tile 只有 actual_m<8 个有效行，也完整计算
            // padding 后的 8x16 tile；边界只在写回 C 时处理。
            gptq_batch_compute_packed_a_tile_8x16(
                packed_tile, w, j, panel_out);

            for (int row = 0; row < actual_m; ++row) {
                fp16_t* row_y =
                    y + (size_t)(row_begin + row) * w.N + col_begin;
                const float* row_panel = panel_out + (size_t)row * NR_F16;
                for (int lane = 0; lane < actual_n; ++lane) {
                    float value = row_panel[lane] +
                        (bias ? (float)bias[col_begin + lane] : 0.0f);
                    row_y[lane] = (fp16_t)value;
                }
            }
        }
    }
    return Status::SUCCESS;
}

Status linear_gptq_int8_batch_panel_range_neon(
    const fp16_t* x,
    int rows,
    const GPTQInt8Weight& w,
    fp16_t* y,
    const fp16_t* bias,
    int panel_begin,
    int panel_end
) {
    alignas(64) float panel_out[GPTQ_BATCH_MAX_ROWS * NR_F16];
    for (int panel = panel_begin; panel < panel_end; ++panel) {
        int col = panel * NR_F16;
        int actual_n = std::min(NR_F16, w.N - col);
        for (int row_base = 0; row_base < rows; row_base += GPTQ_BATCH_MAX_ROWS) {
            int tile_rows = std::min(GPTQ_BATCH_MAX_ROWS, rows - row_base);
            Status status = gptq_batch_compute_panel(
                x + (size_t)row_base * w.K, tile_rows, w, panel, panel_out);
            if (status != Status::SUCCESS) return status;
            for (int row = 0; row < tile_rows; ++row) {
                fp16_t* row_y = y + (size_t)(row_base + row) * w.N + col;
                const float* row_panel = panel_out + (size_t)row * NR_F16;
                for (int lane = 0; lane < actual_n; ++lane) {
                    float value = row_panel[lane] +
                        (bias ? (float)bias[col + lane] : 0.0f);
                    row_y[lane] = (fp16_t)value;
                }
            }
        }
    }
    return Status::SUCCESS;
}

Status gptq_batch_argmax_panel_range_neon(
    const fp16_t* x,
    int rows,
    const GPTQInt8Weight& w,
    int panel_begin,
    int panel_end,
    ArgmaxResult* results
) {
    for (int row = 0; row < rows; ++row) {
        results[row] = ArgmaxResult{-1, -std::numeric_limits<float>::infinity()};
    }
    alignas(64) float panel_out[GPTQ_BATCH_MAX_ROWS * NR_F16];
    for (int panel = panel_begin; panel < panel_end; ++panel) {
        Status status = gptq_batch_compute_panel(x, rows, w, panel, panel_out);
        if (status != Status::SUCCESS) return status;
        int col = panel * NR_F16;
        int actual_n = std::min(NR_F16, w.N - col);
        for (int row = 0; row < rows; ++row) {
            const float* values = panel_out + (size_t)row * NR_F16;
            for (int lane = 0; lane < actual_n; ++lane) {
                int index = col + lane;
                float value = values[lane];
                ArgmaxResult& best = results[row];
                if (value > best.value ||
                    (value == best.value && (best.index < 0 || index < best.index))) {
                    best.index = index;
                    best.value = value;
                }
            }
        }
    }
    return Status::SUCCESS;
}

void record_batch_shape(int rows, int np, int grain, int K) {
    g_batch_kernel_calls.fetch_add(1, std::memory_order_relaxed);
    g_batch_rows_total.fetch_add((uint64_t)rows, std::memory_order_relaxed);
    g_batch_output_panel_tasks.fetch_add(
        (uint64_t)((np + grain - 1) / grain), std::memory_order_relaxed);
    uint64_t subpanels = 0;
    for (int row_base = 0; row_base < rows; row_base += GPTQ_BATCH_MAX_ROWS) {
        int tile_rows = std::min(GPTQ_BATCH_MAX_ROWS, rows - row_base);
        subpanels += tile_rows <= 4 ? 1u : 2u;
    }
    g_batch_weight_vector_loads.fetch_add(
        (uint64_t)np * (uint64_t)K * subpanels, std::memory_order_relaxed);
    g_batch_dequant_vector_ops.fetch_add(
        (uint64_t)np * (uint64_t)K * 4u, std::memory_order_relaxed);
}

void record_packed_a_batch_shape(int rows, int np, int grain, int K) {
    g_batch_kernel_calls.fetch_add(1, std::memory_order_relaxed);
    g_batch_rows_total.fetch_add((uint64_t)rows, std::memory_order_relaxed);
    g_batch_output_panel_tasks.fetch_add(
        (uint64_t)((np + grain - 1) / grain), std::memory_order_relaxed);

    // Packed 路径固定每个 8-row tile 执行两个 8-column subpanel，
    // 包括最后由 Pack A 补零的 tile；统计必须反映真实固定计算量。
    const uint64_t mp =
        (uint64_t)((rows + GPTQ_BATCH_MAX_ROWS - 1) / GPTQ_BATCH_MAX_ROWS);
    const uint64_t subpanels = mp * 2u;
    g_batch_weight_vector_loads.fetch_add(
        (uint64_t)np * (uint64_t)K * subpanels, std::memory_order_relaxed);
    g_batch_dequant_vector_ops.fetch_add(
        (uint64_t)np * (uint64_t)K * 4u, std::memory_order_relaxed);
}

} // namespace

Status pack_gptq_batch_a_f16_neon(
    const fp16_t* x,
    int rows,
    int K,
    fp16_t* packed_x
) {
    if (!x || !packed_x || rows <= 0 || K <= 0) {
        return Status::INVALID_ARGUMENT;
    }
    // 复用已有 8-row FP16 Pack A 布局：[ceil(rows/8), K, 8]。
    // 最后一个不足 8 行的 tile 自动补零，因此微内核仍能安全执行整向量 Load。
    pack_A_f16(x, packed_x, rows, K, K);
    return Status::SUCCESS;
}

Status linear_gptq_int8_batch_packed_a_neon(
    const fp16_t* packed_x,
    int rows,
    const GPTQInt8Weight& w,
    fp16_t* y,
    const fp16_t* bias,
    void*,
    size_t
) {
    if (!packed_x || !y || rows <= 0 ||
        !w.qweight_pack.data || !w.scales_pack.data ||
        w.K <= 0 || w.N <= 0 || w.has_g_idx) {
        return Status::INVALID_ARGUMENT;
    }

    const int np = (w.N + NR_F16 - 1) / NR_F16;
    record_packed_a_batch_shape(rows, np, GPTQ_BATCH_PANEL_GRAIN, w.K);
    if (g_thread_pool == nullptr || g_thread_pool->num_threads() <= 1) {
        return linear_gptq_int8_batch_packed_a_panel_range_neon(
            packed_x, rows, w, y, bias, 0, np);
    }

    std::atomic<int> err_flag(0);
    g_thread_pool->parallel_for(
        0, np, GPTQ_BATCH_PANEL_GRAIN, [&](int panel_begin, int panel_end) {
            Status status = linear_gptq_int8_batch_packed_a_panel_range_neon(
                packed_x, rows, w, y, bias, panel_begin, panel_end);
            if (status != Status::SUCCESS) {
                err_flag.store(static_cast<int>(status), std::memory_order_relaxed);
            }
        });
    const int error = err_flag.load(std::memory_order_relaxed);
    return error == 0 ? Status::SUCCESS : static_cast<Status>(error);
}

Status linear_gptq_int8_batch_neon(
    const fp16_t* x,
    int rows,
    const GPTQInt8Weight& w,
    fp16_t* y,
    const fp16_t* bias,
    void*,
    size_t
) {
    if (!x || !y || rows <= 0 ||
        !w.qweight_pack.data || !w.scales_pack.data ||
        w.K <= 0 || w.N <= 0) {
        return Status::INVALID_ARGUMENT;
    }

    if (rows == 1) {
        return linear_gptq_int8_decode_neon(x, w, y, bias, nullptr, 0);
    }
    if (w.has_g_idx) return Status::INVALID_ARGUMENT;

    const int np = (w.N + NR_F16 - 1) / NR_F16;
    record_batch_shape(rows, np, GPTQ_BATCH_PANEL_GRAIN, w.K);
    Status result = Status::SUCCESS;
    if (g_thread_pool == nullptr || g_thread_pool->num_threads() <= 1) {
        result = linear_gptq_int8_batch_panel_range_neon(x, rows, w, y, bias, 0, np);
    } else {
        std::atomic<int> err_flag(0);
        g_thread_pool->parallel_for(0, np, GPTQ_BATCH_PANEL_GRAIN, [&](int pb, int pe) {
            Status status = linear_gptq_int8_batch_panel_range_neon(
                x, rows, w, y, bias, pb, pe);
            if (status != Status::SUCCESS) {
                err_flag.store(static_cast<int>(status), std::memory_order_relaxed);
            }
        });
        int err = err_flag.load(std::memory_order_relaxed);
        result = err == 0 ? Status::SUCCESS : static_cast<Status>(err);
    }

    if (result == Status::SUCCESS && gptq_batch_env_flag("LLM_GPTQ_BATCH_COMPARE")) {
        float tolerance = gptq_batch_compare_tolerance();
        float max_abs = 0.0f;
        uint64_t mismatches = 0;
        std::vector<fp16_t> reference((size_t)rows * w.N);
        for (int row = 0; row < rows; ++row) {
            Status status = linear_gptq_int8_decode_neon(
                x + (size_t)row * w.K, w,
                reference.data() + (size_t)row * w.N, bias, nullptr, 0);
            if (status != Status::SUCCESS) return status;
        }
        for (size_t i = 0; i < reference.size(); ++i) {
            float diff = std::fabs((float)y[i] - (float)reference[i]);
            max_abs = std::max(max_abs, diff);
            if (diff > tolerance) ++mismatches;
        }
        g_batch_compare_mismatches.fetch_add(mismatches, std::memory_order_relaxed);
        std::cerr << "[GPTQ_BATCH_COMPARE] B=" << rows
                  << " K=" << w.K << " N=" << w.N
                  << " max_abs=" << max_abs
                  << " mismatches=" << mismatches << std::endl;
    }
    return result;
}

static Status fused_gate_up_swiglu_gptq_int8_decode_range_neon(
    const fp16_t* x,
    const GPTQInt8Weight& gate_proj,
    const GPTQInt8Weight& up_proj,
    fp16_t* y,
    int panel_begin,
    int panel_end
) {
    if (!x || !y ||
        !gate_proj.qweight_pack.data || !gate_proj.scales_pack.data ||
        !up_proj.qweight_pack.data || !up_proj.scales_pack.data ||
        gate_proj.K <= 0 || gate_proj.N <= 0 ||
        gate_proj.K != up_proj.K || gate_proj.N != up_proj.N) {
        return Status::INVALID_ARGUMENT;
    }
    if (panel_begin < 0 || panel_end < panel_begin) {
        return Status::INVALID_ARGUMENT;
    }

    const int K = gate_proj.K;
    const int N = gate_proj.N;
    const int K_pad = ((K + 7) / 8) * 8;
    const int np = (N + NR_F16 - 1) / NR_F16;
    if (panel_end > np) panel_end = np;

    const int8_t* g_qpack = gate_proj.qweight_pack.ptr<int8_t>();
    const fp16_t* g_spack = gate_proj.scales_pack.ptr<fp16_t>();
    const int8_t* g_zpack = gate_proj.zeros_pack.data ? gate_proj.zeros_pack.ptr<int8_t>() : nullptr;

    const int8_t* u_qpack = up_proj.qweight_pack.ptr<int8_t>();
    const fp16_t* u_spack = up_proj.scales_pack.ptr<fp16_t>();
    const int8_t* u_zpack = up_proj.zeros_pack.data ? up_proj.zeros_pack.ptr<int8_t>() : nullptr;

    if (!gate_proj.has_g_idx && !up_proj.has_g_idx &&
        gate_proj.group_size > 0 &&
        gate_proj.group_size == up_proj.group_size &&
        gate_proj.num_groups == up_proj.num_groups) {
        for (int panel = panel_begin; panel < panel_end; ++panel) {
            float32x4_t g0 = vdupq_n_f32(0.0f);
            float32x4_t g1 = vdupq_n_f32(0.0f);
            float32x4_t g2 = vdupq_n_f32(0.0f);
            float32x4_t g3 = vdupq_n_f32(0.0f);

            float32x4_t u0 = vdupq_n_f32(0.0f);
            float32x4_t u1 = vdupq_n_f32(0.0f);
            float32x4_t u2 = vdupq_n_f32(0.0f);
            float32x4_t u3 = vdupq_n_f32(0.0f);

            for (int group = 0; group < gate_proj.num_groups; ++group) {
                int k_begin = group * gate_proj.group_size;
                int k_end = std::min(k_begin + gate_proj.group_size, K);
                if (k_begin >= k_end) continue;

                float xsum = 0.0f;
                float32x4_t gsum0 = vdupq_n_f32(0.0f);
                float32x4_t gsum1 = vdupq_n_f32(0.0f);
                float32x4_t gsum2 = vdupq_n_f32(0.0f);
                float32x4_t gsum3 = vdupq_n_f32(0.0f);
                float32x4_t usum0 = vdupq_n_f32(0.0f);
                float32x4_t usum1 = vdupq_n_f32(0.0f);
                float32x4_t usum2 = vdupq_n_f32(0.0f);
                float32x4_t usum3 = vdupq_n_f32(0.0f);

                for (int k = k_begin; k < k_end; ++k) {
                    float xv_scalar = (float)x[k];
                    xsum += xv_scalar;
                    float32x4_t xv = vdupq_n_f32(xv_scalar);

                    const int8_t* q = g_qpack + ((size_t)panel * K_pad + k) * NR_F16;
                    int8x16_t qv = vld1q_s8(q);
                    int16x8_t d0 = vmovl_s8(vget_low_s8(qv));
                    int16x8_t d1 = vmovl_s8(vget_high_s8(qv));

                    gsum0 = vfmaq_f32(gsum0, vcvtq_f32_s32(vmovl_s16(vget_low_s16(d0))), xv);
                    gsum1 = vfmaq_f32(gsum1, vcvtq_f32_s32(vmovl_s16(vget_high_s16(d0))), xv);
                    gsum2 = vfmaq_f32(gsum2, vcvtq_f32_s32(vmovl_s16(vget_low_s16(d1))), xv);
                    gsum3 = vfmaq_f32(gsum3, vcvtq_f32_s32(vmovl_s16(vget_high_s16(d1))), xv);

                    const int8_t* uq = u_qpack + ((size_t)panel * K_pad + k) * NR_F16;
                    int8x16_t uqv = vld1q_s8(uq);
                    int16x8_t ud0 = vmovl_s8(vget_low_s8(uqv));
                    int16x8_t ud1 = vmovl_s8(vget_high_s8(uqv));

                    usum0 = vfmaq_f32(usum0, vcvtq_f32_s32(vmovl_s16(vget_low_s16(ud0))), xv);
                    usum1 = vfmaq_f32(usum1, vcvtq_f32_s32(vmovl_s16(vget_high_s16(ud0))), xv);
                    usum2 = vfmaq_f32(usum2, vcvtq_f32_s32(vmovl_s16(vget_low_s16(ud1))), xv);
                    usum3 = vfmaq_f32(usum3, vcvtq_f32_s32(vmovl_s16(vget_high_s16(ud1))), xv);
                }

                const fp16_t* gs = g_spack + ((size_t)panel * gate_proj.num_groups + group) * NR_F16;
                const int8_t* gz = g_zpack ? g_zpack + ((size_t)panel * gate_proj.num_groups + group) * NR_F16 : nullptr;
                gptq_apply_group_accum_f16(
                    g0, g1, g2, g3,
                    gsum0, gsum1, gsum2, gsum3,
                    gs, gz, xsum);

                const fp16_t* us = u_spack + ((size_t)panel * up_proj.num_groups + group) * NR_F16;
                const int8_t* uz = u_zpack ? u_zpack + ((size_t)panel * up_proj.num_groups + group) * NR_F16 : nullptr;
                gptq_apply_group_accum_f16(
                    u0, u1, u2, u3,
                    usum0, usum1, usum2, usum3,
                    us, uz, xsum);
            }

            // y = silu(gate) * up, computed in FP32 vectors and narrowed once.
            float32x4_t y0 = vmulq_f32(vmulq_f32(g0, sigmoid_neon_f32(g0)), u0);
            float32x4_t y1 = vmulq_f32(vmulq_f32(g1, sigmoid_neon_f32(g1)), u1);
            float32x4_t y2 = vmulq_f32(vmulq_f32(g2, sigmoid_neon_f32(g2)), u2);
            float32x4_t y3 = vmulq_f32(vmulq_f32(g3, sigmoid_neon_f32(g3)), u3);

            int col = panel * NR_F16;
            int actual_n = std::min(NR_F16, N - col);
            if (actual_n == NR_F16) {
                vst1q_f16(y + col, vcombine_f16(vcvt_f16_f32(y0), vcvt_f16_f32(y1)));
                vst1q_f16(y + col + 8, vcombine_f16(vcvt_f16_f32(y2), vcvt_f16_f32(y3)));
            } else {
                fp16_t tmp[NR_F16];
                vst1q_f16(tmp, vcombine_f16(vcvt_f16_f32(y0), vcvt_f16_f32(y1)));
                vst1q_f16(tmp + 8, vcombine_f16(vcvt_f16_f32(y2), vcvt_f16_f32(y3)));
                for (int lane = 0; lane < actual_n; ++lane) {
                    y[col + lane] = tmp[lane];
                }
            }
        }

        return Status::SUCCESS;
    }

    for (int panel = panel_begin; panel < panel_end; ++panel) {
        float32x4_t g0 = vdupq_n_f32(0.0f);
        float32x4_t g1 = vdupq_n_f32(0.0f);
        float32x4_t g2 = vdupq_n_f32(0.0f);
        float32x4_t g3 = vdupq_n_f32(0.0f);

        float32x4_t u0 = vdupq_n_f32(0.0f);
        float32x4_t u1 = vdupq_n_f32(0.0f);
        float32x4_t u2 = vdupq_n_f32(0.0f);
        float32x4_t u3 = vdupq_n_f32(0.0f);

        for (int k = 0; k < K; ++k) {
            float32x4_t xv = vdupq_n_f32((float)x[k]);

            {
                int group = gptq_group_for_k(gate_proj, k);
                const int8_t* q = g_qpack + ((size_t)panel * K_pad + k) * NR_F16;
                const fp16_t* s = g_spack + ((size_t)panel * gate_proj.num_groups + group) * NR_F16;
                const int8_t* z = g_zpack ? g_zpack + ((size_t)panel * gate_proj.num_groups + group) * NR_F16 : nullptr;

                int8x16_t qv = vld1q_s8(q);
                int16x8_t d0 = vmovl_s8(vget_low_s8(qv));
                int16x8_t d1 = vmovl_s8(vget_high_s8(qv));
                if (z) {
                    int8x16_t zv = vld1q_s8(z);
                    d0 = vsubq_s16(d0, vmovl_s8(vget_low_s8(zv)));
                    d1 = vsubq_s16(d1, vmovl_s8(vget_high_s8(zv)));
                }

                float16x8_t sv0 = vld1q_f16(s);
                float16x8_t sv1 = vld1q_f16(s + 8);
                g0 = vfmaq_f32(g0,
                                vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(d0))),
                                          vcvt_f32_f16(vget_low_f16(sv0))),
                                xv);
                g1 = vfmaq_f32(g1,
                                vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(d0))),
                                          vcvt_f32_f16(vget_high_f16(sv0))),
                                xv);
                g2 = vfmaq_f32(g2,
                                vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(d1))),
                                          vcvt_f32_f16(vget_low_f16(sv1))),
                                xv);
                g3 = vfmaq_f32(g3,
                                vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(d1))),
                                          vcvt_f32_f16(vget_high_f16(sv1))),
                                xv);
            }

            {
                int group = gptq_group_for_k(up_proj, k);
                const int8_t* q = u_qpack + ((size_t)panel * K_pad + k) * NR_F16;
                const fp16_t* s = u_spack + ((size_t)panel * up_proj.num_groups + group) * NR_F16;
                const int8_t* z = u_zpack ? u_zpack + ((size_t)panel * up_proj.num_groups + group) * NR_F16 : nullptr;

                int8x16_t qv = vld1q_s8(q);
                int16x8_t d0 = vmovl_s8(vget_low_s8(qv));
                int16x8_t d1 = vmovl_s8(vget_high_s8(qv));
                if (z) {
                    int8x16_t zv = vld1q_s8(z);
                    d0 = vsubq_s16(d0, vmovl_s8(vget_low_s8(zv)));
                    d1 = vsubq_s16(d1, vmovl_s8(vget_high_s8(zv)));
                }

                float16x8_t sv0 = vld1q_f16(s);
                float16x8_t sv1 = vld1q_f16(s + 8);
                u0 = vfmaq_f32(u0,
                                vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(d0))),
                                          vcvt_f32_f16(vget_low_f16(sv0))),
                                xv);
                u1 = vfmaq_f32(u1,
                                vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(d0))),
                                          vcvt_f32_f16(vget_high_f16(sv0))),
                                xv);
                u2 = vfmaq_f32(u2,
                                vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(d1))),
                                          vcvt_f32_f16(vget_low_f16(sv1))),
                                xv);
                u3 = vfmaq_f32(u3,
                                vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(d1))),
                                          vcvt_f32_f16(vget_high_f16(sv1))),
                                xv);
            }
        }

        // y = silu(gate) * up, computed in FP32 vectors and narrowed once.
        float32x4_t y0 = vmulq_f32(vmulq_f32(g0, sigmoid_neon_f32(g0)), u0);
        float32x4_t y1 = vmulq_f32(vmulq_f32(g1, sigmoid_neon_f32(g1)), u1);
        float32x4_t y2 = vmulq_f32(vmulq_f32(g2, sigmoid_neon_f32(g2)), u2);
        float32x4_t y3 = vmulq_f32(vmulq_f32(g3, sigmoid_neon_f32(g3)), u3);

        int col = panel * NR_F16;
        int actual_n = std::min(NR_F16, N - col);
        if (actual_n == NR_F16) {
            vst1q_f16(y + col, vcombine_f16(vcvt_f16_f32(y0), vcvt_f16_f32(y1)));
            vst1q_f16(y + col + 8, vcombine_f16(vcvt_f16_f32(y2), vcvt_f16_f32(y3)));
        } else {
            fp16_t tmp[NR_F16];
            vst1q_f16(tmp, vcombine_f16(vcvt_f16_f32(y0), vcvt_f16_f32(y1)));
            vst1q_f16(tmp + 8, vcombine_f16(vcvt_f16_f32(y2), vcvt_f16_f32(y3)));
            for (int lane = 0; lane < actual_n; ++lane) {
                y[col + lane] = tmp[lane];
            }
        }
    }

    return Status::SUCCESS;
}

Status fused_gate_up_swiglu_gptq_int8_decode_neon(
    const fp16_t* x,
    const GPTQInt8Weight& gate_proj,
    const GPTQInt8Weight& up_proj,
    fp16_t* y,
    void*,
    size_t
) {
    if (!x || !y ||
        !gate_proj.qweight_pack.data || !gate_proj.scales_pack.data ||
        !up_proj.qweight_pack.data || !up_proj.scales_pack.data ||
        gate_proj.K <= 0 || gate_proj.N <= 0 ||
        gate_proj.K != up_proj.K || gate_proj.N != up_proj.N) {
        return Status::INVALID_ARGUMENT;
    }

    int np = (gate_proj.N + NR_F16 - 1) / NR_F16;
    if (g_thread_pool == nullptr ||
        g_thread_pool->num_threads() <= 1 ||
        gate_proj.N < GPTQ_FUSED_PARALLEL_N_THRESHOLD) {
        return fused_gate_up_swiglu_gptq_int8_decode_range_neon(
            x, gate_proj, up_proj, y, 0, np);
    }

    std::atomic<int> err_flag(0);
    g_thread_pool->parallel_for(0, np, GPTQ_PARALLEL_GRAIN_PANELS, [&](int pb, int pe) {
        Status s = fused_gate_up_swiglu_gptq_int8_decode_range_neon(
            x, gate_proj, up_proj, y, pb, pe);
        if (s != Status::SUCCESS) {
            err_flag.store(static_cast<int>(s), std::memory_order_relaxed);
        }
    });

    int err = err_flag.load(std::memory_order_relaxed);
    return err == 0 ? Status::SUCCESS : static_cast<Status>(err);
}

Status fused_gate_up_swiglu_gptq_int8_batch_neon(
    const fp16_t* x,
    int rows,
    const GPTQInt8Weight& gate_proj,
    const GPTQInt8Weight& up_proj,
    fp16_t* y,
    void* workspace,
    size_t workspace_bytes
) {
    if (!x || !y || rows <= 0 ||
        !gate_proj.qweight_pack.data || !gate_proj.scales_pack.data ||
        !up_proj.qweight_pack.data || !up_proj.scales_pack.data ||
        gate_proj.K <= 0 || gate_proj.N <= 0 ||
        gate_proj.K != up_proj.K || gate_proj.N != up_proj.N) {
        return Status::INVALID_ARGUMENT;
    }

    if (rows == 1) {
        return fused_gate_up_swiglu_gptq_int8_decode_neon(
            x, gate_proj, up_proj, y, nullptr, 0);
    }

    size_t up_bytes = (size_t)rows * gate_proj.N * sizeof(fp16_t);
    if (!workspace || workspace_bytes < up_bytes) return Status::OUT_OF_MEMORY;
    fp16_t* up = static_cast<fp16_t*>(workspace);
    Status status = linear_gptq_int8_batch_neon(
        x, rows, gate_proj, y, nullptr, nullptr, 0);
    if (status != Status::SUCCESS) return status;
    status = linear_gptq_int8_batch_neon(
        x, rows, up_proj, up, nullptr, nullptr, 0);
    if (status != Status::SUCCESS) return status;
    swiglu_f16_batch_neon(y, up, rows, gate_proj.N);
    return Status::SUCCESS;
}

struct GptqArgmaxDebugStats {
    int x_nan = 0;
    int scale_nan = 0;
    int logit_nan = 0;
    int logit_finite = 0;
};

static inline void gptq_argmax_scan_panel(
    int panel,
    int N,
    float32x4_t acc0,
    float32x4_t acc1,
    float32x4_t acc2,
    float32x4_t acc3,
    bool debug_numeric,
    int& logit_nan,
    int& logit_finite,
    ArgmaxResult& result
) {
    int col = panel * NR_F16;
    int actual_n = std::min(NR_F16, N - col);
    float logits[NR_F16];
    vst1q_f32(logits, acc0);
    vst1q_f32(logits + 4, acc1);
    vst1q_f32(logits + 8, acc2);
    vst1q_f32(logits + 12, acc3);

    for (int lane = 0; lane < actual_n; ++lane) {
        float v = logits[lane];
        if (debug_numeric) {
            if (std::isnan(v)) logit_nan++;
            else if (std::isfinite(v)) logit_finite++;
        }
        int index = col + lane;
        if (v > result.value || (v == result.value && (result.index < 0 || index < result.index))) {
            result.value = v;
            result.index = index;
        }
    }
}

static ArgmaxResult linear_gptq_int8_decode_argmax_range_neon(
    const fp16_t* x,
    const GPTQInt8Weight& w,
    int panel_begin,
    int panel_end,
    bool debug_numeric,
    GptqArgmaxDebugStats* stats
) {
    ArgmaxResult result{-1, -std::numeric_limits<float>::infinity()};
    if (!x || !w.qweight_pack.data || !w.scales_pack.data || w.K <= 0 || w.N <= 0) {
        return result;
    }
    if (panel_begin < 0 || panel_end < panel_begin) {
        return result;
    }

    const int K_pad = ((w.K + 7) / 8) * 8;
    const int np = (w.N + NR_F16 - 1) / NR_F16;
    if (panel_end > np) panel_end = np;
    const int8_t* qpack = w.qweight_pack.ptr<int8_t>();
    const fp16_t* spack = w.scales_pack.ptr<fp16_t>();
    const int8_t* zpack = w.zeros_pack.data ? w.zeros_pack.ptr<int8_t>() : nullptr;
    int x_nan = 0;
    int scale_nan = 0;
    int logit_nan = 0;
    int logit_finite = 0;

    if (!w.has_g_idx && w.group_size > 0) {
        int panel = panel_begin;
        for (; panel + 1 < panel_end; panel += 2) {
            float32x4_t p0_acc0 = vdupq_n_f32(0.0f);
            float32x4_t p0_acc1 = vdupq_n_f32(0.0f);
            float32x4_t p0_acc2 = vdupq_n_f32(0.0f);
            float32x4_t p0_acc3 = vdupq_n_f32(0.0f);
            float32x4_t p1_acc0 = vdupq_n_f32(0.0f);
            float32x4_t p1_acc1 = vdupq_n_f32(0.0f);
            float32x4_t p1_acc2 = vdupq_n_f32(0.0f);
            float32x4_t p1_acc3 = vdupq_n_f32(0.0f);

            int panel1 = panel + 1;
            for (int group = 0; group < w.num_groups; ++group) {
                int k_begin = group * w.group_size;
                int k_end = std::min(k_begin + w.group_size, w.K);
                if (k_begin >= k_end) continue;

                float xsum = 0.0f;
                float32x4_t p0_sum0 = vdupq_n_f32(0.0f);
                float32x4_t p0_sum1 = vdupq_n_f32(0.0f);
                float32x4_t p0_sum2 = vdupq_n_f32(0.0f);
                float32x4_t p0_sum3 = vdupq_n_f32(0.0f);
                float32x4_t p1_sum0 = vdupq_n_f32(0.0f);
                float32x4_t p1_sum1 = vdupq_n_f32(0.0f);
                float32x4_t p1_sum2 = vdupq_n_f32(0.0f);
                float32x4_t p1_sum3 = vdupq_n_f32(0.0f);

                for (int k = k_begin; k < k_end; ++k) {
                    float xv_scalar = (float)x[k];
                    if (debug_numeric && std::isnan(xv_scalar)) x_nan += 2;
                    xsum += xv_scalar;
                    float32x4_t xv = vdupq_n_f32(xv_scalar);

                    {
                        const int8_t* q = qpack + ((size_t)panel * K_pad + k) * NR_F16;
                        int8x16_t qv = vld1q_s8(q);
                        int16x8_t d0 = vmovl_s8(vget_low_s8(qv));
                        int16x8_t d1 = vmovl_s8(vget_high_s8(qv));

                        p0_sum0 = vfmaq_f32(p0_sum0, vcvtq_f32_s32(vmovl_s16(vget_low_s16(d0))), xv);
                        p0_sum1 = vfmaq_f32(p0_sum1, vcvtq_f32_s32(vmovl_s16(vget_high_s16(d0))), xv);
                        p0_sum2 = vfmaq_f32(p0_sum2, vcvtq_f32_s32(vmovl_s16(vget_low_s16(d1))), xv);
                        p0_sum3 = vfmaq_f32(p0_sum3, vcvtq_f32_s32(vmovl_s16(vget_high_s16(d1))), xv);
                    }

                    {
                        const int8_t* q = qpack + ((size_t)panel1 * K_pad + k) * NR_F16;
                        int8x16_t qv = vld1q_s8(q);
                        int16x8_t d0 = vmovl_s8(vget_low_s8(qv));
                        int16x8_t d1 = vmovl_s8(vget_high_s8(qv));

                        p1_sum0 = vfmaq_f32(p1_sum0, vcvtq_f32_s32(vmovl_s16(vget_low_s16(d0))), xv);
                        p1_sum1 = vfmaq_f32(p1_sum1, vcvtq_f32_s32(vmovl_s16(vget_high_s16(d0))), xv);
                        p1_sum2 = vfmaq_f32(p1_sum2, vcvtq_f32_s32(vmovl_s16(vget_low_s16(d1))), xv);
                        p1_sum3 = vfmaq_f32(p1_sum3, vcvtq_f32_s32(vmovl_s16(vget_high_s16(d1))), xv);
                    }
                }

                const fp16_t* p0_s = spack + ((size_t)panel * w.num_groups + group) * NR_F16;
                const int8_t* p0_z = zpack ? zpack + ((size_t)panel * w.num_groups + group) * NR_F16 : nullptr;
                if (debug_numeric) {
                    for (int lane = 0; lane < NR_F16; ++lane) {
                        if (std::isnan((float)p0_s[lane])) scale_nan++;
                    }
                }
                gptq_apply_group_accum_f16(
                    p0_acc0, p0_acc1, p0_acc2, p0_acc3,
                    p0_sum0, p0_sum1, p0_sum2, p0_sum3,
                    p0_s, p0_z, xsum);

                const fp16_t* p1_s = spack + ((size_t)panel1 * w.num_groups + group) * NR_F16;
                const int8_t* p1_z = zpack ? zpack + ((size_t)panel1 * w.num_groups + group) * NR_F16 : nullptr;
                if (debug_numeric) {
                    for (int lane = 0; lane < NR_F16; ++lane) {
                        if (std::isnan((float)p1_s[lane])) scale_nan++;
                    }
                }
                gptq_apply_group_accum_f16(
                    p1_acc0, p1_acc1, p1_acc2, p1_acc3,
                    p1_sum0, p1_sum1, p1_sum2, p1_sum3,
                    p1_s, p1_z, xsum);
            }

            gptq_argmax_scan_panel(
                panel, w.N, p0_acc0, p0_acc1, p0_acc2, p0_acc3,
                debug_numeric, logit_nan, logit_finite, result);
            gptq_argmax_scan_panel(
                panel1, w.N, p1_acc0, p1_acc1, p1_acc2, p1_acc3,
                debug_numeric, logit_nan, logit_finite, result);
        }

        for (; panel < panel_end; ++panel) {
            float32x4_t acc0 = vdupq_n_f32(0.0f);
            float32x4_t acc1 = vdupq_n_f32(0.0f);
            float32x4_t acc2 = vdupq_n_f32(0.0f);
            float32x4_t acc3 = vdupq_n_f32(0.0f);

            for (int group = 0; group < w.num_groups; ++group) {
                int k_begin = group * w.group_size;
                int k_end = std::min(k_begin + w.group_size, w.K);
                if (k_begin >= k_end) continue;

                float xsum = 0.0f;
                float32x4_t sum0 = vdupq_n_f32(0.0f);
                float32x4_t sum1 = vdupq_n_f32(0.0f);
                float32x4_t sum2 = vdupq_n_f32(0.0f);
                float32x4_t sum3 = vdupq_n_f32(0.0f);

                for (int k = k_begin; k < k_end; ++k) {
                    float xv_scalar = (float)x[k];
                    if (debug_numeric && std::isnan(xv_scalar)) x_nan++;
                    xsum += xv_scalar;
                    float32x4_t xv = vdupq_n_f32(xv_scalar);

                    const int8_t* q = qpack + ((size_t)panel * K_pad + k) * NR_F16;
                    int8x16_t qv = vld1q_s8(q);
                    int16x8_t d0 = vmovl_s8(vget_low_s8(qv));
                    int16x8_t d1 = vmovl_s8(vget_high_s8(qv));

                    sum0 = vfmaq_f32(sum0, vcvtq_f32_s32(vmovl_s16(vget_low_s16(d0))), xv);
                    sum1 = vfmaq_f32(sum1, vcvtq_f32_s32(vmovl_s16(vget_high_s16(d0))), xv);
                    sum2 = vfmaq_f32(sum2, vcvtq_f32_s32(vmovl_s16(vget_low_s16(d1))), xv);
                    sum3 = vfmaq_f32(sum3, vcvtq_f32_s32(vmovl_s16(vget_high_s16(d1))), xv);
                }

                const fp16_t* s = spack + ((size_t)panel * w.num_groups + group) * NR_F16;
                const int8_t* z = zpack ? zpack + ((size_t)panel * w.num_groups + group) * NR_F16 : nullptr;
                if (debug_numeric) {
                    for (int lane = 0; lane < NR_F16; ++lane) {
                        if (std::isnan((float)s[lane])) scale_nan++;
                    }
                }
                gptq_apply_group_accum_f16(
                    acc0, acc1, acc2, acc3,
                    sum0, sum1, sum2, sum3,
                    s, z, xsum);
            }

            gptq_argmax_scan_panel(
                panel, w.N, acc0, acc1, acc2, acc3,
                debug_numeric, logit_nan, logit_finite, result);
        }

        if (debug_numeric && stats) {
            stats->x_nan += x_nan;
            stats->scale_nan += scale_nan;
            stats->logit_nan += logit_nan;
            stats->logit_finite += logit_finite;
        }

        return result;
    }

    int panel = panel_begin;
    for (; panel + 1 < panel_end; panel += 2) {
        float32x4_t p0_acc0 = vdupq_n_f32(0.0f);
        float32x4_t p0_acc1 = vdupq_n_f32(0.0f);
        float32x4_t p0_acc2 = vdupq_n_f32(0.0f);
        float32x4_t p0_acc3 = vdupq_n_f32(0.0f);
        float32x4_t p1_acc0 = vdupq_n_f32(0.0f);
        float32x4_t p1_acc1 = vdupq_n_f32(0.0f);
        float32x4_t p1_acc2 = vdupq_n_f32(0.0f);
        float32x4_t p1_acc3 = vdupq_n_f32(0.0f);

        for (int k = 0; k < w.K; ++k) {
            int group = gptq_group_for_k(w, k);
            float xv = (float)x[k];
            if (debug_numeric && std::isnan(xv)) x_nan += 2;
            float32x4_t xvv = vdupq_n_f32(xv);

            {
                const int8_t* q = qpack + ((size_t)panel * K_pad + k) * NR_F16;
                const fp16_t* s = spack + ((size_t)panel * w.num_groups + group) * NR_F16;
                const int8_t* z = zpack ? zpack + ((size_t)panel * w.num_groups + group) * NR_F16 : nullptr;

                int8x16_t qv = vld1q_s8(q);
                int16x8_t d0 = vmovl_s8(vget_low_s8(qv));
                int16x8_t d1 = vmovl_s8(vget_high_s8(qv));
                if (z) {
                    int8x16_t zv = vld1q_s8(z);
                    d0 = vsubq_s16(d0, vmovl_s8(vget_low_s8(zv)));
                    d1 = vsubq_s16(d1, vmovl_s8(vget_high_s8(zv)));
                }

                float16x8_t sv0 = vld1q_f16(s);
                float16x8_t sv1 = vld1q_f16(s + 8);
                if (debug_numeric) {
                    for (int lane = 0; lane < NR_F16; ++lane) {
                        if (std::isnan((float)s[lane])) scale_nan++;
                    }
                }

                p0_acc0 = vfmaq_f32(p0_acc0,
                                     vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(d0))),
                                               vcvt_f32_f16(vget_low_f16(sv0))),
                                     xvv);
                p0_acc1 = vfmaq_f32(p0_acc1,
                                     vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(d0))),
                                               vcvt_f32_f16(vget_high_f16(sv0))),
                                     xvv);
                p0_acc2 = vfmaq_f32(p0_acc2,
                                     vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(d1))),
                                               vcvt_f32_f16(vget_low_f16(sv1))),
                                     xvv);
                p0_acc3 = vfmaq_f32(p0_acc3,
                                     vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(d1))),
                                               vcvt_f32_f16(vget_high_f16(sv1))),
                                     xvv);
            }

            {
                int panel1 = panel + 1;
                const int8_t* q = qpack + ((size_t)panel1 * K_pad + k) * NR_F16;
                const fp16_t* s = spack + ((size_t)panel1 * w.num_groups + group) * NR_F16;
                const int8_t* z = zpack ? zpack + ((size_t)panel1 * w.num_groups + group) * NR_F16 : nullptr;

                int8x16_t qv = vld1q_s8(q);
                int16x8_t d0 = vmovl_s8(vget_low_s8(qv));
                int16x8_t d1 = vmovl_s8(vget_high_s8(qv));
                if (z) {
                    int8x16_t zv = vld1q_s8(z);
                    d0 = vsubq_s16(d0, vmovl_s8(vget_low_s8(zv)));
                    d1 = vsubq_s16(d1, vmovl_s8(vget_high_s8(zv)));
                }

                float16x8_t sv0 = vld1q_f16(s);
                float16x8_t sv1 = vld1q_f16(s + 8);
                if (debug_numeric) {
                    for (int lane = 0; lane < NR_F16; ++lane) {
                        if (std::isnan((float)s[lane])) scale_nan++;
                    }
                }

                p1_acc0 = vfmaq_f32(p1_acc0,
                                     vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(d0))),
                                               vcvt_f32_f16(vget_low_f16(sv0))),
                                     xvv);
                p1_acc1 = vfmaq_f32(p1_acc1,
                                     vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(d0))),
                                               vcvt_f32_f16(vget_high_f16(sv0))),
                                     xvv);
                p1_acc2 = vfmaq_f32(p1_acc2,
                                     vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(d1))),
                                               vcvt_f32_f16(vget_low_f16(sv1))),
                                     xvv);
                p1_acc3 = vfmaq_f32(p1_acc3,
                                     vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(d1))),
                                               vcvt_f32_f16(vget_high_f16(sv1))),
                                     xvv);
            }
        }

        gptq_argmax_scan_panel(
            panel, w.N, p0_acc0, p0_acc1, p0_acc2, p0_acc3,
            debug_numeric, logit_nan, logit_finite, result);
        gptq_argmax_scan_panel(
            panel + 1, w.N, p1_acc0, p1_acc1, p1_acc2, p1_acc3,
            debug_numeric, logit_nan, logit_finite, result);
    }

    for (; panel < panel_end; ++panel) {
        float32x4_t acc0 = vdupq_n_f32(0.0f);
        float32x4_t acc1 = vdupq_n_f32(0.0f);
        float32x4_t acc2 = vdupq_n_f32(0.0f);
        float32x4_t acc3 = vdupq_n_f32(0.0f);

        for (int k = 0; k < w.K; ++k) {
            int group = gptq_group_for_k(w, k);
            const int8_t* q = qpack + ((size_t)panel * K_pad + k) * NR_F16;
            const fp16_t* s = spack + ((size_t)panel * w.num_groups + group) * NR_F16;
            const int8_t* z = zpack ? zpack + ((size_t)panel * w.num_groups + group) * NR_F16 : nullptr;

            float xv = (float)x[k];
            if (debug_numeric && std::isnan(xv)) x_nan++;
            int8x16_t qv = vld1q_s8(q);
            int16x8_t d0 = vmovl_s8(vget_low_s8(qv));
            int16x8_t d1 = vmovl_s8(vget_high_s8(qv));
            if (z) {
                int8x16_t zv = vld1q_s8(z);
                d0 = vsubq_s16(d0, vmovl_s8(vget_low_s8(zv)));
                d1 = vsubq_s16(d1, vmovl_s8(vget_high_s8(zv)));
            }

            float16x8_t sv0 = vld1q_f16(s);
            float16x8_t sv1 = vld1q_f16(s + 8);
            if (debug_numeric) {
                for (int lane = 0; lane < NR_F16; ++lane) {
                    if (std::isnan((float)s[lane])) scale_nan++;
                }
            }

            float32x4_t xvv = vdupq_n_f32(xv);
            float32x4_t q0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(d0)));
            float32x4_t q1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(d0)));
            float32x4_t q2 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(d1)));
            float32x4_t q3 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(d1)));

            float32x4_t s0 = vcvt_f32_f16(vget_low_f16(sv0));
            float32x4_t s1 = vcvt_f32_f16(vget_high_f16(sv0));
            float32x4_t s2 = vcvt_f32_f16(vget_low_f16(sv1));
            float32x4_t s3 = vcvt_f32_f16(vget_high_f16(sv1));

            acc0 = vfmaq_f32(acc0, vmulq_f32(q0, s0), xvv);
            acc1 = vfmaq_f32(acc1, vmulq_f32(q1, s1), xvv);
            acc2 = vfmaq_f32(acc2, vmulq_f32(q2, s2), xvv);
            acc3 = vfmaq_f32(acc3, vmulq_f32(q3, s3), xvv);
        }

        gptq_argmax_scan_panel(
            panel, w.N, acc0, acc1, acc2, acc3,
            debug_numeric, logit_nan, logit_finite, result);
    }

    if (debug_numeric && stats) {
        stats->x_nan += x_nan;
        stats->scale_nan += scale_nan;
        stats->logit_nan += logit_nan;
        stats->logit_finite += logit_finite;
    }

    return result;
}

ArgmaxResult linear_gptq_int8_decode_argmax_neon(
    const fp16_t* x,
    const GPTQInt8Weight& w,
    void*,
    size_t
) {
    ArgmaxResult result{-1, -std::numeric_limits<float>::infinity()};
    if (!x || !w.qweight_pack.data || !w.scales_pack.data || w.K <= 0 || w.N <= 0) {
        return result;
    }

    const int np = (w.N + NR_F16 - 1) / NR_F16;
    bool debug_numeric = debug_numeric_enabled();
    GptqArgmaxDebugStats total_stats;

    if (g_thread_pool == nullptr ||
        g_thread_pool->num_threads() <= 1 ||
        w.N < GPTQ_PARALLEL_N_THRESHOLD) {
        result = linear_gptq_int8_decode_argmax_range_neon(
            x, w, 0, np, debug_numeric, &total_stats);
    } else {
        int n_threads = g_thread_pool->num_threads();
        int max_chunks = (np + GPTQ_PARALLEL_GRAIN_PANELS - 1) / GPTQ_PARALLEL_GRAIN_PANELS;
        int n_chunks = std::min(n_threads, max_chunks);
        if (n_chunks <= 1) {
            result = linear_gptq_int8_decode_argmax_range_neon(
                x, w, 0, np, debug_numeric, &total_stats);
        } else {
            std::vector<ArgmaxResult> partials(
                n_chunks, ArgmaxResult{-1, -std::numeric_limits<float>::infinity()});
            std::vector<GptqArgmaxDebugStats> partial_stats(n_chunks);
            std::vector<std::pair<int, int>> ranges;
            ranges.reserve(n_chunks);

            int base = np / n_chunks;
            int rem = np % n_chunks;
            int cursor = 0;
            for (int i = 0; i < n_chunks; ++i) {
                int len = base + (i < rem ? 1 : 0);
                ranges.push_back({cursor, cursor + len});
                cursor += len;
            }

            g_thread_pool->parallel_for(0, n_chunks, 1, [&](int sb, int se) {
                for (int slot = sb; slot < se; ++slot) {
                    partials[slot] = linear_gptq_int8_decode_argmax_range_neon(
                        x, w,
                        ranges[slot].first,
                        ranges[slot].second,
                        debug_numeric,
                        &partial_stats[slot]);
                }
            });

            for (int i = 0; i < n_chunks; ++i) {
                const auto& p = partials[i];
                if (p.index >= 0 &&
                    (p.value > result.value ||
                     (p.value == result.value && (result.index < 0 || p.index < result.index)))) {
                    result = p;
                }
                if (debug_numeric) {
                    total_stats.x_nan += partial_stats[i].x_nan;
                    total_stats.scale_nan += partial_stats[i].scale_nan;
                    total_stats.logit_nan += partial_stats[i].logit_nan;
                    total_stats.logit_finite += partial_stats[i].logit_finite;
                }
            }
        }
    }

    if (debug_numeric) {
        std::cerr << "[NUMERIC] lm_head_argmax"
                  << " x_nan=" << total_stats.x_nan
                  << " scale_nan=" << total_stats.scale_nan
                  << " logit_nan=" << total_stats.logit_nan
                  << " logit_finite=" << total_stats.logit_finite
                  << " result_index=" << result.index
                  << " result_value=" << result.value
                  << std::endl;
    }

    return result;
}

size_t linear_gptq_int8_batch_argmax_workspace_bytes(
    int rows,
    const GPTQInt8Weight& w
) {
    if (rows <= 1 || rows > GPTQ_BATCH_ARGMAX_MAX_ROWS || w.N <= 0) return 0;
    int np = (w.N + NR_F16 - 1) / NR_F16;
    int chunks = (np + GPTQ_BATCH_ARGMAX_GRAIN - 1) / GPTQ_BATCH_ARGMAX_GRAIN;
    return (size_t)chunks * rows * sizeof(ArgmaxResult);
}

Status linear_gptq_int8_decode_argmax_batch_neon(
    const fp16_t* x,
    int rows,
    const GPTQInt8Weight& w,
    ArgmaxResult* results,
    void* workspace,
    size_t workspace_bytes
) {
    if (!x || !results || rows <= 0 || rows > GPTQ_BATCH_ARGMAX_MAX_ROWS ||
        !w.qweight_pack.data || !w.scales_pack.data || w.K <= 0 || w.N <= 0 ||
        w.has_g_idx) {
        return Status::INVALID_ARGUMENT;
    }
    if (rows == 1) {
        results[0] = linear_gptq_int8_decode_argmax_neon(x, w, nullptr, 0);
        return results[0].index >= 0 ? Status::SUCCESS : Status::INVALID_ARGUMENT;
    }

    int np = (w.N + NR_F16 - 1) / NR_F16;
    int chunks = (np + GPTQ_BATCH_ARGMAX_GRAIN - 1) / GPTQ_BATCH_ARGMAX_GRAIN;
    size_t required = (size_t)chunks * rows * sizeof(ArgmaxResult);
    if (!workspace || workspace_bytes < required) return Status::OUT_OF_MEMORY;
    ArgmaxResult* partials = static_cast<ArgmaxResult*>(workspace);

    g_batch_argmax_calls.fetch_add(1, std::memory_order_relaxed);
    g_batch_argmax_rows.fetch_add((uint64_t)rows, std::memory_order_relaxed);
    record_batch_shape(rows, np, GPTQ_BATCH_ARGMAX_GRAIN, w.K);

    std::atomic<int> err_flag(0);
    auto run_chunks = [&](int begin, int end) {
        for (int chunk = begin; chunk < end; ++chunk) {
            int pb = chunk * GPTQ_BATCH_ARGMAX_GRAIN;
            int pe = std::min(pb + GPTQ_BATCH_ARGMAX_GRAIN, np);
            Status status = gptq_batch_argmax_panel_range_neon(
                x, rows, w, pb, pe, partials + (size_t)chunk * rows);
            if (status != Status::SUCCESS) {
                err_flag.store(static_cast<int>(status), std::memory_order_relaxed);
            }
        }
    };
    if (g_thread_pool == nullptr || g_thread_pool->num_threads() <= 1 || chunks <= 1) {
        run_chunks(0, chunks);
    } else {
        g_thread_pool->parallel_for(0, chunks, 1, run_chunks);
    }
    int err = err_flag.load(std::memory_order_relaxed);
    if (err != 0) return static_cast<Status>(err);

    for (int row = 0; row < rows; ++row) {
        ArgmaxResult best{-1, -std::numeric_limits<float>::infinity()};
        for (int chunk = 0; chunk < chunks; ++chunk) {
            const ArgmaxResult& candidate = partials[(size_t)chunk * rows + row];
            if (candidate.index >= 0 &&
                (candidate.value > best.value ||
                 (candidate.value == best.value &&
                  (best.index < 0 || candidate.index < best.index)))) {
                best = candidate;
            }
        }
        results[row] = best;
        if (best.index < 0) return Status::INVALID_ARGUMENT;
    }

    if (gptq_batch_env_flag("LLM_GPTQ_BATCH_COMPARE")) {
        uint64_t mismatches = 0;
        for (int row = 0; row < rows; ++row) {
            ArgmaxResult reference = linear_gptq_int8_decode_argmax_neon(
                x + (size_t)row * w.K, w, nullptr, 0);
            bool match = reference.index == results[row].index;
            if (!match) ++mismatches;
            std::cerr << "[GPTQ_BATCH_ARGMAX_COMPARE] B=" << rows
                      << " row=" << row
                      << " token=" << results[row].index
                      << " ref_token=" << reference.index
                      << " value=" << results[row].value
                      << " ref_value=" << reference.value
                      << " match=" << (match ? 1 : 0) << std::endl;
        }
        g_batch_compare_mismatches.fetch_add(mismatches, std::memory_order_relaxed);
    }
    return Status::SUCCESS;
}

GPTQBatchKernelStats snapshot_gptq_batch_kernel_stats() {
    GPTQBatchKernelStats stats;
    stats.kernel_calls = g_batch_kernel_calls.load(std::memory_order_relaxed);
    stats.rows_total = g_batch_rows_total.load(std::memory_order_relaxed);
    stats.output_panel_tasks = g_batch_output_panel_tasks.load(std::memory_order_relaxed);
    stats.row_gemv_fallbacks = g_batch_row_gemv_fallbacks.load(std::memory_order_relaxed);
    stats.weight_vector_loads = g_batch_weight_vector_loads.load(std::memory_order_relaxed);
    stats.dequant_vector_ops = g_batch_dequant_vector_ops.load(std::memory_order_relaxed);
    stats.argmax_calls = g_batch_argmax_calls.load(std::memory_order_relaxed);
    stats.argmax_rows = g_batch_argmax_rows.load(std::memory_order_relaxed);
    stats.full_logits_elements_written =
        g_batch_full_logits_elements_written.load(std::memory_order_relaxed);
    stats.compare_mismatches = g_batch_compare_mismatches.load(std::memory_order_relaxed);
    return stats;
}

GPTQBatchKernelStats diff_gptq_batch_kernel_stats(
    const GPTQBatchKernelStats& begin,
    const GPTQBatchKernelStats& end
) {
    GPTQBatchKernelStats result;
    result.kernel_calls = end.kernel_calls - begin.kernel_calls;
    result.rows_total = end.rows_total - begin.rows_total;
    result.output_panel_tasks = end.output_panel_tasks - begin.output_panel_tasks;
    result.row_gemv_fallbacks = end.row_gemv_fallbacks - begin.row_gemv_fallbacks;
    result.weight_vector_loads = end.weight_vector_loads - begin.weight_vector_loads;
    result.dequant_vector_ops = end.dequant_vector_ops - begin.dequant_vector_ops;
    result.argmax_calls = end.argmax_calls - begin.argmax_calls;
    result.argmax_rows = end.argmax_rows - begin.argmax_rows;
    result.full_logits_elements_written =
        end.full_logits_elements_written - begin.full_logits_elements_written;
    result.compare_mismatches = end.compare_mismatches - begin.compare_mismatches;
    return result;
}

// =============================================================================
// FFN 专用 fused kernel：gate + up + SwiGLU 在一个 panel 循环里完成
// y[col + i] = silu(gate[i]) * up[i]，silu(x) = x / (1 + exp(-x))
//
// 一次扫过权重就能写出最终 intermediate，省掉 gate / up 两个完整中间向量
// 的写回 + 读回；同时 K 维只读一遍 x[k]，gate/up 各一行 NEON load。
// 第一版 swiglu 用标量 std::exp，保证正确性，后续可换近似 sigmoid。
// =============================================================================
Status fused_gate_up_swiglu_prepacked_range_neon(
    const float* x,
    const float* w_gate_pack,
    const float* w_up_pack,
    float* y,
    int K,
    int N,
    int panel_begin,
    int panel_end
) {
    if (!x || !w_gate_pack || !w_up_pack || !y || K <= 0 || N <= 0) {
        return Status::INVALID_ARGUMENT;
    }
    if (panel_begin < 0 || panel_end < panel_begin) {
        return Status::INVALID_ARGUMENT;
    }

    int np = (N + NR - 1) / NR;
    if (panel_end > np) panel_end = np;

    for (int panel = panel_begin; panel < panel_end; ++panel) {
        const float* wg = w_gate_pack + (size_t)panel * K * NR;
        const float* wu = w_up_pack   + (size_t)panel * K * NR;

        float32x4_t g0 = vdupq_n_f32(0.0f);
        float32x4_t g1 = vdupq_n_f32(0.0f);
        float32x4_t g2 = vdupq_n_f32(0.0f);
        float32x4_t u0 = vdupq_n_f32(0.0f);
        float32x4_t u1 = vdupq_n_f32(0.0f);
        float32x4_t u2 = vdupq_n_f32(0.0f);

        for (int k = 0; k < K; ++k) {
            const float* gr = wg + k * NR;
            const float* ur = wu + k * NR;
            float xv = x[k];

            g0 = vfmaq_n_f32(g0, vld1q_f32(gr),     xv);
            g1 = vfmaq_n_f32(g1, vld1q_f32(gr + 4), xv);
            g2 = vfmaq_n_f32(g2, vld1q_f32(gr + 8), xv);

            u0 = vfmaq_n_f32(u0, vld1q_f32(ur),     xv);
            u1 = vfmaq_n_f32(u1, vld1q_f32(ur + 4), xv);
            u2 = vfmaq_n_f32(u2, vld1q_f32(ur + 8), xv);
        }

        int col = panel * NR;
        int actual_n = std::min(NR, N - col);

        float gate_tmp[NR];
        float up_tmp[NR];
        vst1q_f32(gate_tmp,     g0);
        vst1q_f32(gate_tmp + 4, g1);
        vst1q_f32(gate_tmp + 8, g2);
        vst1q_f32(up_tmp,       u0);
        vst1q_f32(up_tmp + 4,   u1);
        vst1q_f32(up_tmp + 8,   u2);

        for (int i = 0; i < actual_n; ++i) {
            float gv = gate_tmp[i];
            float uv = up_tmp[i];
            float silu = gv / (1.0f + std::exp(-gv));
            y[col + i] = silu * uv;
        }
    }

    return Status::SUCCESS;
}

Status fused_gate_up_swiglu_prepacked_parallel_neon(
    const float* x,
    const float* w_gate_pack,
    const float* w_up_pack,
    float* y,
    int K,
    int N
) {
    if (!x || !w_gate_pack || !w_up_pack || !y || K <= 0 || N <= 0) {
        return Status::INVALID_ARGUMENT;
    }

    int np = (N + NR - 1) / NR;

    // 串行 fallback：线程池不可用、单线程、或 N 太小
    if (g_thread_pool == nullptr ||
        g_thread_pool->num_threads() <= 1 ||
        N < 1024) {
        return fused_gate_up_swiglu_prepacked_range_neon(
            x, w_gate_pack, w_up_pack, y, K, N, 0, np
        );
    }

    constexpr int grain = 16; // 16 panels = 192 intermediate channels

    std::atomic<int> err_flag(0);
    g_thread_pool->parallel_for(0, np, grain, [&](int pb, int pe) {
        Status s = fused_gate_up_swiglu_prepacked_range_neon(
            x, w_gate_pack, w_up_pack, y, K, N, pb, pe
        );
        if (s != Status::SUCCESS) {
            err_flag.store(static_cast<int>(s), std::memory_order_relaxed);
        }
    });

    int err = err_flag.load(std::memory_order_relaxed);
    return err == 0 ? Status::SUCCESS : static_cast<Status>(err);
}

} // namespace arm_neon
} // namespace llm_engine
