#include "backends/cpu/arm_neon/neon_ops.h"
#include "backends/cpu/arm_neon/kernel_common.h"
#include "llm_engine/runtime/thread_pool.h"
#include <algorithm>
#include <atomic>
#include <cmath>
#include <limits>
#include <vector>
#include <arm_neon.h>

namespace llm_engine {
namespace arm_neon {

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
