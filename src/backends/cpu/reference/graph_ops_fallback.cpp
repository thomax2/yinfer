#include "backends/cpu/arm_neon/neon_ops.h"
#include "backends/cpu/reference/math_ref.h"

#include <cmath>
#include <cstring>
#include <limits>

namespace llm_engine::arm_neon {

Status matmul_neon(
    const Tensor& a, const Tensor& b, Tensor& c,
    float*, bool trans_b, const float* bias
) {
    if (!trans_b && !bias) return reference::matmul_ref(a, b, c);
    if (a.dtype != DataType::FP32 || b.dtype != DataType::FP32 ||
        c.dtype != DataType::FP32 || a.shape.size() != 2 ||
        b.shape.size() != 2 || c.shape.size() != 2) {
        return Status::INVALID_ARGUMENT;
    }
    const int m = a.shape[0];
    const int k = a.shape[1];
    const int n = trans_b ? b.shape[0] : b.shape[1];
    if ((trans_b ? b.shape[1] : b.shape[0]) != k ||
        c.shape[0] != m || c.shape[1] != n) {
        return Status::SHAPE_MISMATCH;
    }
    const float* ap = a.ptr<float>();
    const float* bp = b.ptr<float>();
    float* cp = c.ptr<float>();
    for (int row = 0; row < m; ++row) {
        for (int col = 0; col < n; ++col) {
            float sum = bias ? bias[col] : 0.0f;
            for (int inner = 0; inner < k; ++inner) {
                const float bv = trans_b ? bp[col * k + inner] : bp[inner * n + col];
                sum += ap[row * k + inner] * bv;
            }
            cp[row * n + col] = sum;
        }
    }
    return Status::SUCCESS;
}

Status matmul_f16_neon(
    const Tensor&, const Tensor&, Tensor&, fp16_t*, bool, const fp16_t*
) {
    return Status::INVALID_ARGUMENT;
}

Status gemv_neon_transposed(
    const Tensor& a, const Tensor& b_t, Tensor& c, const float* bias
) {
    if (a.dtype != DataType::FP32 || b_t.dtype != DataType::FP32 ||
        c.dtype != DataType::FP32 || a.shape.size() != 2 ||
        b_t.shape.size() != 2 || a.shape[0] != 1 ||
        b_t.shape[1] != a.shape[1] || c.size() != static_cast<size_t>(b_t.shape[0])) {
        return Status::INVALID_ARGUMENT;
    }
    const int k = a.shape[1];
    const int n = b_t.shape[0];
    const float* ap = a.ptr<float>();
    const float* bp = b_t.ptr<float>();
    float* cp = c.ptr<float>();
    for (int col = 0; col < n; ++col) {
        float sum = bias ? bias[col] : 0.0f;
        for (int inner = 0; inner < k; ++inner) sum += ap[inner] * bp[col * k + inner];
        cp[col] = sum;
    }
    return Status::SUCCESS;
}

void add_neon(const Tensor& a, const Tensor& b, Tensor& c) {
    (void)reference::add_ref(a, b, c);
}

void add_f16_neon(const Tensor&, const Tensor&, Tensor&) {}

void rmsnorm_neon(const float* x, const float* weight, float* y, int n, float eps) {
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) sum += x[i] * x[i];
    const float scale = 1.0f / std::sqrt(sum / n + eps);
    for (int i = 0; i < n; ++i) y[i] = x[i] * scale * weight[i];
}

void rmsnorm_f16_neon(const fp16_t*, const fp16_t*, fp16_t*, int, float) {}

Status rmsnorm_f16_batch_neon(
    const fp16_t*, const fp16_t*, fp16_t*, int, int, float
) {
    return Status::INVALID_ARGUMENT;
}

// Mixed Selective Batch 在 x86 上只需要能够完整编译、链接；数值执行仍明确返回
// INVALID_ARGUMENT，避免把 uint16_t 存储类型误当成真正的 ARM FP16 运算。
Status linear_gptq_int8_decode_neon(
    const fp16_t*, const GPTQInt8Weight&, fp16_t*, const fp16_t*, void*, size_t
) {
    return Status::INVALID_ARGUMENT;
}

Status linear_gptq_int8_batch_neon(
    const fp16_t*, int, const GPTQInt8Weight&, fp16_t*, const fp16_t*, void*, size_t
) {
    return Status::INVALID_ARGUMENT;
}

Status pack_gptq_batch_a_f16_neon(
    const fp16_t*, int, int, fp16_t*
) {
    return Status::INVALID_ARGUMENT;
}

Status linear_gptq_int8_batch_packed_a_neon(
    const fp16_t*, int, const GPTQInt8Weight&, fp16_t*, const fp16_t*, void*, size_t
) {
    return Status::INVALID_ARGUMENT;
}

size_t linear_gptq_int8_batch_argmax_workspace_bytes(
    int, const GPTQInt8Weight&
) {
    return 0;
}

Status linear_gptq_int8_decode_argmax_batch_neon(
    const fp16_t*, int, const GPTQInt8Weight&, ArgmaxResult*, void*, size_t
) {
    return Status::INVALID_ARGUMENT;
}

GPTQBatchKernelStats snapshot_gptq_batch_kernel_stats() {
    return GPTQBatchKernelStats{};
}

GPTQBatchKernelStats diff_gptq_batch_kernel_stats(
    const GPTQBatchKernelStats&, const GPTQBatchKernelStats&
) {
    return GPTQBatchKernelStats{};
}

Status softmax_f16_inplace_neon(fp16_t*, int, int) {
    return Status::INVALID_ARGUMENT;
}

void attention_decode_score_f16_neon_public(
    const fp16_t*, const fp16_t*, fp16_t*, int, int, int, float
) {}

void attention_decode_value_f16_neon_public(
    const fp16_t*, const fp16_t*, fp16_t*, int, int, int
) {}

Status attention_decode_score_paged_f16_neon_public(
    const fp16_t*, fp16_t*, int, int, int, float,
    const fp16_t*, const int*, int, int, int, int, int, int, int
) {
    return Status::INVALID_ARGUMENT;
}

Status attention_decode_value_paged_f16_neon_public(
    const fp16_t*, fp16_t*, int, int, int,
    const fp16_t*, const int*, int, int, int, int, int, int, int
) {
    return Status::INVALID_ARGUMENT;
}

Status attention_prefill_paged_f16_neon_public(
    const fp16_t*, fp16_t*, int, int, int, int, int, float,
    const fp16_t*, const fp16_t*, const int*, int, int, int, int,
    int, int, int, fp16_t*, size_t, fp16_t*, size_t
) {
    return Status::INVALID_ARGUMENT;
}

Status attention_prefill_paged_online_f16_neon_public(
    const fp16_t*, fp16_t*, int, int, int, int, int, float,
    const fp16_t*, const fp16_t*, const int*, int, int, int, int,
    int, int, int, fp16_t*, size_t, float*, size_t
) {
    return Status::INVALID_ARGUMENT;
}

void add_f16_batch_neon(
    const fp16_t* a, const fp16_t* b, fp16_t* out, size_t elements
) {
    // On non-ARM hosts fp16_t is only a storage type. Preserve deterministic
    // behavior for alias/copy tests; numeric FP16 kernels remain unsupported.
    if (!out || !a) return;
    if (!b) {
        std::memcpy(out, a, elements * sizeof(fp16_t));
    }
}

ArgmaxResult linear_gptq_int8_decode_argmax_neon(
    const fp16_t*, const GPTQInt8Weight&, void*, size_t
) {
    return {-1, -std::numeric_limits<float>::infinity()};
}

void rope_neon(float* x, const float* cosine, const float* sine, int n) {
    const int half = n / 2;
    for (int i = 0; i < half; ++i) {
        const float first = x[i];
        const float second = x[i + half];
        x[i] = first * cosine[i] - second * sine[i];
        x[i + half] = second * cosine[i] + first * sine[i];
    }
}

void rope_f16_neon(fp16_t*, const fp16_t*, const fp16_t*, int) {}

Status rope_qk_f16_batch_neon(
    fp16_t*, fp16_t*, const int*, int, int, int, int,
    const fp16_t*, const fp16_t*
) {
    return Status::INVALID_ARGUMENT;
}

void swiglu_neon(const float* gate, const float* up, float* y, int n) {
    for (int i = 0; i < n; ++i) {
        y[i] = gate[i] / (1.0f + std::exp(-gate[i])) * up[i];
    }
}

void swiglu_f16_neon(const fp16_t*, const fp16_t*, fp16_t*, int) {}

void swiglu_f16_batch_neon(fp16_t*, const fp16_t*, int, int) {}

Status attention_f16_gptq_neon(
    const Tensor&, Tensor&,
    const GPTQInt8Weight&, const GPTQInt8Weight&,
    const GPTQInt8Weight&, const GPTQInt8Weight&,
    const fp16_t*, const fp16_t*, const fp16_t*,
    const fp16_t*, const fp16_t*, KVCache&, int, int,
    const AttentionConfig&, Workspace&
) {
    return Status::INVALID_ARGUMENT;
}

Status attention_f16_gptq_prefill_neon(
    const Tensor&, Tensor&,
    const GPTQInt8Weight&, const GPTQInt8Weight&,
    const GPTQInt8Weight&, const GPTQInt8Weight&,
    const fp16_t*, const fp16_t*, const fp16_t*,
    const fp16_t*, const fp16_t*, KVCache&, int, int,
    const AttentionConfig&, Workspace&
) {
    return Status::INVALID_ARGUMENT;
}

Status ffn_f16_gptq_neon(
    const Tensor&, Tensor&,
    const GPTQInt8Weight&, const GPTQInt8Weight&, const GPTQInt8Weight&,
    const FFNConfig&, Workspace&
) {
    return Status::INVALID_ARGUMENT;
}

Status ffn_f16_gptq_batch_neon(
    const Tensor&, Tensor&,
    const GPTQInt8Weight&, const GPTQInt8Weight&, const GPTQInt8Weight&,
    const FFNConfig&, Workspace&
) {
    return Status::INVALID_ARGUMENT;
}

Status qwen_block_neon(
    Tensor&, const Tensor&,
    const Tensor&, const Tensor&, const Tensor&, const Tensor&,
    const Tensor&, const Tensor&, const Tensor&, const Tensor&,
    const float*, const float*, const float*, const float*, const float*,
    const Tensor&, const Tensor&, const Tensor&, const Tensor&,
    const Tensor&, const Tensor&, const Tensor&,
    KVCache&, int, int, const AttentionConfig&, const FFNConfig&, float, Workspace&
) {
    return Status::INVALID_ARGUMENT;
}

Status qwen_block_neon(
    Tensor&, const Tensor&,
    const Tensor&, const Tensor&, const Tensor&, const Tensor&,
    const float*, const float*, const float*, const float*, const float*,
    const Tensor&, const Tensor&, const Tensor&, const Tensor&,
    KVCache&, int, int, const AttentionConfig&, const FFNConfig&, float, Workspace&
) {
    return Status::INVALID_ARGUMENT;
}

Status qwen_block_f16_gptq_neon(
    Tensor&, const Tensor&,
    const GPTQInt8Weight&, const GPTQInt8Weight&, const GPTQInt8Weight&, const GPTQInt8Weight&,
    const fp16_t*, const fp16_t*, const fp16_t*, const fp16_t*, const fp16_t*,
    const Tensor&,
    const GPTQInt8Weight&, const GPTQInt8Weight&, const GPTQInt8Weight&,
    KVCache&, int, int, const AttentionConfig&, const FFNConfig&, float, Workspace&
) {
    return Status::INVALID_ARGUMENT;
}

} // namespace llm_engine::arm_neon
