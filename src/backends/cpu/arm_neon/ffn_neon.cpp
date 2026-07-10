#include "backends/cpu/arm_neon/neon_ops.h"
#include "backends/cpu/arm_neon/kernel_common.h"
#include "llm_engine/memory/workspace.h"
#include "llm_engine/tensor.h"
#include <algorithm>
#include <iostream>

namespace llm_engine {
namespace arm_neon {

static size_t get_matmul_pack_bytes(int M, int N, int K) {
    int mp = (M + MR - 1) / MR;
    int np = (N + NR - 1) / NR;
    return (size_t)(mp * MR * K + np * NR * K) * sizeof(float);
}

Status ffn_neon(
    const Tensor& hidden_states,
    Tensor& ffn_output,
    const Tensor& w_gate,
    const Tensor& w_up,
    const Tensor& w_down,
    const Tensor& w_gate_pack,
    const Tensor& w_up_pack,
    const Tensor& w_down_pack,
    const FFNConfig& config,
    Workspace& workspace
) {
    int num_tokens = hidden_states.shape[0];
    int hidden_dim = config.hidden_dim;
    int intermediate_size = config.intermediate_size;

    size_t gate_up_bytes = 2 * (size_t)num_tokens * intermediate_size * sizeof(float);
    size_t proj_pack_bytes = get_matmul_pack_bytes(num_tokens, intermediate_size, hidden_dim);
    size_t down_pack_bytes = get_matmul_pack_bytes(num_tokens, hidden_dim, intermediate_size);
    size_t max_pack_bytes = std::max(proj_pack_bytes, down_pack_bytes);
    size_t required_bytes = gate_up_bytes + max_pack_bytes;

    if (workspace.size() < required_bytes) {
        std::cerr << "[ERROR] FFN Workspace Out of Memory! Required: "
                  << required_bytes << " bytes, but got: " << workspace.size() << " bytes." << std::endl;
        return Status::INVALID_ARGUMENT;
    }

    char* ws_base = static_cast<char*>(workspace.data());

    ws_base = align_ptr(ws_base);
    float* gate_ptr = reinterpret_cast<float*>(ws_base);
    ws_base += num_tokens * intermediate_size * sizeof(float);

    ws_base = align_ptr(ws_base);
    float* up_ptr = reinterpret_cast<float*>(ws_base);
    ws_base += num_tokens * intermediate_size * sizeof(float);

    ws_base = align_ptr(ws_base);
    float* matmul_ws_ptr = reinterpret_cast<float*>(ws_base);

    Tensor Gate({num_tokens, intermediate_size}, gate_ptr);
    Tensor Up({num_tokens, intermediate_size}, up_ptr);

    Status status = Status::SUCCESS;

    // Decode 快速路径：num_tokens == 1 且三个 packed 权重都齐全时，
    // 走 fused gate+up+SwiGLU + 串行 down。
    // - fused kernel 在一次 panel 扫描里同时累加 gate/up 并写出 silu(gate)*up，
    //   省掉一对完整 intermediate 中间向量的写回 + 读回。
    // - down 的 N=hidden_dim=896，并行收益小，保持串行 packed linear。
    if (num_tokens == 1 &&
        w_gate_pack.data &&
        w_up_pack.data &&
        w_down_pack.data) {

        status = fused_gate_up_swiglu_prepacked_parallel_neon(
            hidden_states.ptr<float>(),
            w_gate_pack.ptr<float>(),
            w_up_pack.ptr<float>(),
            gate_ptr,
            hidden_dim,
            intermediate_size
        );
        if (status != Status::SUCCESS) return status;

        status = linear_decode_prepacked_neon(
            gate_ptr,
            w_down_pack.ptr<float>(),
            ffn_output.ptr<float>(),
            intermediate_size,
            hidden_dim,
            nullptr
        );
        return status;
    }

    // batch / prefill 路径：保持原 matmul + swiglu 实现。
    status = matmul_neon(hidden_states, w_gate, Gate, matmul_ws_ptr, false, nullptr);
    if (status != Status::SUCCESS) return status;

    status = matmul_neon(hidden_states, w_up, Up, matmul_ws_ptr, false, nullptr);
    if (status != Status::SUCCESS) return status;

    int total_elements = num_tokens * intermediate_size;
    swiglu_neon(gate_ptr, up_ptr, gate_ptr, total_elements);

    status = matmul_neon(Gate, w_down, ffn_output, matmul_ws_ptr, false, nullptr);
    return status;
}

Status ffn_neon(
    const Tensor& hidden_states,
    Tensor& ffn_output,
    const Tensor& w_gate,
    const Tensor& w_up,
    const Tensor& w_down,
    const FFNConfig& config,
    Workspace& workspace
) {
    Tensor empty;
    return ffn_neon(
        hidden_states, ffn_output,
        w_gate, w_up, w_down,
        empty, empty, empty,
        config, workspace
    );
}

Status ffn_f16_gptq_neon(
    const Tensor& hidden_states,
    Tensor& ffn_output,
    const GPTQInt8Weight& gate_proj,
    const GPTQInt8Weight& up_proj,
    const GPTQInt8Weight& down_proj,
    const FFNConfig& config,
    Workspace& workspace
) {
    if (hidden_states.dtype != DataType::FP16 || ffn_output.dtype != DataType::FP16) {
        return Status::INVALID_ARGUMENT;
    }

    size_t gate_bytes = align_size((size_t)config.intermediate_size * sizeof(fp16_t));
    size_t required = gate_bytes;
    if (workspace.size() < required) {
        return Status::OUT_OF_MEMORY;
    }

    char* base = static_cast<char*>(workspace.data());
    base = align_ptr(base);
    fp16_t* gate = reinterpret_cast<fp16_t*>(base);

    Status status = fused_gate_up_swiglu_gptq_int8_decode_neon(
        hidden_states.ptr<fp16_t>(), gate_proj, up_proj, gate, nullptr, 0);
    if (status != Status::SUCCESS) return status;

    return linear_gptq_int8_decode_neon(
        gate, down_proj, ffn_output.ptr<fp16_t>(), nullptr, nullptr, 0);
}

Status ffn_f16_gptq_batch_neon(
    const Tensor& hidden_states,
    Tensor& ffn_output,
    const GPTQInt8Weight& gate_proj,
    const GPTQInt8Weight& up_proj,
    const GPTQInt8Weight& down_proj,
    const FFNConfig& config,
    Workspace& workspace
) {
    if (hidden_states.dtype != DataType::FP16 || ffn_output.dtype != DataType::FP16) {
        return Status::INVALID_ARGUMENT;
    }
    if (hidden_states.shape.size() < 2 || ffn_output.shape.size() < 2) {
        return Status::INVALID_ARGUMENT;
    }

    const int rows = hidden_states.shape[0];
    const int hidden_dim = hidden_states.shape[1];
    if (rows <= 0 ||
        hidden_dim != config.hidden_dim ||
        ffn_output.shape[0] != rows ||
        ffn_output.shape[1] != config.hidden_dim) {
        return Status::INVALID_ARGUMENT;
    }

    if (rows == 1) {
        return ffn_f16_gptq_neon(
            hidden_states, ffn_output,
            gate_proj, up_proj, down_proj,
            config, workspace);
    }

    size_t gate_bytes = align_size((size_t)rows * config.intermediate_size * sizeof(fp16_t));
    size_t up_bytes = align_size((size_t)rows * config.intermediate_size * sizeof(fp16_t));
    if (workspace.size() < gate_bytes + up_bytes) {
        return Status::OUT_OF_MEMORY;
    }

    char* base = static_cast<char*>(workspace.data());
    base = align_ptr(base);
    fp16_t* gate = reinterpret_cast<fp16_t*>(base);
    base += gate_bytes;
    base = align_ptr(base);
    fp16_t* up = reinterpret_cast<fp16_t*>(base);

    Status status = linear_gptq_int8_batch_neon(
        hidden_states.ptr<fp16_t>(),
        rows,
        gate_proj,
        gate,
        nullptr,
        nullptr,
        0);
    if (status != Status::SUCCESS) return status;

    status = linear_gptq_int8_batch_neon(
        hidden_states.ptr<fp16_t>(), rows, up_proj, up, nullptr, nullptr, 0);
    if (status != Status::SUCCESS) return status;
    swiglu_f16_batch_neon(gate, up, rows, config.intermediate_size);

    return linear_gptq_int8_batch_neon(
        gate,
        rows,
        down_proj,
        ffn_output.ptr<fp16_t>(),
        nullptr,
        nullptr,
        0);
}

} // namespace arm_neon
} // namespace llm_engine
