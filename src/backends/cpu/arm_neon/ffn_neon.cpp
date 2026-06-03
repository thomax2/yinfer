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
    if (num_tokens == 1 && w_gate_pack.data && w_up_pack.data) {
        status = linear_decode_prepacked_parallel_neon(
            hidden_states.ptr<float>(), w_gate_pack.ptr<float>(), gate_ptr,
            hidden_dim, intermediate_size, nullptr
        );
        if (status != Status::SUCCESS) return status;

        status = linear_decode_prepacked_parallel_neon(
            hidden_states.ptr<float>(), w_up_pack.ptr<float>(), up_ptr,
            hidden_dim, intermediate_size, nullptr
        );
        if (status != Status::SUCCESS) return status;
    } else {
        status = matmul_neon(hidden_states, w_gate, Gate, matmul_ws_ptr, false, nullptr);
        if (status != Status::SUCCESS) return status;

        status = matmul_neon(hidden_states, w_up, Up, matmul_ws_ptr, false, nullptr);
        if (status != Status::SUCCESS) return status;
    }

    int total_elements = num_tokens * intermediate_size;
    swiglu_neon(gate_ptr, up_ptr, gate_ptr, total_elements);

    if (num_tokens == 1 && w_down_pack.data) {
        status = linear_decode_prepacked_parallel_neon(
            gate_ptr, w_down_pack.ptr<float>(), ffn_output.ptr<float>(),
            intermediate_size, hidden_dim, nullptr
        );
    } else {
        status = matmul_neon(Gate, w_down, ffn_output, matmul_ws_ptr, false, nullptr);
    }

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

} // namespace arm_neon
} // namespace llm_engine
