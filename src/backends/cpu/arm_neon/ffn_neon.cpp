#include "src/backends/cpu/arm_neon/neon_ops.h"
#include "src/backends/cpu/arm_neon/kernel_common.h" // 引入 MR, NR 用于精确计算 pack 内存
#include "llm_engine/memory/workspace.h"
#include "llm_engine/tensor.h"
#include <algorithm>
#include <iostream>

namespace llm_engine {
namespace arm_neon {

// 辅助函数：精准计算底层 matmul_neon 所需的 pack 缓存大小 (单位: 字节)
static size_t get_matmul_pack_bytes(int M, int N, int K) {
    int mp = (M + MR - 1) / MR;
    int np = (N + NR - 1) / NR;
    // Pack A: mp * MR * K, Pack B: np * NR * K
    return (size_t)(mp * MR * K + np * NR * K) * sizeof(float);
}

// FFN(x) = (SiLU(x*Wgate)*(x*Wup))*Wdown
Status ffn_neon(
    const Tensor& hidden_states, // 输入 [num_tokens, hidden_dim]
    Tensor& ffn_output,          // 输出 [num_tokens, hidden_dim]
    const Tensor& w_gate,        // 权重 [hidden_dim, intermediate_size]
    const Tensor& w_up,          // 权重 [hidden_dim, intermediate_size]
    const Tensor& w_down,        // 权重 [intermediate_size, hidden_dim]
    const FFNConfig& config,
    Workspace& workspace
) {
    // 💡 修复坑2：动态获取 Token 数量，完美兼容 Prefill 和 Decode
    int num_tokens = hidden_states.shape[0]; 
    int hidden_dim = config.hidden_dim;
    int intermediate_size = config.intermediate_size;

    // ==========================================
    // 0. Workspace 内存需求严格校验 (💡 修复坑1)
    // ==========================================
    // Gate 和 Up 张量所需的大小
    size_t gate_up_bytes = 2 * (size_t)num_tokens * intermediate_size * sizeof(float);

    // 计算这三次 Matmul 中最大的 Pack 缓存需求
    // 1. gate / up: [num_tokens, hidden_dim] * [hidden_dim, intermediate_size]
    size_t proj_pack_bytes = get_matmul_pack_bytes(num_tokens, intermediate_size, hidden_dim);
    // 2. down: [num_tokens, intermediate_size] * [intermediate_size, hidden_dim]
    size_t down_pack_bytes = get_matmul_pack_bytes(num_tokens, hidden_dim, intermediate_size);
    
    size_t max_pack_bytes = std::max(proj_pack_bytes, down_pack_bytes);
    size_t required_bytes = gate_up_bytes + max_pack_bytes;

    if (workspace.size() < required_bytes) {
        std::cerr << "[ERROR] FFN Workspace Out of Memory! Required: " 
                  << required_bytes << " bytes, but got: " << workspace.size() << " bytes." << std::endl;
        return Status::INVALID_ARGUMENT; // 或者你可以定义一个 Status::OUT_OF_MEMORY
    }

    // ==========================================
    // 1. 安全且 16-byte 对齐的内存切分
    // ==========================================
    char* ws_base = static_cast<char*>(workspace.data()); 
    
    // 1. 分配 Gate (起始地址默认对齐)
    ws_base = align_ptr(ws_base);
    float* gate_ptr = reinterpret_cast<float*>(ws_base);                
    ws_base += num_tokens * intermediate_size * sizeof(float);
    
    // 2. 分配 Up (切分偏移后，可能不对齐了，强制对齐！)
    ws_base = align_ptr(ws_base);
    float* up_ptr   = reinterpret_cast<float*>(ws_base);                
    ws_base += num_tokens * intermediate_size * sizeof(float);
    
    // 3. 分配 matmul_ws_ptr (强制对齐！)
    ws_base = align_ptr(ws_base);
    float* matmul_ws_ptr = reinterpret_cast<float*>(ws_base);


    // 动态 2D 张量视图
    Tensor Gate({num_tokens, intermediate_size}, gate_ptr);
    Tensor Up({num_tokens, intermediate_size}, up_ptr);

    // ==========================================
    // 2. 向下投影与门控投影
    // ==========================================
    Status status = matmul_neon(hidden_states, w_gate, Gate, matmul_ws_ptr);
    if (status != Status::SUCCESS) return status;

    status = matmul_neon(hidden_states, w_up, Up, matmul_ws_ptr);
    if (status != Status::SUCCESS) return status;

    // ==========================================
    // 3. SwiGLU 激活 (支持多 Token 批量激活)
    // ==========================================
    // 总元素个数 = num_tokens * intermediate_size
    int total_elements = num_tokens * intermediate_size;
    swiglu_neon(gate_ptr, up_ptr, gate_ptr, total_elements);        // SiLU(x*Wgate)*(x*Wup)

    // ==========================================
    // 4. 向下投影
    // ==========================================
    status = matmul_neon(Gate, w_down, ffn_output, matmul_ws_ptr);

    return status;
}

} // namespace arm_neon
} // namespace llm_engine