#pragma once

#include <memory>
#include <vector>

#include "llm_engine/tensor.h"
#include "llm_engine/status.h"
#include "llm_engine/memory/kv_cache.h"
#include "backends/cpu/arm_neon/neon_ops.h"

namespace llm_engine {

class GraphNode{
public:
    std::vector<Tensor*> inputs;
    std::vector<Tensor*> outputs;

    virtual Status forward() = 0;
    virtual ~GraphNode() = default;
    virtual bool is_temporary() const { return false; }
};


class MatmulNode : public GraphNode {
public:
    MatmulNode(Tensor* A, Tensor* B, Tensor* C);

    Status forward() override;  // 重写基类的纯虚函数，在cpp中完成具体的实现。
};

class GemvNode : public GraphNode {
public:
    GemvNode(Tensor* A, Tensor* B_T, Tensor* C);
    Status forward() override;
};

class AddNode : public GraphNode {
public:
    AddNode(Tensor* A, Tensor* B, Tensor* C);
    
    Status forward() override;
};

class RMSNormNode : public GraphNode {
public:
    float eps;
    RMSNormNode(Tensor* X, Tensor* Weight, Tensor* Y, float eps);
    Status forward() override;
};

class SwiGLUNode : public GraphNode {
public:
    SwiGLUNode(Tensor* Gate, Tensor* Up, Tensor* Y);
    Status forward() override;
};

class RoPENode : public GraphNode {
public:
    // 注意：你写的 RoPE 是 in-place 原地修改的，所以输入输出都是 X
    RoPENode(Tensor* X, Tensor* Cos, Tensor* Sin);
    Status forward() override;
};

class QwenBlockNode : public GraphNode {
public:
    arm_neon::AttentionConfig attn_config;
    arm_neon::FFNConfig ffn_config;
    float rms_norm_eps;
    int layer_id;
    int* current_pos_ptr; // 💡 使用指针，方便外部的推理大循环统一更新位置
    KVCache* kv_cache;

    // 构造函数：接收所有的参数和张量
    QwenBlockNode(
        Tensor* hidden_states,
        Tensor* norm1_weight,
        Tensor* w_q, Tensor* w_k, Tensor* w_v, Tensor* w_o,
        Tensor* b_q, Tensor* b_k, Tensor* b_v,
        Tensor* cos, Tensor* sin,
        Tensor* norm2_weight,
        Tensor* w_gate, Tensor* w_up, Tensor* w_down,
        KVCache* cache, 
        int l_id, 
        int* pos_ptr, 
        arm_neon::AttentionConfig a_conf,
        arm_neon::FFNConfig f_conf,
        float eps
    );

    Status forward() override;
};

class ComputationGraph {
public:
    std::vector<std::unique_ptr<GraphNode>> nodes;
    std::vector<std::unique_ptr<Tensor>> tensors;

    size_t arena_size = 0;
    void* arena_buffer = nullptr;

    ComputationGraph() = default;
    ~ComputationGraph() {
        // 如果建图时申请了 Arena 大内存，在这里将其归还给内存池
        if (arena_buffer != nullptr) {
            g_memory_pool->free_block(arena_buffer);
            arena_buffer = nullptr;
            arena_size = 0;
        }
    }

    // 禁用拷贝构造和拷贝赋值（C++ 资源管理最佳实践）
    ComputationGraph(const ComputationGraph&) = delete;
    ComputationGraph& operator=(const ComputationGraph&) = delete;

public:
    
    Tensor* create_tensor(
        const std::vector<int>& shape,
        DataType dtype = DataType::FP32
    );

    Tensor* create_tensor_from_ptr(
        const std::vector<int>& shape,
        void* data,
        DataType dtype = DataType::FP32
    );

    MatmulNode*     add_matmul(Tensor* A, Tensor* B, Tensor* C);
    GemvNode*       add_gemv(Tensor* A, Tensor* B_T, Tensor* C);
    AddNode*        add_add(Tensor* A, Tensor* B, Tensor* C);

    RMSNormNode*    add_rmsnorm(Tensor* X, Tensor* Weight, Tensor* Y, float eps);
    SwiGLUNode*     add_swiglu(Tensor* Gate, Tensor* Up, Tensor* Y);
    RoPENode*       add_rope(Tensor* X, Tensor* Cos, Tensor* Sin);
    
    QwenBlockNode* add_qwen_block(
        Tensor* hidden_states, Tensor* norm1_weight,
        Tensor* w_q, Tensor* w_k, Tensor* w_v, Tensor* w_o,
        Tensor* b_q, Tensor* b_k, Tensor* b_v,
        Tensor* cos, Tensor* sin,
        Tensor* norm2_weight,
        Tensor* w_gate, Tensor* w_up, Tensor* w_down,
        KVCache* cache, int l_id, int* pos_ptr,
        arm_neon::AttentionConfig a_conf, arm_neon::FFNConfig f_conf, float eps
    );
};

}