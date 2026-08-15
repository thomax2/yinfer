#pragma once

#include <cstddef>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "llm_engine/tensor.h"
#include "llm_engine/status.h"
#include "llm_engine/memory/kv_cache.h"
#include "backends/cpu/arm_neon/neon_ops.h"

namespace llm_engine {

struct InplaceAliasCandidate {
    size_t output_index;
    size_t input_index;
};

struct QwenBlockWeights {
    Tensor norm1_w;
    Tensor norm2_w;

    Tensor b_q;
    Tensor b_k;
    Tensor b_v;

    arm_neon::GPTQInt8Weight q_proj;
    arm_neon::GPTQInt8Weight k_proj;
    arm_neon::GPTQInt8Weight v_proj;
    arm_neon::GPTQInt8Weight o_proj;

    arm_neon::GPTQInt8Weight gate_proj;
    arm_neon::GPTQInt8Weight up_proj;
    arm_neon::GPTQInt8Weight down_proj;
};

class GraphNode{
public:
    std::vector<Tensor*> inputs;
    std::vector<Tensor*> outputs;

    virtual Status forward() = 0;
    virtual ~GraphNode() = default;
    virtual bool is_temporary() const { return false; }
    virtual std::vector<InplaceAliasCandidate> inplace_alias_candidates() const {
        return {};
    }
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
    RoPENode(Tensor* X, Tensor* Cos, Tensor* Sin, Tensor* Y);
    Status forward() override;
    std::vector<InplaceAliasCandidate> inplace_alias_candidates() const override;
};

class CopyBackNode : public GraphNode {
public:
    CopyBackNode(Tensor* source, Tensor* target);
    Status forward() override;
    bool is_temporary() const override { return true; }

private:
    Tensor* target_;
};

class QwenBlockNode : public GraphNode {
public:
    arm_neon::AttentionConfig attn_config;
    arm_neon::FFNConfig ffn_config;
    QwenBlockWeights* weights = nullptr;
    float rms_norm_eps;
    int layer_id;
    int* current_pos_ptr; // 💡 使用指针，方便外部的推理大循环统一更新位置
    KVCache* kv_cache;
    void* external_workspace = nullptr;
    size_t external_workspace_bytes = 0;

    // 构造函数：接收所有的参数和张量
    QwenBlockNode(
        Tensor* hidden_states, Tensor* output_hidden_states,
        Tensor* norm1_weight,
        Tensor* b_q, Tensor* b_k, Tensor* b_v,
        Tensor* cos, Tensor* sin,
        Tensor* norm2_weight,
        QwenBlockWeights* block_weights,
        KVCache* cache,
        int l_id,
        int* pos_ptr,
        arm_neon::AttentionConfig a_conf,
        arm_neon::FFNConfig f_conf,
        float eps
    );

    QwenBlockNode(
        Tensor* hidden_states, Tensor* output_hidden_states,
        Tensor* norm1_weight,
        Tensor* w_q, Tensor* w_k, Tensor* w_v, Tensor* w_o,
        Tensor* w_q_pack, Tensor* w_k_pack, Tensor* w_v_pack, Tensor* w_o_pack,
        Tensor* b_q, Tensor* b_k, Tensor* b_v,
        Tensor* cos, Tensor* sin,
        Tensor* norm2_weight,
        Tensor* w_gate, Tensor* w_up, Tensor* w_down,
        Tensor* w_gate_pack, Tensor* w_up_pack, Tensor* w_down_pack,
        KVCache* cache,
        int l_id,
        int* pos_ptr,
        arm_neon::AttentionConfig a_conf,
        arm_neon::FFNConfig f_conf,
        float eps
    );

    QwenBlockNode(
        Tensor* hidden_states, Tensor* output_hidden_states,
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
    void set_external_workspace(void* workspace, size_t bytes);
    std::vector<InplaceAliasCandidate> inplace_alias_candidates() const override;
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
        DataType dtype = DataType::FP16
    );

    Tensor* create_tensor_from_ptr(
        const std::vector<int>& shape,
        void* data,
        DataType dtype = DataType::FP16
    );

    // 返回句柄当前对应的最新逻辑值。调用方可以继续持有最初的 Tensor*；
    // Graph builder 会在添加后续节点时自动解析到 mutation 后的版本。
    Tensor* latest_value(Tensor* tensor) const;

    // 自定义 mutation 节点的扩展入口。先创建独立输出版本，再用 add_node
    // 添加 inputs/outputs 已分离的节点。
    Tensor* create_mutation_version(Tensor* tensor);
    GraphNode* add_node(std::unique_ptr<GraphNode> node);

    MatmulNode*     add_matmul(Tensor* A, Tensor* B, Tensor* C);
    GemvNode*       add_gemv(Tensor* A, Tensor* B_T, Tensor* C);
    AddNode*        add_add(Tensor* A, Tensor* B, Tensor* C);

    RMSNormNode*    add_rmsnorm(Tensor* X, Tensor* Weight, Tensor* Y, float eps);
    SwiGLUNode*     add_swiglu(Tensor* Gate, Tensor* Up, Tensor* Y);
    RoPENode*       add_rope(Tensor* X, Tensor* Cos, Tensor* Sin);
    
    QwenBlockNode* add_qwen_block(
        Tensor* hidden_states, Tensor* norm1_weight,
        Tensor* b_q, Tensor* b_k, Tensor* b_v,
        Tensor* cos, Tensor* sin,
        Tensor* norm2_weight,
        QwenBlockWeights* weights,
        KVCache* cache, int l_id, int* pos_ptr,
        arm_neon::AttentionConfig a_conf, arm_neon::FFNConfig f_conf, float eps
    );

    QwenBlockNode* add_qwen_block(
        Tensor* hidden_states, Tensor* norm1_weight,
        Tensor* w_q, Tensor* w_k, Tensor* w_v, Tensor* w_o,
        Tensor* w_q_pack, Tensor* w_k_pack, Tensor* w_v_pack, Tensor* w_o_pack,
        Tensor* b_q, Tensor* b_k, Tensor* b_v,
        Tensor* cos, Tensor* sin,
        Tensor* norm2_weight,
        Tensor* w_gate, Tensor* w_up, Tensor* w_down,
        Tensor* w_gate_pack, Tensor* w_up_pack, Tensor* w_down_pack,
        KVCache* cache, int l_id, int* pos_ptr,
        arm_neon::AttentionConfig a_conf, arm_neon::FFNConfig f_conf, float eps
    );

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

private:
    friend class GraphCompiler;

    Tensor* mutation_root(Tensor* tensor) const;
    Tensor* prepare_output(Tensor* requested, const std::vector<Tensor*>& inputs);
    void register_tensor(Tensor* tensor);
    void register_node_outputs(GraphNode* node);
    void materialize_mutation_fixups();
    bool is_mutation_version(Tensor* tensor) const;
    void ensure_building() const;

    std::unordered_map<Tensor*, Tensor*> mutation_root_;
    std::unordered_map<Tensor*, Tensor*> latest_value_;
    std::unordered_set<Tensor*> produced_values_;
    std::unordered_set<Tensor*> mutation_versions_;
    std::vector<Tensor*> mutation_roots_in_order_;
    bool mutation_fixups_materialized_ = false;
};

}
