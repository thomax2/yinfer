#include "llm_engine/graph/graph.h"
#include "backends/cpu/reference/math_ref.h"
#include "backends/cpu/arm_neon/neon_ops.h"
#include "backends/cpu/arm_neon/kernel_common.h"

namespace llm_engine {

using namespace reference;
using namespace arm_neon;
/*                                   定义 node                                          */
// 类名::函数名(参数列表) : 初始化列表
MatmulNode::MatmulNode(Tensor* A, Tensor* B, Tensor* C) {
    inputs = {A, B};
    outputs = {C};
}

Status MatmulNode::forward() {
    Tensor* A = inputs[0];
    Tensor* B = inputs[1];
    Tensor* C = outputs[0];

    int M = A->shape[0];
    int K = A->shape[1];
    int N = B->shape[1];

    // 1. 计算需要的 Workspace 大小
    if (A->dtype == DataType::FP16) {
        int mp = (M + MR_F16 - 1) / MR_F16;
        int np = (N + NR_F16 - 1) / NR_F16;
        size_t ws_size = (size_t)(mp * MR_F16 * K + np * NR_F16 * K) * sizeof(fp16_t);
        fp16_t* workspace = (fp16_t*)g_memory_pool->allocate(ws_size);
        Status status = arm_neon::matmul_f16_neon(*A, *B, *C, workspace, false, nullptr);
        g_memory_pool->free_block(workspace);
        return status;
    }

    int mp = (M + MR - 1) / MR;
    int np = (N + NR - 1) / NR;
    size_t ws_size = (mp * MR * K + np * NR * K) * sizeof(float);

    // 2. 从全局内存池临时申请 Workspace
    float* workspace = (float*)g_memory_pool->allocate(ws_size);

    // 3. 执行 NEON 矩阵乘法
    Status status = arm_neon::matmul_neon(*A, *B, *C, workspace, false, nullptr);

    // 4. 执行完毕，立刻释放临时内存，防止显存/内存泄漏！
    g_memory_pool->free_block(workspace);

    return status;
}

GemvNode::GemvNode(Tensor* A, Tensor* B_T, Tensor* C) {
    inputs = {A, B_T};
    outputs = {C};
}

Status GemvNode::forward() {
    Tensor* A = inputs[0];
    Tensor* B_T = inputs[1];
    Tensor* C = outputs[0];

    // 不需要任何 Workspace 内存分配，直接调用算子，极速执行！
    return arm_neon::gemv_neon_transposed(*A, *B_T, *C, nullptr);
}

GemvNode* ComputationGraph::add_gemv(Tensor* A, Tensor* B_T, Tensor* C) {
    nodes.push_back(std::make_unique<GemvNode>(A, B_T, C));
    return static_cast<GemvNode*>(nodes.back().get());
}

AddNode::AddNode(Tensor* A, Tensor* B, Tensor* C) {
    inputs = {A, B};
    outputs = {C};
}

Status AddNode::forward() {
    // 调用你手写的 NEON add 算子
    if (inputs[0]->dtype == DataType::FP16) {
        add_f16_neon(*inputs[0], *inputs[1], *outputs[0]);
    } else {
        add_neon(*inputs[0], *inputs[1], *outputs[0]);
    }
    return Status::SUCCESS;
}

// --- RMSNorm ---
RMSNormNode::RMSNormNode(Tensor* X, Tensor* Weight, Tensor* Y, float eps) : eps(eps) {
    inputs = {X, Weight};
    outputs = {Y};
}

Status RMSNormNode::forward() {
    int n = inputs[0]->shape.back(); // 获取最后一个维度
    if (inputs[0]->dtype == DataType::FP16) {
        arm_neon::rmsnorm_f16_neon(
            inputs[0]->ptr<fp16_t>(),
            inputs[1]->ptr<fp16_t>(),
            outputs[0]->ptr<fp16_t>(),
            n, eps
        );
    } else {
        arm_neon::rmsnorm_neon(
            inputs[0]->ptr<float>(),
            inputs[1]->ptr<float>(),
            outputs[0]->ptr<float>(),
            n, eps
        );
    }
    return Status::SUCCESS;
}

// --- SwiGLU ---
SwiGLUNode::SwiGLUNode(Tensor* Gate, Tensor* Up, Tensor* Y) {
    inputs = {Gate, Up};
    outputs = {Y};
}

Status SwiGLUNode::forward() {
    int n = inputs[0]->size();
    if (inputs[0]->dtype == DataType::FP16) {
        arm_neon::swiglu_f16_neon(
            inputs[0]->ptr<fp16_t>(),
            inputs[1]->ptr<fp16_t>(),
            outputs[0]->ptr<fp16_t>(),
            n
        );
    } else {
        arm_neon::swiglu_neon(
            inputs[0]->ptr<float>(),
            inputs[1]->ptr<float>(),
            outputs[0]->ptr<float>(),
            n
        );
    }
    return Status::SUCCESS;
}

// --- RoPE ---
RoPENode::RoPENode(Tensor* X, Tensor* Cos, Tensor* Sin) {
    inputs = {X, Cos, Sin};
    outputs = {X}; // In-place 修改
}

Status RoPENode::forward() {
    int n = inputs[0]->shape.back();
    if (inputs[0]->dtype == DataType::FP16) {
        arm_neon::rope_f16_neon(
            inputs[0]->ptr<fp16_t>(),
            inputs[1]->ptr<fp16_t>(),
            inputs[2]->ptr<fp16_t>(),
            n
        );
    } else {
        arm_neon::rope_neon(
            inputs[0]->ptr<float>(),
            inputs[1]->ptr<float>(),
            inputs[2]->ptr<float>(),
            n
        );
    }
    return Status::SUCCESS;
}

QwenBlockNode::QwenBlockNode(
    Tensor* hidden_states, Tensor* norm1_weight,
    Tensor* w_q, Tensor* w_k, Tensor* w_v, Tensor* w_o,
    Tensor* w_q_pack, Tensor* w_k_pack, Tensor* w_v_pack, Tensor* w_o_pack,
    Tensor* b_q, Tensor* b_k, Tensor* b_v,
    Tensor* cos, Tensor* sin,
    Tensor* norm2_weight,
    Tensor* w_gate, Tensor* w_up, Tensor* w_down,
    Tensor* w_gate_pack, Tensor* w_up_pack, Tensor* w_down_pack,
    KVCache* cache, int l_id, int* pos_ptr,
    arm_neon::AttentionConfig a_conf, arm_neon::FFNConfig f_conf, float eps)
    : kv_cache(cache), layer_id(l_id), current_pos_ptr(pos_ptr),
      attn_config(a_conf), ffn_config(f_conf), rms_norm_eps(eps)
{
    inputs.push_back(hidden_states);
    inputs.push_back(norm1_weight);
    inputs.push_back(w_q);
    inputs.push_back(w_k);
    inputs.push_back(w_v);
    inputs.push_back(w_o);
    inputs.push_back(w_q_pack);
    inputs.push_back(w_k_pack);
    inputs.push_back(w_v_pack);
    inputs.push_back(w_o_pack);
    inputs.push_back(b_q);
    inputs.push_back(b_k);
    inputs.push_back(b_v);
    inputs.push_back(cos);
    inputs.push_back(sin);
    inputs.push_back(norm2_weight);
    inputs.push_back(w_gate);
    inputs.push_back(w_up);
    inputs.push_back(w_down);
    inputs.push_back(w_gate_pack);
    inputs.push_back(w_up_pack);
    inputs.push_back(w_down_pack);

    outputs.push_back(hidden_states);
}

QwenBlockNode::QwenBlockNode(
    Tensor* hidden_states, Tensor* norm1_weight,
    Tensor* w_q, Tensor* w_k, Tensor* w_v, Tensor* w_o,
    Tensor* b_q, Tensor* b_k, Tensor* b_v,
    Tensor* cos, Tensor* sin,
    Tensor* norm2_weight,
    Tensor* w_gate, Tensor* w_up, Tensor* w_down,
    KVCache* cache, int l_id, int* pos_ptr,
    arm_neon::AttentionConfig a_conf, arm_neon::FFNConfig f_conf, float eps)
    : kv_cache(cache), layer_id(l_id), current_pos_ptr(pos_ptr),
      attn_config(a_conf), ffn_config(f_conf), rms_norm_eps(eps) 
{
    // 将所有依赖的 Tensor 依次存入 inputs，以便图层可以统一管理它们
    inputs.push_back(hidden_states); // inputs[0]
    inputs.push_back(norm1_weight);  // inputs[1]
    inputs.push_back(w_q);           // inputs[2]
    inputs.push_back(w_k);           // inputs[3]
    inputs.push_back(w_v);           // inputs[4]
    inputs.push_back(w_o);           // inputs[5]
    inputs.push_back(b_q);           // inputs[6]
    inputs.push_back(b_k);           // inputs[7]
    inputs.push_back(b_v);           // inputs[8]
    inputs.push_back(cos);           // inputs[9]
    inputs.push_back(sin);           // inputs[10]
    inputs.push_back(norm2_weight);  // inputs[11]
    inputs.push_back(w_gate);        // inputs[12]
    inputs.push_back(w_up);          // inputs[13]
    inputs.push_back(w_down);        // inputs[14]

    // 因为 QwenBlock 是原地修改 (In-place) 计算，输出同样是 hidden_states
    outputs.push_back(hidden_states);
}

Status QwenBlockNode::forward() {
    if (weights) {
        Tensor* hidden_states = inputs[0];
        Tensor* norm1_weight = inputs[1];
        Tensor* b_q = inputs[2];
        Tensor* b_k = inputs[3];
        Tensor* b_v = inputs[4];
        Tensor* cos = inputs[5];
        Tensor* sin = inputs[6];
        Tensor* norm2_weight = inputs[7];

        int num_tokens = hidden_states->shape[0];
        int hidden_dim = hidden_states->shape[1];
        size_t buffer_bytes = 2 * align_size((size_t)num_tokens * hidden_dim * sizeof(fp16_t));
        size_t pack_bytes = 64ULL * 1024 * 1024;
        size_t total_ws_size = buffer_bytes + pack_bytes + 64;
        bool owns_workspace = false;
        void* raw_ptr = external_workspace;
        size_t workspace_size = external_workspace_bytes;
        if (!raw_ptr || workspace_size < total_ws_size) {
            raw_ptr = g_memory_pool->allocate(total_ws_size);
            workspace_size = total_ws_size;
            owns_workspace = true;
        }
        if (!raw_ptr) return Status::OUT_OF_MEMORY;

        Workspace temp_workspace(raw_ptr, workspace_size);
        Status status = arm_neon::qwen_block_f16_gptq_neon(
            *hidden_states,
            *norm1_weight,
            weights->q_proj, weights->k_proj, weights->v_proj, weights->o_proj,
            b_q->ptr<fp16_t>(), b_k->ptr<fp16_t>(), b_v->ptr<fp16_t>(),
            cos->ptr<fp16_t>(), sin->ptr<fp16_t>(),
            *norm2_weight,
            weights->gate_proj, weights->up_proj, weights->down_proj,
            *kv_cache,
            layer_id,
            *current_pos_ptr,
            attn_config,
            ffn_config,
            rms_norm_eps,
            temp_workspace
        );
        if (owns_workspace) {
            g_memory_pool->free_block(raw_ptr);
        }
        return status;
    }

    // 1. 解包 Inputs (顺序与构造函数中 push_back 的顺序一致)
    Tensor* hidden_states = inputs[0];
    Tensor* norm1_weight  = inputs[1];
    Tensor* w_q = inputs[2]; Tensor* w_k = inputs[3]; Tensor* w_v = inputs[4]; Tensor* w_o = inputs[5];
    Tensor* w_q_pack = nullptr; Tensor* w_k_pack = nullptr; Tensor* w_v_pack = nullptr; Tensor* w_o_pack = nullptr;
    Tensor* b_q = nullptr; Tensor* b_k = nullptr; Tensor* b_v = nullptr;
    Tensor* cos = nullptr; Tensor* sin = nullptr;
    Tensor* norm2_weight  = nullptr;
    Tensor* w_gate = nullptr; Tensor* w_up = nullptr; Tensor* w_down = nullptr;
    Tensor* w_gate_pack = nullptr; Tensor* w_up_pack = nullptr; Tensor* w_down_pack = nullptr;

    bool has_packed_weights = inputs.size() >= 22;
    if (has_packed_weights) {
        w_q_pack = inputs[6]; w_k_pack = inputs[7]; w_v_pack = inputs[8]; w_o_pack = inputs[9];
        b_q = inputs[10]; b_k = inputs[11]; b_v = inputs[12];
        cos = inputs[13]; sin = inputs[14];
        norm2_weight = inputs[15];
        w_gate = inputs[16]; w_up = inputs[17]; w_down = inputs[18];
        w_gate_pack = inputs[19]; w_up_pack = inputs[20]; w_down_pack = inputs[21];
    } else {
        b_q = inputs[6]; b_k = inputs[7]; b_v = inputs[8];
        cos = inputs[9]; sin = inputs[10];
        norm2_weight = inputs[11];
        w_gate = inputs[12]; w_up = inputs[13]; w_down = inputs[14];
    }

    int num_tokens = hidden_states->shape[0];
    int hidden_dim = hidden_states->shape[1];

    // ==========================================
    // 2. 精确计算当前前向传播需要的 Workspace 大小
    // ==========================================
    // (1) 内部残差和归一化缓存：需要存 residual 和 norm_out，共2个 Tensor
    size_t buffer_bytes = 2 * num_tokens * hidden_dim * sizeof(float);
    
    // (2) 矩阵乘法的 Pack 缓存（给 NEON 底层计算预留）
    // 正常给 1MB~2MB 就足够处理 Qwen-0.5B 的 Decode 和小批量 Prefill 了
    size_t pack_bytes = 32 * 1024 * 1024; 
    
    size_t total_ws_size = buffer_bytes + pack_bytes;

    // ==========================================
    // 3. 从全局内存池临时申请并包装为 Workspace
    // ==========================================
    void* raw_ptr = g_memory_pool->allocate(total_ws_size);
    if (!raw_ptr) {
        return Status::OUT_OF_MEMORY;
    }

    // 调用我们在上一节修改好的“无开销子工作区”构造函数
    Workspace temp_workspace(raw_ptr, total_ws_size);

    // ==========================================
    // 4. 调用后端的纯数学计算算子
    // ==========================================
    Status status = Status::SUCCESS;
    if (has_packed_weights) {
        status = arm_neon::qwen_block_neon(
            *hidden_states, *norm1_weight,
            *w_q, *w_k, *w_v, *w_o,
            *w_q_pack, *w_k_pack, *w_v_pack, *w_o_pack,
            b_q->ptr<float>(), b_k->ptr<float>(), b_v->ptr<float>(),
            cos->ptr<float>(), sin->ptr<float>(),
            *norm2_weight,
            *w_gate, *w_up, *w_down,
            *w_gate_pack, *w_up_pack, *w_down_pack,
            *kv_cache,
            layer_id,
            *current_pos_ptr,
            attn_config, ffn_config, rms_norm_eps,
            temp_workspace
        );
    } else {
        status = arm_neon::qwen_block_neon(
            *hidden_states, *norm1_weight,
            *w_q, *w_k, *w_v, *w_o,
            b_q->ptr<float>(), b_k->ptr<float>(), b_v->ptr<float>(),
            cos->ptr<float>(), sin->ptr<float>(),
            *norm2_weight, *w_gate, *w_up, *w_down,
            *kv_cache,
            layer_id,
            *current_pos_ptr,
            attn_config, ffn_config, rms_norm_eps,
            temp_workspace
        );
    }

    // ==========================================
    // 5. 立即释放内存（完璧归赵）
    // ==========================================
    g_memory_pool->free_block(raw_ptr);

    return status;
}

void QwenBlockNode::set_external_workspace(void* workspace, size_t bytes) {
    external_workspace = workspace;
    external_workspace_bytes = bytes;
}


/*                                       创建 tensor                                         */
Tensor* ComputationGraph::create_tensor(
    const std::vector<int>& shape,
    DataType dtype
) {
    // 创建一个 unique_ptr<Tensor> ,所有权属于 tensors, 当ComputationGraph被销毁时，所有的Tensor也会被自动销毁。
    tensors.push_back(std::make_unique<Tensor>(shape, dtype));
    return tensors.back().get();
}

Tensor* ComputationGraph::create_tensor_from_ptr(
    const std::vector<int>& shape,
    void* data,
    DataType dtype
) {
    auto t = std::make_unique<Tensor>(shape, dtype);
    // 假设你的 Tensor 类里有 data 和 owns_data 成员
    t->data = data; 
    t->owns_data = false; // 极其关键：告诉引擎不要去 free 它！
    tensors.push_back(std::move(t));
    return tensors.back().get();
}

/*                                      添加 node                                          */
MatmulNode* ComputationGraph::add_matmul(
    Tensor* A,
    Tensor* B,
    Tensor* C
) {
    nodes.push_back(std::make_unique<MatmulNode>(A, B, C));

    // static_cast 把通用的基类指针，变回具体的子类指针。
    return static_cast<MatmulNode*>(nodes.back().get());
}

AddNode* ComputationGraph::add_add(
    Tensor* A,
    Tensor* B,
    Tensor* C
) {
    nodes.push_back(std::make_unique<AddNode>(A, B, C));
    return static_cast<AddNode*>(nodes.back().get());
}

RMSNormNode* ComputationGraph::add_rmsnorm(
    Tensor* X,
    Tensor* Weight, 
    Tensor* Y, 
    float eps
) {
    nodes.push_back(std::make_unique<RMSNormNode>(X, Weight, Y, eps));
    return static_cast<RMSNormNode*>(nodes.back().get());
}

SwiGLUNode* ComputationGraph::add_swiglu(Tensor* Gate,
    Tensor* Up, 
    Tensor* Y
) {
    nodes.push_back(std::make_unique<SwiGLUNode>(Gate, Up, Y));
    return static_cast<SwiGLUNode*>(nodes.back().get());
}

RoPENode* ComputationGraph::add_rope(Tensor* X, 
    Tensor* Cos,
    Tensor* Sin
) {
    nodes.push_back(std::make_unique<RoPENode>(X, Cos, Sin));
    return static_cast<RoPENode*>(nodes.back().get());
}

QwenBlockNode::QwenBlockNode(
    Tensor* hidden_states, Tensor* norm1_weight,
    Tensor* b_q, Tensor* b_k, Tensor* b_v,
    Tensor* cos, Tensor* sin,
    Tensor* norm2_weight,
    QwenBlockWeights* block_weights,
    KVCache* cache, int l_id, int* pos_ptr,
    arm_neon::AttentionConfig a_conf, arm_neon::FFNConfig f_conf, float eps)
    : attn_config(a_conf), ffn_config(f_conf), weights(block_weights),
      rms_norm_eps(eps), layer_id(l_id), current_pos_ptr(pos_ptr), kv_cache(cache)
{
    inputs = {hidden_states, norm1_weight, b_q, b_k, b_v, cos, sin, norm2_weight};
    outputs = {hidden_states};
}

QwenBlockNode* ComputationGraph::add_qwen_block(
    Tensor* hidden_states, Tensor* norm1_weight,
    Tensor* b_q, Tensor* b_k, Tensor* b_v,
    Tensor* cos, Tensor* sin,
    Tensor* norm2_weight,
    QwenBlockWeights* weights,
    KVCache* cache, int l_id, int* pos_ptr,
    arm_neon::AttentionConfig a_conf, arm_neon::FFNConfig f_conf, float eps
) {
    nodes.push_back(std::make_unique<QwenBlockNode>(
        hidden_states, norm1_weight,
        b_q, b_k, b_v,
        cos, sin,
        norm2_weight,
        weights,
        cache, l_id, pos_ptr, a_conf, f_conf, eps
    ));
    return static_cast<QwenBlockNode*>(nodes.back().get());
}


QwenBlockNode* ComputationGraph::add_qwen_block(
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
) {
    nodes.push_back(std::make_unique<QwenBlockNode>(
        hidden_states, norm1_weight,
        w_q, w_k, w_v, w_o,
        w_q_pack, w_k_pack, w_v_pack, w_o_pack,
        b_q, b_k, b_v,
        cos, sin, norm2_weight,
        w_gate, w_up, w_down,
        w_gate_pack, w_up_pack, w_down_pack,
        cache, l_id, pos_ptr, a_conf, f_conf, eps
    ));

    return static_cast<QwenBlockNode*>(nodes.back().get());
}

QwenBlockNode* ComputationGraph::add_qwen_block(
    Tensor* hidden_states, Tensor* norm1_weight,
    Tensor* w_q, Tensor* w_k, Tensor* w_v, Tensor* w_o,
    Tensor* b_q, Tensor* b_k, Tensor* b_v,
    Tensor* cos, Tensor* sin,
    Tensor* norm2_weight,
    Tensor* w_gate, Tensor* w_up, Tensor* w_down,
    KVCache* cache, int l_id, int* pos_ptr,
    arm_neon::AttentionConfig a_conf, arm_neon::FFNConfig f_conf, float eps
) {
    nodes.push_back(std::make_unique<QwenBlockNode>(
        hidden_states, norm1_weight,
        w_q, w_k, w_v, w_o, b_q, b_k, b_v,
        cos, sin, norm2_weight, w_gate, w_up, w_down,
        cache, l_id, pos_ptr, a_conf, f_conf, eps
    ));

    // 完美对齐你的架构：把通用的基类指针变回具体的子类指针并返回
    return static_cast<QwenBlockNode*>(nodes.back().get());
}

}
