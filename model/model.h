#pragma once

#include "llm_engine/tensor.h"
#include "llm_engine/cache/prefix_cache.h"
#include "llm_engine/memory/kv_cache.h"
#include "llm_engine/memory/kv_cache_manager.h"
#include "llm_engine/memory/workspace.h"
#include "llm_engine/graph/graph.h"
#include "llm_engine/graph/compiler.h"
#include "llm_engine/graph/executable_plan.h"
#include "llm_engine/engine/sampling.h"
#include "llm_engine/engine/sequence_state.h"
#include "llm_engine/runtime/thread_pool.h"
#include "backends/cpu/arm_neon/neon_ops.h"
#include <memory>
#include <array>
#include <random>
#include <vector>
#include <string>
#include <functional>
#include <unordered_map>

namespace llm_engine {

// Qwen2.5-1.5B-Instruct-GPTQ-Int8 配置参数
struct QwenConfig {
    int num_layers = 28;
    int hidden_dim = 1536;
    int intermediate_size = 8960;
    int num_q_heads = 12;
    int num_kv_heads = 2;
    int head_dim = 128;
    int vocab_size = 151936; 
    float rms_norm_eps = 1e-6f;
    int max_seq_len = 8192;
    float rope_theta = 1000000.0f;
    bool tie_word_embeddings = true;
};

class QwenModel {
public:
    QwenConfig config;
    
    // 0. 词表嵌入层权重
    Tensor embed_tokens_w;
    
    // 1. Transformer 权重
    std::vector<QwenBlockWeights> layers;
    
    // 2. 最后的输出层权重
    Tensor final_norm_w;
    arm_neon::GPTQInt8Weight lm_head;

    // RoPE 查表缓存
    Tensor cos_cache;
    Tensor sin_cache;

    // 记录所有权重的内存指针，用于析构释放
    std::vector<void*> weight_ptrs;


    // ==========================================
    // 🔥 新增：计算图核心组件
    // ==========================================
    ComputationGraph graph;
    GraphCompiler compiler;
    GraphRuntime runtime;
    std::vector<GraphNode*> plan;

    // Lowering 后的单 token Decode 执行计划。高层 QwenBlock 不参与运行，
    // plan 持有 RMSNorm/Attention/Residual/FFN/LMHead 等中粒度 KernelNode。
    std::unique_ptr<ExecutablePlan> fine_decode_plan; // 单 token Decode 固定执行计划
    ExecutionWorkspace fine_execution_workspace; // Decode 计划的 Arena
    PlanValueId fine_hidden_input = kInvalidPlanValue; // Decode 隐藏层输入 ID
    PlanValueId fine_norm_output = kInvalidPlanValue; // Decode LayerNorm 输出 ID
    PlanValueId fine_argmax_output = kInvalidPlanValue; // Decode ArgMax 输出 ID
    size_t coarse_decode_workspace_bytes = 0; // 粗粒度解码工作区字节数
    bool use_fine_decode_plan = false; // 是否启用精粒度解码计划

    struct BoundExecutablePlan {
        std::unique_ptr<ExecutablePlan> plan; // 可执行计划
        PlanValueId hidden_input = kInvalidPlanValue; // 隐藏层输入 ID
        PlanValueId norm_output = kInvalidPlanValue; // LayerNorm 输出 ID
        PlanValueId argmax_output = kInvalidPlanValue; // ArgMax 输出 ID
    };
    std::unordered_map<int, BoundExecutablePlan> fine_prefill_plans; // 不同 token capacity 对应的 Prefill 计划缓存
    ExecutionWorkspace fine_prefill_workspace; // 所有 Prefill capacity 计划共享的可扩容 Arena
    // Scheduler 开启后，Prefill 与 Decode 共用这一份 token-major Mixed Plan。
    // Plan 保存节点依赖，实际 token 行数由 ExecutionContext::actual_rows 指定。
    std::unique_ptr<ExecutablePlan> fine_mixed_batch_plan;
    ExecutionWorkspace fine_mixed_control_workspace;

    // 图的边界张量和中间张量指针
    std::vector<fp16_t> ext_hidden_states; // 图的入口物理内存
    Tensor* t_hidden_states = nullptr;    // 图入口张量 (X)
    std::vector<fp16_t> ext_norm_out;      // final RMSNorm 的外部输出物理内存
    Tensor* t_norm_out = nullptr;         // 图中间张量

    // 动态边界指针（极其巧妙的零开销技巧：每步只需修改它们的 data 指向）
    Tensor* t_cos = nullptr;
    Tensor* t_sin = nullptr;

    int current_pos = 0;
    bool is_graph_built = false;

    // ========== 关键修改1：使用智能指针管理 KV Cache ==========
    std::unique_ptr<KVCache> kv_cache;

    // ========== 持久线程池（算子内部并行复用）==========
    int num_threads = 4;
    std::unique_ptr<ThreadPool> thread_pool;
    void* block_workspace = nullptr;
    size_t block_workspace_bytes = 0;
    // Mixed Batch 的计划容量与持久 Workspace 都在 Scheduler 初始化时固定。
    // 运行阶段只复用这块内存，不允许根据本轮 token 数量扩容。
    void* mixed_batch_workspace = nullptr;
    size_t mixed_batch_workspace_bytes = 0;
    int mixed_batch_workspace_max_rows = 0;
    uint64_t mixed_batch_workspace_reallocations = 0;
    int mixed_batch_row_capacity = 0;

    // 【新增】：全局历史位置追踪
    int history_pos = 0; 

    struct PrefillChunkStats {
        bool real_batch_used = false;
        bool token_loop_used = false;
        bool fallback = false;
        bool compare_mismatch = false;
        double batch_ms = 0.0;
        double token_loop_ms = 0.0;
    };

    PrefillChunkStats last_prefill_chunk_stats;

    enum class MixedBatchItemKind {
        DECODE,
        PREFILL
    };

    // 一个 item 对应同一个 Sequence 上的一段连续 token 行。
    // Decode 的 row_count 恒为 1；Prefill 的 row_count 是本轮分到的 chunk 大小。
    struct MixedBatchItem {
        MixedBatchItemKind kind = MixedBatchItemKind::DECODE;
        SequenceState* seq = nullptr;
        int row_begin = 0;
        int row_count = 0;
        int start_position = 0;
        bool requires_logits = false;
    };

    struct MixedBatchOutput {
        int next_token = -1;
        bool success = false;
        std::string error_message;
    };

    struct MixedBatchStats {
        int item_count = 0;
        int total_rows = 0;
        int decode_rows = 0;
        int prefill_rows = 0;
        int linear_batch_rows = 0;
        int attention_sequence_segments = 0;
        int lm_head_rows = 0;
        bool state_modified = false;
        uint64_t gptq_batch_kernel_calls = 0;
        uint64_t gptq_batch_rows_total = 0;
        uint64_t gptq_batch_output_panel_tasks = 0;
        uint64_t gptq_batch_row_gemv_fallbacks = 0;
        uint64_t gptq_batch_weight_vector_loads = 0;
        uint64_t gptq_batch_dequant_vector_ops = 0;
        uint64_t gptq_batch_argmax_calls = 0;
        uint64_t gptq_batch_argmax_rows = 0;
        uint64_t gptq_batch_full_logits_elements_written = 0;
        uint64_t gptq_batch_compare_mismatches = 0;
        uint64_t mixed_batch_hotpath_allocations = 0;
        uint64_t mixed_batch_workspace_reallocations = 0;
        double model_ms = 0.0;
    };

    // 兼容旧的 Decode-only 调用入口。Scheduler 的主路径已经改用 Mixed Batch，
    // 这组类型仅供保守回退和现有外部调用继续编译，内部仍转成 MixedBatchItem。
    struct SelectiveDecodeItem {
        SequenceState* seq = nullptr;
        int input_token = -1;
    };

    struct SelectiveDecodeOutput {
        int next_token = -1;
        bool success = false;
        std::string error_message;
    };

    struct SelectiveDecodeStats {
        int batch_size = 0;
        int linear_batch_rows = 0;
        int attention_per_sequence_calls = 0;
        int lm_head_rows = 0;
        bool state_modified = false;
        uint64_t gptq_batch_kernel_calls = 0;
        uint64_t gptq_batch_rows_total = 0;
        uint64_t gptq_batch_output_panel_tasks = 0;
        uint64_t gptq_batch_row_gemv_fallbacks = 0;
        uint64_t gptq_batch_weight_vector_loads = 0;
        uint64_t gptq_batch_dequant_vector_ops = 0;
        uint64_t gptq_batch_argmax_calls = 0;
        uint64_t gptq_batch_argmax_rows = 0;
        uint64_t gptq_batch_full_logits_elements_written = 0;
        uint64_t gptq_batch_compare_mismatches = 0;
        uint64_t selective_decode_hotpath_allocations = 0;
        uint64_t selective_decode_workspace_reallocations = 0;
        double model_ms = 0.0;
    };
    
    // 【新增】：提供一个手动清空记忆的接口
    void clear_history() {
        if (kv_cache) kv_cache->clear();
        history_pos = 0;
    }

    QwenModel(const QwenConfig& cfg);
    ~QwenModel();

    // 从 /weights 目录加载所有权重
    bool load_weights(const std::string& weights_dir);

    // 完整的前向传播，输入当前 token，输出预测的下一个 token
    int forward(int token_id, int current_pos, KVCache& kv_cache);


    // 新增完整的生成大循环
    // input_tokens: 用户输入的提示词转换成的 ID 数组
    // max_new_tokens: 最多生成多少个新词
    // callback: 每生成一个新词，就会触发这个回调函数，用于流式打印
    void generate(
        const std::vector<int>& input_tokens, 
        int max_new_tokens, 
        std::function<bool(int)> callback
    );

    void generate_for_sequence(
        SequenceState& seq,
        const std::vector<int>& input_tokens,
        int max_new_tokens,
        KVCacheManager& kv_manager,
        PrefixCache* prefix_cache,
        std::function<bool(int)> callback
    );

    int prefill_one_for_sequence(
        SequenceState& seq,
        const std::vector<int>& prompt_tokens,
        int prompt_index,
        KVCacheManager& kv_manager,
        PrefixCache* prefix_cache
    );

    int prefill_chunk_for_sequence(
        SequenceState& seq,
        const std::vector<int>& prompt_tokens,
        int prompt_begin,
        int prompt_end,
        KVCacheManager& kv_manager,
        PrefixCache* prefix_cache
    );

    int decode_one_for_sequence(
        SequenceState& seq,
        int input_token,
        KVCacheManager& kv_manager,
        PrefixCache* prefix_cache
    );

    int decode_one_for_sequence_sampled(
        SequenceState& seq,
        int input_token,
        KVCacheManager& kv_manager,
        PrefixCache* prefix_cache,
        const SamplingParams& sampling,
        std::mt19937_64& rng,
        SamplingRuntimeStats* stats
    );

    // Scheduler 初始化阶段构建并编译固定容量的 Mixed Selective Batch 计划，
    // 同时按 row_capacity 一次性分配持久 Workspace。
    // 相同容量的重复调用幂等；不同容量的重复调用属于配置错误。
    void build_mixed_batch_plan(int row_capacity);

    bool run_mixed_batch_for_sequences(
        const std::vector<int>& token_ids,
        const std::vector<MixedBatchItem>& items,
        KVCacheManager& kv_manager,
        PrefixCache* prefix_cache,
        std::vector<MixedBatchOutput>* outputs,
        MixedBatchStats* stats
    );

    bool decode_selective_batch_for_sequences(
        const std::vector<SelectiveDecodeItem>& items,
        KVCacheManager& kv_manager,
        PrefixCache* prefix_cache,
        std::vector<SelectiveDecodeOutput>* outputs,
        SelectiveDecodeStats* stats
    );

    int sample_next_token_from_last_logits(
        const SamplingParams& sampling,
        std::mt19937_64& rng,
        SamplingRuntimeStats* stats
    );

private:
    // 内部辅助函数：分配固定内存并绑定给 Tensor
    void allocate_tensor(Tensor& t, const std::vector<int>& shape, DataType dtype);
    void allocate_gptq_weight(arm_neon::GPTQInt8Weight& w, int K, int N, int group_size = 128);
    // 内部辅助函数：读取二进制文件
    bool load_tensor_from_bin(const std::string& filepath, Tensor& tensor);
    bool load_gptq_weight_from_bins(const std::string& prefix, arm_neon::GPTQInt8Weight& w);
    void init_rope_cache();
    void build_graph(KVCache& kv_cache);
    void ensure_block_workspace();
    bool ensure_mixed_batch_workspace(int max_rows);

    struct MixedBatchWorkspaceView {
        fp16_t* hidden = nullptr;
        fp16_t* residual = nullptr;
        fp16_t* norm = nullptr;
        // Mixed 普通 Linear 统一使用的 Packed A 缓冲区。
        // 容量按 max(hidden_dim, q_size, intermediate_size) 预留，布局为
        // [ceil(max_rows/8), K, 8]；不同阶段按当前 Linear 的 K 覆盖复用。
        fp16_t* packed_a = nullptr;
        fp16_t* q = nullptr;
        fp16_t* k = nullptr;
        fp16_t* v = nullptr;
        fp16_t* attn_out = nullptr;
        fp16_t* proj_out = nullptr;
        fp16_t* gate = nullptr;
        fp16_t* up = nullptr;
        fp16_t* score = nullptr;
        // Paged Prefill QK^T 的 16-token K 转置区：[head_dim, 16]。
        fp16_t* attention_k_tile = nullptr;
        // Online Paged Prefill 的归一化前 FP32 输出累加器：[4, head_dim]。
        // 它由不同 Query Tile / Q Head 复用，容量不随历史长度增长。
        float* attention_online_out = nullptr;
        void* argmax_partials = nullptr;
        size_t argmax_partials_bytes = 0;
        size_t bytes = 0;
    };

    struct MixedPlanRunState {
        const std::vector<MixedBatchItem>* items = nullptr;
        std::vector<MixedBatchOutput>* outputs = nullptr;
        MixedBatchStats* stats = nullptr;
        MixedBatchWorkspaceView ws;
        const std::vector<int>* positions = nullptr;
        const std::vector<int>* logit_rows = nullptr;
        std::vector<arm_neon::ArgmaxResult>* argmax_results = nullptr;
        int total_rows = 0;
        bool paged_attention_requested = false;
        bool strict_paged_attention = false;
        std::string error;
    };

    MixedBatchWorkspaceView mixed_batch_workspace_view(int total_rows);
    int prefill_prompt_batch(const std::vector<int>& input_tokens, int start_pos, KVCache& kv_cache);

    // ========== Debug 辅助 ==========
    static bool env_flag(const char* name);
    static int env_int(const char* name, int default_value);
    static float env_float(const char* name, float default_value);

    struct ForwardDebugResult {
        int token_id = -1;
        float logit = 0.0f;
    };

    struct DebugTopKItem {
        int id;
        float value;
    };

    ForwardDebugResult forward_debug_last_result;
    std::vector<fp16_t> last_batch_norm_out_;
    bool last_batch_norm_out_valid_ = false;
};

} // namespace llm_engine
