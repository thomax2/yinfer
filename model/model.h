#pragma once

#include "llm_engine/tensor.h"
#include "llm_engine/cache/prefix_cache.h"
#include "llm_engine/memory/kv_cache.h"
#include "llm_engine/memory/kv_cache_manager.h"
#include "llm_engine/memory/workspace.h"
#include "llm_engine/graph/graph.h"
#include "llm_engine/graph/compiler.h"
#include "llm_engine/engine/sampling.h"
#include "llm_engine/engine/sequence_state.h"
#include "llm_engine/runtime/thread_pool.h"
#include "backends/cpu/arm_neon/neon_ops.h"
#include <memory>
#include <random>
#include <vector>
#include <string>
#include <functional>

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
