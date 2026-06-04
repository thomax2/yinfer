#include "model.h"
#include "backends/cpu/arm_neon/neon_ops.h"
#include "backends/cpu/arm_neon/pack.h"
#include <fstream>
#include <iostream>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <cstring>
#include <cstdint>
#include <string>

namespace llm_engine {

// ========== Debug 辅助：环境变量解析 ==========
bool QwenModel::env_flag(const char* name) {
    const char* v = std::getenv(name);
    if (!v) return false;
    std::string s(v);
    return s == "1" || s == "true" || s == "TRUE" || s == "on" || s == "ON";
}

int QwenModel::env_int(const char* name, int default_value) {
    const char* v = std::getenv(name);
    if (!v) return default_value;
    char* end = nullptr;
    long x = std::strtol(v, &end, 10);
    if (end == v) return default_value;
    return static_cast<int>(x);
}

float QwenModel::env_float(const char* name, float default_value) {
    const char* v = std::getenv(name);
    if (!v) return default_value;
    char* end = nullptr;
    float x = std::strtof(v, &end);
    if (end == v) return default_value;
    return x;
}

// --- 内存分配：跳过 memory_pool，直接 malloc ---
void QwenModel::allocate_weight(Tensor& t, const std::vector<int>& shape) {
    size_t size = 1;
    for (int dim : shape) size *= dim;
    size_t bytes = size * sizeof(float);
    
    // 使用标准 malloc 分配持久内存
    void* ptr = std::malloc(bytes);
    if (!ptr) {
        std::cerr << "内存分配失败，请求大小: " << bytes << " 字节" << std::endl;
        exit(EXIT_FAILURE);
    }
    
    weight_ptrs.push_back(ptr); // 记录指针
    
    // 调用 Tensor.h 中现成的外部指针构造函数
    // 此时 Tensor 的 owns_data 自动为 false，不会触发 g_memory_pool 的析构
    t = Tensor(shape, ptr, DataType::FP32);
}

void QwenModel::allocate_packed_weight(Tensor& t, int K, int N) {
    int np = (N + arm_neon::NR - 1) / arm_neon::NR;
    allocate_weight(t, {np, K, arm_neon::NR});
}

void QwenModel::prepack_all_weights() {
    int q_out_dim = config.num_q_heads * config.head_dim;
    int kv_out_dim = config.num_kv_heads * config.head_dim;

    for (auto& layer : layers) {
        arm_neon::pack_weight_for_linear_decode(layer.w_q.ptr<float>(), layer.w_q_pack.ptr<float>(), config.hidden_dim, q_out_dim);
        arm_neon::pack_weight_for_linear_decode(layer.w_k.ptr<float>(), layer.w_k_pack.ptr<float>(), config.hidden_dim, kv_out_dim);
        arm_neon::pack_weight_for_linear_decode(layer.w_v.ptr<float>(), layer.w_v_pack.ptr<float>(), config.hidden_dim, kv_out_dim);
        arm_neon::pack_weight_for_linear_decode(layer.w_o.ptr<float>(), layer.w_o_pack.ptr<float>(), q_out_dim, config.hidden_dim);

        arm_neon::pack_weight_for_linear_decode(layer.w_gate.ptr<float>(), layer.w_gate_pack.ptr<float>(), config.hidden_dim, config.intermediate_size);
        arm_neon::pack_weight_for_linear_decode(layer.w_up.ptr<float>(), layer.w_up_pack.ptr<float>(), config.hidden_dim, config.intermediate_size);
        arm_neon::pack_weight_for_linear_decode(layer.w_down.ptr<float>(), layer.w_down_pack.ptr<float>(), config.intermediate_size, config.hidden_dim);
    }

    arm_neon::pack_weight_for_linear_decode(lm_head_w.ptr<float>(), lm_head_pack.ptr<float>(), config.hidden_dim, config.vocab_size);
}

// --- 构造函数：根据配置构建完整的模型骨架 ---
QwenModel::QwenModel(const QwenConfig& cfg) : config(cfg) {
    // ========== 持久线程池：算子内部并行复用 ==========
    {
        int threads = 4;
        const char* env = std::getenv("LLM_NUM_THREADS");
        if (env) {
            int v = std::atoi(env);
            if (v > 0) threads = v;
        }
        threads = std::max(1, std::min(threads, 8));
        num_threads = threads;
        thread_pool = std::make_unique<ThreadPool>(num_threads);
        g_thread_pool = thread_pool.get();
        std::cout << "[Info] ThreadPool initialized with " << num_threads << " threads" << std::endl;
    }

    layers.resize(config.num_layers);

    // ========== 关键修改2：构造函数中只分配一次 KV Cache ==========
    // 只分配一次，生命周期伴随模型
    kv_cache = std::make_unique<KVCache>(
        config.num_layers, 
        config.max_seq_len, 
        config.num_kv_heads, 
        config.head_dim
    );

    // 0. Embedding 层
    allocate_weight(embed_tokens_w, {config.vocab_size, config.hidden_dim});

    // 1. Transformer 层
    for (int i = 0; i < config.num_layers; ++i) {
        auto& layer = layers[i];
        
        allocate_weight(layer.norm1_w, {config.hidden_dim});
        
        // 注意：线性层已在 Python 端转置，所以是 [in, out]
        allocate_weight(layer.w_q, {config.hidden_dim, config.num_q_heads * config.head_dim});
        allocate_weight(layer.w_k, {config.hidden_dim, config.num_kv_heads * config.head_dim});
        allocate_weight(layer.w_v, {config.hidden_dim, config.num_kv_heads * config.head_dim});
        allocate_weight(layer.w_o, {config.num_q_heads * config.head_dim, config.hidden_dim});

        allocate_packed_weight(layer.w_q_pack, config.hidden_dim, config.num_q_heads * config.head_dim);
        allocate_packed_weight(layer.w_k_pack, config.hidden_dim, config.num_kv_heads * config.head_dim);
        allocate_packed_weight(layer.w_v_pack, config.hidden_dim, config.num_kv_heads * config.head_dim);
        allocate_packed_weight(layer.w_o_pack, config.num_q_heads * config.head_dim, config.hidden_dim);

        // Bias 是一维的
        allocate_weight(layer.b_q, {config.num_q_heads * config.head_dim});
        allocate_weight(layer.b_k, {config.num_kv_heads * config.head_dim});
        allocate_weight(layer.b_v, {config.num_kv_heads * config.head_dim});

        allocate_weight(layer.norm2_w, {config.hidden_dim});
        
        // FFN 也已转置
        allocate_weight(layer.w_gate, {config.hidden_dim, config.intermediate_size});
        allocate_weight(layer.w_up, {config.hidden_dim, config.intermediate_size});
        allocate_weight(layer.w_down, {config.intermediate_size, config.hidden_dim});

        allocate_packed_weight(layer.w_gate_pack, config.hidden_dim, config.intermediate_size);
        allocate_packed_weight(layer.w_up_pack, config.hidden_dim, config.intermediate_size);
        allocate_packed_weight(layer.w_down_pack, config.intermediate_size, config.hidden_dim);
    }

    // 2. 最后的输出层权重
    allocate_weight(final_norm_w, {config.hidden_dim});
    allocate_weight(lm_head_w, {config.hidden_dim, config.vocab_size}); // 已转置

    // LM Head 默认走 lm_head_pack + parallel argmax，不再需要 lm_head_w_T。
    allocate_packed_weight(lm_head_pack, config.hidden_dim, config.vocab_size);

    // 初始化 RoPE 查表缓存，确保图边界张量 t_cos/t_sin 绑定到有效内存
    init_rope_cache();
}

// --- 析构函数：统一释放 malloc 的权重 ---
QwenModel::~QwenModel() {
    // 解绑全局线程池指针，避免悬挂
    if (g_thread_pool == thread_pool.get()) {
        g_thread_pool = nullptr;
    }
    for (void* ptr : weight_ptrs) {
        std::free(ptr);
    }
    // unique_ptr 自动释放 kv_cache / thread_pool，无需手动 delete
}

// --- 文件读取辅助函数 ---
bool QwenModel::load_tensor_from_bin(const std::string& filepath, Tensor& tensor) {
    std::ifstream file(filepath, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        std::cerr << "文件打开失败: " << filepath << std::endl;
        return false;
    }

    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    if (size != tensor.bytes()) {
        std::cerr << "文件大小不匹配: " << filepath 
                  << " (期待: " << tensor.bytes() << " 字节, 实际: " << size << " 字节)" << std::endl;
        return false;
    }

    if (!file.read(reinterpret_cast<char*>(tensor.data), size)) {
        std::cerr << "读取失败: " << filepath << std::endl;
        return false;
    }
    return true;
}

// --- 外部接口：加载整个文件夹 ---
bool QwenModel::load_weights(const std::string& weights_dir) {
    std::cout << "开始加载模型权重..." << std::endl;
    
    if (!load_tensor_from_bin(weights_dir + "/embed_tokens_w.bin", embed_tokens_w)) return false;

    for (int i = 0; i < config.num_layers; ++i) {
        std::string prefix = weights_dir + "/layer" + std::to_string(i) + "_";
        auto& layer = layers[i];

        if (!load_tensor_from_bin(prefix + "norm1_w.bin", layer.norm1_w)) return false;
        if (!load_tensor_from_bin(prefix + "w_q.bin", layer.w_q)) return false;
        if (!load_tensor_from_bin(prefix + "w_k.bin", layer.w_k)) return false;
        if (!load_tensor_from_bin(prefix + "w_v.bin", layer.w_v)) return false;
        if (!load_tensor_from_bin(prefix + "w_o.bin", layer.w_o)) return false;
        if (!load_tensor_from_bin(prefix + "b_q.bin", layer.b_q)) return false;
        if (!load_tensor_from_bin(prefix + "b_k.bin", layer.b_k)) return false;
        if (!load_tensor_from_bin(prefix + "b_v.bin", layer.b_v)) return false;
        
        if (!load_tensor_from_bin(prefix + "norm2_w.bin", layer.norm2_w)) return false;
        if (!load_tensor_from_bin(prefix + "w_gate.bin", layer.w_gate)) return false;
        if (!load_tensor_from_bin(prefix + "w_up.bin", layer.w_up)) return false;
        if (!load_tensor_from_bin(prefix + "w_down.bin", layer.w_down)) return false;
        
        if ((i + 1) % 4 == 0) {
            std::cout << "已加载 " << i + 1 << "/" << config.num_layers << " 层权重..." << std::endl;
        }
    }

    if (!load_tensor_from_bin(weights_dir + "/final_norm_w.bin", final_norm_w)) return false;
    if (!load_tensor_from_bin(weights_dir + "/lm_head_w.bin", lm_head_w)) return false;

    prepack_all_weights();
    
    std::cout << "所有权重加载成功！" << std::endl;
    return true;
}

void QwenModel::init_rope_cache() {
    int max_seq_len = config.max_seq_len; // 例如 2048
    int head_dim = config.head_dim;       // 例如 64
    int half_dim = head_dim / 2;

    // 分配内存
    allocate_weight(cos_cache, {max_seq_len, head_dim});
    allocate_weight(sin_cache, {max_seq_len, head_dim});
    
    float* cos_ptr = cos_cache.ptr<float>();
    float* sin_ptr = sin_cache.ptr<float>();
    
    // Qwen2.5 默认的 base 是 1000000.0 (10^6)
    float base = 1000000.0f;
    
        for (int pos = 0; pos < max_seq_len; ++pos) {
        for (int i = 0; i < half_dim; ++i) {
            // 频率只由前一半的维度下标决定
            float inv_freq = 1.0f / std::pow(base, (float)(i * 2) / head_dim);
            float freq = pos * inv_freq;

            float c = std::cos(freq);
            float s = std::sin(freq);

            // 前半段和后半段的相同位置存相同的 cos / sin
            cos_ptr[pos * head_dim + i] = c;
            cos_ptr[pos * head_dim + i + half_dim] = c;

            sin_ptr[pos * head_dim + i] = s;
            sin_ptr[pos * head_dim + i + half_dim] = s;
        }
    }
}

void QwenModel::build_graph(KVCache& kv_cache) {
    int hidden_dim = config.hidden_dim;
    int vocab_size = config.vocab_size;
    int head_dim = config.head_dim;

    // 1. 准备入口边界内存
    ext_hidden_states.resize(hidden_dim, 0.0f);
    t_hidden_states = graph.create_tensor_from_ptr({1, hidden_dim}, ext_hidden_states.data(), DataType::FP32);
    // 💡 新增：准备出口边界内存
    ext_logits.resize(vocab_size, 0.0f);
    // 💡 新增：final RMSNorm 的输出物理内存（图的最终输出，必须是外部绑定张量）
    ext_norm_out.resize(hidden_dim, 0.0f);

    // 2. 利用巧妙的"虚拟边界"处理 RoPE 的动态位移
    // 随便绑定一个初始指针，骗过编译器的防线。真正执行时在 forward 里覆盖！
    t_cos = graph.create_tensor_from_ptr({1, head_dim}, cos_cache.ptr<float>(), DataType::FP32);
    t_sin = graph.create_tensor_from_ptr({1, head_dim}, sin_cache.ptr<float>(), DataType::FP32);

    // 3. 创建图内部代管的临时张量 (编译器 Arena 会自动为它们复用内存)
    // t_norm_out 是图的最终输出（lm_head 在图外做），必须绑外部 data，
    // 否则 GraphCompiler 会把无消费者的张量判定为边界并报错。
    t_norm_out = graph.create_tensor_from_ptr({1, hidden_dim}, ext_norm_out.data(), DataType::FP32);
    //t_logits = graph.create_tensor({1, vocab_size}, DataType::FP32);
    t_logits = graph.create_tensor_from_ptr({1, vocab_size}, ext_logits.data(), DataType::FP32);

    arm_neon::AttentionConfig attn_cfg = {
        config.hidden_dim, config.num_q_heads, config.num_kv_heads, config.head_dim
    };
    arm_neon::FFNConfig ffn_cfg = {
        config.hidden_dim, config.intermediate_size
    };

    // 4. 搭积木：逐层堆叠 Transformer Block
    for (int i = 0; i < config.num_layers; ++i) {
        auto& layer = layers[i];
        
        // 直接将权重的地址传入图节点中
        graph.add_qwen_block(
            t_hidden_states, 
            &layer.norm1_w, 
            &layer.w_q, &layer.w_k, &layer.w_v, &layer.w_o,
            &layer.w_q_pack, &layer.w_k_pack, &layer.w_v_pack, &layer.w_o_pack,
            &layer.b_q, &layer.b_k, &layer.b_v,
            t_cos, t_sin,
            &layer.norm2_w, 
            &layer.w_gate, &layer.w_up, &layer.w_down,
            &layer.w_gate_pack, &layer.w_up_pack, &layer.w_down_pack,
            &kv_cache, i, &current_pos, // 传入 current_pos 的指针
            attn_cfg, ffn_cfg, config.rms_norm_eps
        );
    }

    // 5. 全局 RMSNorm 层
    graph.add_rmsnorm(t_hidden_states, &final_norm_w, t_norm_out, config.rms_norm_eps);

    // 6. 最终的 LM Head 映射
    //    默认走 lm_head_pack + parallel argmax（在 forward 里完成），
    //    所以 graph 里不再追加 GemvNode。lm_head_w_T 已经移除。

    // 7. 触发编译器：执行生命周期推断、申请大块 Arena 内存并规划所有临时变量！
    plan = compiler.compile(graph);
    is_graph_built = true;
    
    std::cout << "[Info] Computation Graph built successfully! Total nodes: " << plan.size() << "\n";
}

// ========== 关键修改4：forward 函数中 build_graph 传入稳定的地址 ==========
int QwenModel::forward(int token_id, int pos, KVCache& kv_cache) {
    bool dbg_forward = env_flag("LLM_DEBUG_FORWARD");
    if (dbg_forward) {
        std::cerr << "[FWD_BEGIN]"
                  << " pos=" << pos
                  << " token_in=" << token_id
                  << " history_pos_member=" << history_pos
                  << " graph_built=" << is_graph_built
                  << std::endl;
    }

    // 1. 首个 Token 进来时，触发建图
    bool just_built = false;
    if (!is_graph_built) {
        // 传入的是成员变量 kv_cache 的引用，地址永远不变！
        build_graph(kv_cache);
        just_built = true;
    }
    if (dbg_forward && just_built) {
        std::cerr << "[FWD_GRAPH_BUILT]"
                  << " nodes=" << plan.size()
                  << std::endl;
    }

    // 2. 更新控制全局状态的变量 (BlockNode内部透传了这里的指针)
    current_pos = pos;

    // 3. Embedding 查表并拷贝到图的【入口内存】
    float* embed_ptr = embed_tokens_w.ptr<float>() + token_id * config.hidden_dim;
    std::memcpy(t_hidden_states->data, embed_ptr, config.hidden_dim * sizeof(float));

    // 4. 零开销技巧：直接强行覆盖图中动态边界张量的数据指针，指向正确的 RoPE 偏移处！
    t_cos->data = cos_cache.ptr<float>() + current_pos * config.head_dim;
    t_sin->data = sin_cache.ptr<float>() + current_pos * config.head_dim;

    // 5. 触发计算图执行！不需要再关心具体的算子、Workspace 等细节
    if (dbg_forward) {
        std::cerr << "[FWD_RUN_BEGIN] pos=" << pos << std::endl;
    }
    CHECK_STATUS(runtime.run(plan));
    if (dbg_forward) {
        std::cerr << "[FWD_RUN_END] pos=" << pos << std::endl;
    }

    // 6. LM Head：在图外做，使用 packed 权重 + 并行 fused argmax，
    //    避免分配/写入完整 logits 数组。lm_head_w_T fallback 已移除。
    if (lm_head_pack.data == nullptr) {
        std::cerr << "[ERROR] lm_head_pack is null." << std::endl;
        return 0;
    }
    auto rp = arm_neon::linear_decode_prepacked_argmax_parallel_neon(
        t_norm_out->ptr<float>(),
        lm_head_pack.ptr<float>(),
        config.hidden_dim,
        config.vocab_size
    );
    int next_token = rp.index;

    forward_debug_last_result.token_id = next_token;
    forward_debug_last_result.logit = rp.value;

    // 串行/并行一致性检查
    if (env_flag("LLM_DEBUG_ARGMAX_COMPARE")) {
        auto rs = arm_neon::linear_decode_prepacked_argmax_serial_neon(
            t_norm_out->ptr<float>(),
            lm_head_pack.ptr<float>(),
            config.hidden_dim,
            config.vocab_size
        );
        bool same = (rp.index == rs.index);
        std::cerr << "[ARGMAX_COMPARE]"
                  << " pos=" << pos
                  << " parallel_id=" << rp.index
                  << " parallel_val=" << rp.value
                  << " serial_id=" << rs.index
                  << " serial_val=" << rs.value
                  << " same=" << (same ? 1 : 0)
                  << " delta=" << (rp.value - rs.value)
                  << std::endl;
        if (!same) {
            std::cerr << "[ARGMAX_MISMATCH]"
                      << " pos=" << pos
                      << " token_in=" << token_id
                      << std::endl;
        }
    }

    if (dbg_forward) {
        std::cerr << "[FWD_END]"
                  << " pos=" << pos
                  << " token_in=" << token_id
                  << " token_out=" << next_token
                  << " logit=" << rp.value
                  << std::endl;
    }

    // Note: 传入的 workspace 参数已经不再需要使用了，因为
    // 中间结果 (logits, norm_out) 由 GraphCompiler 的 Arena 接管。
    // 而底层的 QwenBlockNeon 和 Matmul 等计算，也直接调用了全局 g_memory_pool。

    return next_token;
}

// ========== 关键修改3：generate 函数中只 clear，不重新分配 ==========
void QwenModel::generate(const std::vector<int>& input_tokens, int max_new_tokens, std::function<bool(int)> callback) {
    if (input_tokens.empty()) return;

    static uint64_t s_turn_id = 0;
    uint64_t turn_id = ++s_turn_id;

    bool dbg_gen = env_flag("LLM_DEBUG_GENERATE");
    bool dbg_prefill = env_flag("LLM_DEBUG_PREFILL");
    bool dbg_decode = env_flag("LLM_DEBUG_DECODE") || dbg_gen;
    bool fix_maxnew_writeback = env_flag("LLM_FIX_MAXNEW_WRITEBACK");

    int history_before = history_pos;
    int prompt_len = static_cast<int>(input_tokens.size());
    int history_after_prefill = -1;
    int emitted_visible = 0;
    std::string stop_reason = "UNKNOWN";

    if (dbg_gen) {
        std::cerr << "[GEN_BEGIN]"
                  << " turn=" << turn_id
                  << " history_before=" << history_before
                  << " prompt_len=" << prompt_len
                  << " max_new_tokens=" << max_new_tokens
                  << " max_seq_len=" << config.max_seq_len
                  << std::endl;

        std::cerr << "[GEN_PROMPT_TOKENS] turn=" << turn_id << " ids=";
        for (size_t i = 0; i < input_tokens.size(); ++i) {
            if (i) std::cerr << ",";
            std::cerr << input_tokens[i];
        }
        std::cerr << std::endl;
    }

    auto print_summary = [&]() {
        std::cerr << "[GEN_END]"
                  << " turn=" << turn_id
                  << " stop=" << stop_reason
                  << " history_before=" << history_before
                  << " prompt_len=" << prompt_len
                  << " history_after_prefill=" << history_after_prefill
                  << " emitted_visible=" << emitted_visible
                  << " history_after=" << history_pos
                  << std::endl;

        int expected_rough = history_before + prompt_len + emitted_visible + 1;
        std::cerr << "[GEN_INVARIANT]"
                  << " turn=" << turn_id
                  << " expected_rough=" << expected_rough
                  << " actual_history=" << history_pos
                  << " diff=" << (history_pos - expected_rough)
                  << " note=rough_expected_assumes_one_closing_token"
                  << std::endl;
    };

    // 【新增安全检查】：如果内存快满了，必须清空，否则会越界崩溃 (Segfault)
    if (history_pos + input_tokens.size() + max_new_tokens >= (config.max_seq_len * 0.9)) {
        if (dbg_gen) {
            std::cerr << "[GEN_CONTEXT_CLEAR]"
                      << " turn=" << turn_id
                      << " history_before_clear=" << history_pos
                      << " prompt_len=" << input_tokens.size()
                      << " max_new_tokens=" << max_new_tokens
                      << std::endl;
        }
        std::cout << "\n[Warning] Context limit reached. Clearing history..." << std::endl;
        clear_history();
        history_before = history_pos; // refresh after clear
    }

    // ==========================================
    // 阶段 A：Prefill（预填充阶段）
    // ==========================================
    // 将前面的 Token 逐个喂给模型，累积 KV Cache，但不需要它们的输出
    for (size_t i = 0; i < input_tokens.size() - 1; ++i) {
        int pos_before = history_pos;
        int out = forward(input_tokens[i], history_pos, *kv_cache);
        history_pos++; // 每次 forward 后全局指针 +1

        if (dbg_prefill) {
            std::cerr << "[PREFILL]"
                      << " turn=" << turn_id
                      << " i=" << i
                      << " pos_before=" << pos_before
                      << " token_in=" << input_tokens[i]
                      << " token_pred=" << out
                      << " pos_after=" << history_pos
                      << std::endl;
        }
    }

    history_after_prefill = history_pos;
    if (dbg_gen) {
        std::cerr << "[GEN_PREFILL_END]"
                  << " turn=" << turn_id
                  << " history_after_prefill=" << history_after_prefill
                  << " current_token=" << input_tokens.back()
                  << std::endl;
    }

    // ==========================================
    // 阶段 B：Decode（解码阶段）
    // ==========================================
    // 拿出 Prompt 的最后一个词
    int current_token = input_tokens.back();

    // assistant 段必须以 <|im_end|> 闭合，否则下一轮 prompt 拼接出来的是格式破损的
    // ChatML，0.5B 这种小模型会被拽回旧 assistant 上下文，出现复读。
    // 三条退出路径（自然 EOS / 回调中断 / 跑满 max_new_tokens）都要把闭合符写进 KV 缓存。
    constexpr int IM_END_ID = 151645;
    constexpr int EOT_ID    = 151643;

    auto inject_im_end = [&]() {
        if (history_pos >= config.max_seq_len) return;
        forward(IM_END_ID, history_pos, *kv_cache);
        history_pos++;
    };

    bool decode_done = false;
    for (int i = 0; i < max_new_tokens && !decode_done; ++i) {
        int pos_before = history_pos;
        int input_to_forward = current_token;
        // 推理出下一个词
        int next_token = forward(current_token, history_pos, *kv_cache);
        history_pos++; // 模型自己生成的 Token 也会顺延存入 KV Cache

        if (dbg_decode) {
            std::cerr << "[DECODE]"
                      << " turn=" << turn_id
                      << " step=" << i
                      << " pos_before=" << pos_before
                      << " token_in=" << input_to_forward
                      << " token_out=" << next_token
                      << " logit=" << forward_debug_last_result.logit
                      << " pos_after=" << history_pos
                      << std::endl;
        }

        // 遇到结束符：把它自己也喂回 forward 一次，让 KV 缓存里真的存有 <|im_end|>，
        // 否则下一轮 prompt 拼到一段没有闭合的 assistant 后面，会复读上一轮内容。
        if (next_token == IM_END_ID || next_token == EOT_ID) {
            stop_reason = (next_token == IM_END_ID) ? "IM_END" : "EOT";
            if (next_token == IM_END_ID) {
                inject_im_end();
            } else {
                if (history_pos < config.max_seq_len) {
                    forward(next_token, history_pos, *kv_cache);
                    history_pos++;
                }
            }
            decode_done = true;
            break;
        }

        // 将生成的词通过回调函数送回前端（比如打印到屏幕）
        // 如果回调函数返回 false，则提前终止生成（可用于实现用户强行打断）
        if (!callback(next_token)) {
            emitted_visible++;
            stop_reason = "CALLBACK_FALSE";
            inject_im_end();
            decode_done = true;
            break;
        }

        emitted_visible++;
        // 把刚生成的词变成下一次的输入
        current_token = next_token;
    }

    if (!decode_done) {
        stop_reason = "MAX_NEW";

        if (dbg_gen) {
            std::cerr << "[MAXNEW_REACHED]"
                      << " turn=" << turn_id
                      << " current_token=" << current_token
                      << " history_pos_before_close=" << history_pos
                      << " fix_writeback=" << (fix_maxnew_writeback ? 1 : 0)
                      << std::endl;
        }

        // 实验性修复：让最后一个已经打印的 visible token 真正写进 KV，
        // 否则下一轮 prompt 拼到的 KV 序列里这个 token 实际不存在，
        // 0.5B 容易被这种状态污染。
        if (fix_maxnew_writeback) {
            if (history_pos < config.max_seq_len) {
                int pos_before = history_pos;
                int pred = forward(current_token, history_pos, *kv_cache);
                history_pos++;
                if (dbg_gen) {
                    std::cerr << "[MAXNEW_WRITEBACK_LAST_VISIBLE]"
                              << " turn=" << turn_id
                              << " pos_before=" << pos_before
                              << " token_written=" << current_token
                              << " pred=" << pred
                              << " pos_after=" << history_pos
                              << std::endl;
                }
            }
        }

        // 跑满 max_new_tokens 也没遇到 EOS：手动追加 <|im_end|>，给本轮 assistant 段封口。
        inject_im_end();
    }

    if (dbg_gen) {
        print_summary();
    }
}


} // namespace llm_engine
