#include "model.h"
#include "src/backends/cpu/arm_neon/neon_ops.h"
#include <fstream>
#include <iostream>
#include <cstdlib>
#include <cmath>

namespace llm_engine {

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

// --- 构造函数：根据配置构建完整的模型骨架 ---
QwenModel::QwenModel(const QwenConfig& cfg) : config(cfg) {
    layers.resize(config.num_layers);

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

        // Bias 是一维的
        allocate_weight(layer.b_q, {config.num_q_heads * config.head_dim});
        allocate_weight(layer.b_k, {config.num_kv_heads * config.head_dim});
        allocate_weight(layer.b_v, {config.num_kv_heads * config.head_dim});

        allocate_weight(layer.norm2_w, {config.hidden_dim});
        
        // FFN 也已转置
        allocate_weight(layer.w_gate, {config.hidden_dim, config.intermediate_size});
        allocate_weight(layer.w_up, {config.hidden_dim, config.intermediate_size});
        allocate_weight(layer.w_down, {config.intermediate_size, config.hidden_dim});
    }

    // 2. 最后的输出层
    allocate_weight(final_norm_w, {config.hidden_dim});
    allocate_weight(lm_head_w, {config.hidden_dim, config.vocab_size}); // 已转置

    // 新增：给 lm_head_w_T 分配内存，形状是 [N, K] = [vocab_size, hidden_dim]
    allocate_weight(lm_head_w_T, {config.vocab_size, config.hidden_dim});
}

// --- 析构函数：统一释放 malloc 的权重 ---
QwenModel::~QwenModel() {
    for (void* ptr : weight_ptrs) {
        std::free(ptr);
    }
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
    
    if (!load_tensor_from_bin(weights_dir + "/lm_head_w_T.bin", lm_head_w_T)) return false;
    
    std::cout << "所有权重加载成功！" << std::endl;
    return true;
}

void QwenModel::init_rope_cache() {
    int max_seq_len = config.max_seq_len; // 例如 2048
    int head_dim = config.head_dim;       // 例如 64
    
    // 分配内存
    allocate_weight(cos_cache, {max_seq_len, head_dim});
    allocate_weight(sin_cache, {max_seq_len, head_dim});
    
    float* cos_ptr = cos_cache.ptr<float>();
    float* sin_ptr = sin_cache.ptr<float>();
    
    // Qwen2.5 默认的 base 是 1000000.0 (10^6)
    float base = 500000.0f;
    
    for (int pos = 0; pos < max_seq_len; pos++) {
        for (int i = 0; i < head_dim; i += 2) {
            float inv_freq = 1.0f / std::pow(base, (float)i / head_dim);
            float freq = pos * inv_freq;
            
            cos_ptr[pos * head_dim + i] = std::cos(freq);
            cos_ptr[pos * head_dim + i + 1] = std::cos(freq);
            
            sin_ptr[pos * head_dim + i] = std::sin(freq);
            sin_ptr[pos * head_dim + i + 1] = std::sin(freq);
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

    // 2. 利用巧妙的"虚拟边界"处理 RoPE 的动态位移
    // 随便绑定一个初始指针，骗过编译器的防线。真正执行时在 forward 里覆盖！
    t_cos = graph.create_tensor_from_ptr({1, head_dim}, cos_cache.ptr<float>(), DataType::FP32);
    t_sin = graph.create_tensor_from_ptr({1, head_dim}, sin_cache.ptr<float>(), DataType::FP32);

    // 3. 创建图内部代管的临时张量 (编译器 Arena 会自动为它们复用内存)
    t_norm_out = graph.create_tensor({1, hidden_dim}, DataType::FP32);
    t_logits = graph.create_tensor({1, vocab_size}, DataType::FP32);

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
            &layer.b_q, &layer.b_k, &layer.b_v,
            t_cos, t_sin,
            &layer.norm2_w, 
            &layer.w_gate, &layer.w_up, &layer.w_down,
            &kv_cache, i, &current_pos, // 传入 current_pos 的指针
            attn_cfg, ffn_cfg, config.rms_norm_eps
        );
    }

    // 5. 全局 RMSNorm 层
    graph.add_rmsnorm(t_hidden_states, &final_norm_w, t_norm_out, config.rms_norm_eps);

    // 6. 最终的 LM Head 映射
    // 注意：Python 导出的 lm_head_w 通常是转置过的，底层 Matmul 默认不转置B时刚好吻合
    // graph.add_matmul(t_norm_out, &lm_head_w, t_logits);
    graph.add_gemv(t_norm_out, &lm_head_w_T, t_logits);

    // 7. 触发编译器：执行生命周期推断、申请大块 Arena 内存并规划所有临时变量！
    plan = compiler.compile(graph);
    is_graph_built = true;
    
    std::cout << "[Info] Computation Graph built successfully! Total nodes: " << plan.size() << "\n";
}

int QwenModel::forward(int token_id, int pos, KVCache& kv_cache) {
    // 1. 首个 Token 进来时，触发建图
    if (!is_graph_built) {
        build_graph(kv_cache);
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
    CHECK_STATUS(runtime.run(plan));

    // 6. Argmax (贪心搜索：找出概率最大的词)
    int next_token = 0;
    float max_val = -1e9f;
    float* logits_ptr = t_logits->ptr<float>(); // 直接从图的出口拿结果
    
    for (int v = 0; v < config.vocab_size; ++v) {
        if (logits_ptr[v] > max_val) {
            max_val = logits_ptr[v];
            next_token = v;
        }
    }

    // Note: 传入的 workspace 参数已经不再需要使用了，因为
    // 中间结果 (logits, norm_out) 由 GraphCompiler 的 Arena 接管。
    // 而底层的 QwenBlockNeon 和 Matmul 等计算，也直接调用了全局 g_memory_pool。

    return next_token;
}

void QwenModel::generate(const std::vector<int>& input_tokens, int max_new_tokens, std::function<bool(int)> callback) {
    if (input_tokens.empty()) return;

    // 1. 初始化属于这一次对话的 KV Cache
    KVCache kv_cache(config.num_layers, config.max_seq_len, config.num_kv_heads, config.head_dim);
    int pos = 0;

    // ==========================================
    // 阶段 A：Prefill（预填充阶段）
    // ==========================================
    // 将前面的 Token 逐个喂给模型，累积 KV Cache，但不需要它们的输出
    for (size_t i = 0; i < input_tokens.size() - 1; ++i) {
        forward(input_tokens[i], pos, kv_cache);
        pos++;
    }

    // ==========================================
    // 阶段 B：Decode（解码阶段）
    // ==========================================
    // 拿出 Prompt 的最后一个词
    int current_token = input_tokens.back();

    for (int i = 0; i < max_new_tokens; ++i) {
        // 推理出下一个词
        int next_token = forward(current_token, pos, kv_cache);
        pos++;

        // 将生成的词通过回调函数送回前端（比如打印到屏幕）
        // 如果回调函数返回 false，则提前终止生成（可用于实现用户强行打断）
        if (!callback(next_token)) {
            break;
        }

        // 检查是否遇到了 Qwen 的对话结束符 (通常是 151645 <|im_end|>)
        if (next_token == 151645 || next_token == 151643) {
            break;
        }

        // 把刚生成的词变成下一次的输入
        current_token = next_token;
    }
}

} // namespace llm_engine