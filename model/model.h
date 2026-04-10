#include "llm_engine/tensor.h"
#include "llm_engine/memory/kv_cache.h"
#include "llm_engine/memory/workspace.h"
#include "llm_engine/graph/graph.h"
#include "llm_engine/graph/compiler.h"
#include "backends/cpu/arm_neon/neon_ops.h"
#include <vector>
#include <string>
#include <functional>

namespace llm_engine {

// Qwen 模型的配置参数 (对应 0.5B)
struct QwenConfig {
    int num_layers = 24;
    int hidden_dim = 896;
    int intermediate_size = 4864;
    int num_q_heads = 14;
    int num_kv_heads = 2;
    int head_dim = 64;
    int vocab_size = 151936; 
    float rms_norm_eps = 1e-6;
    int max_seq_len = 2048; // 你支持的最大上下文长度
};

// 单个 Block 的权重容器
struct QwenBlockWeights {
    Tensor norm1_w, norm2_w;
    Tensor w_q, w_k, w_v, w_o;
    Tensor b_q, b_k, b_v;
    Tensor w_gate, w_up, w_down;
};

class QwenModel {
public:
    QwenConfig config;
    
    // 0. 词表嵌入层权重
    Tensor embed_tokens_w;
    
    // 1. 24 层 Transformer 权重
    std::vector<QwenBlockWeights> layers;
    
    // 2. 最后的输出层权重
    Tensor final_norm_w;
    Tensor lm_head_w;
    Tensor lm_head_w_T;     // 新增：[N, K] 专供 decode 阶段的 GEMV 极速版使用！

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
    std::vector<float> ext_hidden_states; // 图的入口物理内存
    Tensor* t_hidden_states = nullptr;    // 图入口张量 (X)
    std::vector<float> ext_logits;        // 💡 新增：图的出口物理内存
    Tensor* t_norm_out = nullptr;         // 图中间张量
    Tensor* t_logits = nullptr;           // 图出口张量
    
    // 动态边界指针（极其巧妙的零开销技巧：每步只需修改它们的 data 指向）
    Tensor* t_cos = nullptr;
    Tensor* t_sin = nullptr;

    int current_pos = 0; 
    bool is_graph_built = false;

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

private:
    // 内部辅助函数：分配固定内存并绑定给 Tensor
    void allocate_weight(Tensor& t, const std::vector<int>& shape);
    // 内部辅助函数：读取二进制文件
    bool load_tensor_from_bin(const std::string& filepath, Tensor& tensor);
    void init_rope_cache();
    void build_graph(KVCache& kv_cache);
};

} // namespace llm_engine