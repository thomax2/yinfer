#include "model.h"
#include "backends/cpu/arm_neon/kernel_common.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <sstream>

namespace llm_engine {

namespace {

constexpr int QWEN_EOT_ID = 151643;      // <|endoftext|>
constexpr int QWEN_IM_START_ID = 151644; // <|im_start|>
constexpr int QWEN_IM_END_ID = 151645;   // <|im_end|>

bool is_stop_token(int token_id) {
    return token_id == QWEN_EOT_ID ||
           token_id == QWEN_IM_START_ID ||
           token_id == QWEN_IM_END_ID;
}

std::string join_path(const std::string& dir, const std::string& name) {
    if (dir.empty()) return name;
    char last = dir.back();
    if (last == '/' || last == '\\') return dir + name;
    return dir + "/" + name;
}

bool file_exists(const std::string& path) {
    std::ifstream fin(path, std::ios::binary);
    return fin.good();
}

bool read_exact_file(const std::string& path, void* dst, size_t bytes) {
    std::ifstream fin(path, std::ios::binary | std::ios::ate);
    if (!fin) {
        std::cerr << "[ERROR] missing weight file: " << path << std::endl;
        return false;
    }
    std::streamsize size = fin.tellg();
    if (size != static_cast<std::streamsize>(bytes)) {
        std::cerr << "[ERROR] weight size mismatch: " << path
                  << " expected=" << bytes
                  << " actual=" << size << std::endl;
        return false;
    }
    fin.seekg(0, std::ios::beg);
    return static_cast<bool>(fin.read(reinterpret_cast<char*>(dst), size));
}

int model_env_int(const char* name, int default_value) {
    const char* v = std::getenv(name);
    if (!v || !*v) return default_value;

    char* end = nullptr;
    long x = std::strtol(v, &end, 10);
    if (end == v) return default_value;
    if (x < 1) return 1;
    return static_cast<int>(x);
}

} // namespace

QwenModel::QwenModel(const QwenConfig& cfg)
    : config(cfg),
      layers(cfg.num_layers),
      kv_cache(std::make_unique<KVCache>(
          cfg.num_layers, cfg.max_seq_len, cfg.num_kv_heads, cfg.head_dim)),
      num_threads(model_env_int("LLM_NUM_THREADS", 4)),
      thread_pool(std::make_unique<ThreadPool>(num_threads))
{
    g_thread_pool = thread_pool.get();
    std::cerr << "[THREAD_POOL] num_threads=" << num_threads << std::endl;

    allocate_tensor(embed_tokens_w, {config.vocab_size, config.hidden_dim}, DataType::FP16);

    for (auto& layer : layers) {
        allocate_tensor(layer.norm1_w, {config.hidden_dim}, DataType::FP16);
        allocate_tensor(layer.norm2_w, {config.hidden_dim}, DataType::FP16);
        allocate_tensor(layer.b_q, {config.num_q_heads * config.head_dim}, DataType::FP16);
        allocate_tensor(layer.b_k, {config.num_kv_heads * config.head_dim}, DataType::FP16);
        allocate_tensor(layer.b_v, {config.num_kv_heads * config.head_dim}, DataType::FP16);

        allocate_gptq_weight(layer.q_proj, config.hidden_dim, config.num_q_heads * config.head_dim);
        allocate_gptq_weight(layer.k_proj, config.hidden_dim, config.num_kv_heads * config.head_dim);
        allocate_gptq_weight(layer.v_proj, config.hidden_dim, config.num_kv_heads * config.head_dim);
        allocate_gptq_weight(layer.o_proj, config.num_q_heads * config.head_dim, config.hidden_dim);
        allocate_gptq_weight(layer.gate_proj, config.hidden_dim, config.intermediate_size);
        allocate_gptq_weight(layer.up_proj, config.hidden_dim, config.intermediate_size);
        allocate_gptq_weight(layer.down_proj, config.intermediate_size, config.hidden_dim);
    }

    allocate_tensor(final_norm_w, {config.hidden_dim}, DataType::FP16);
    allocate_gptq_weight(lm_head, config.hidden_dim, config.vocab_size);
    allocate_tensor(cos_cache, {config.max_seq_len, config.head_dim}, DataType::FP16);
    allocate_tensor(sin_cache, {config.max_seq_len, config.head_dim}, DataType::FP16);

    init_rope_cache();
}

QwenModel::~QwenModel() {
    if (block_workspace) {
        g_memory_pool->free_block(block_workspace);
        block_workspace = nullptr;
        block_workspace_bytes = 0;
    }
    for (void* p : weight_ptrs) {
        if (p) g_memory_pool->free_block(p);
    }
    weight_ptrs.clear();
    if (g_thread_pool == thread_pool.get()) {
        g_thread_pool = nullptr;
    }
}

void QwenModel::allocate_tensor(Tensor& t, const std::vector<int>& shape, DataType dtype) {
    t.shape = shape;
    t.dtype = dtype;
    t.device = DeviceType::CPU;
    t.owns_data = false;

    size_t elements = 1;
    for (int d : shape) elements *= static_cast<size_t>(d);
    size_t bytes = elements * dtype_size(dtype);
    t.data = g_memory_pool->allocate(bytes);
    if (!t.data) {
        std::ostringstream oss;
        oss << "MemoryPool allocation failed: bytes=" << bytes << " shape=[";
        for (size_t i = 0; i < shape.size(); ++i) {
            if (i) oss << ",";
            oss << shape[i];
        }
        oss << "] dtype=" << static_cast<int>(dtype);
        throw std::runtime_error(oss.str());
    }
    std::memset(t.data, 0, bytes);
    weight_ptrs.push_back(t.data);

    t.stride.resize(shape.size());
    int acc = 1;
    for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
        t.stride[i] = acc;
        acc *= shape[i];
    }
}

void QwenModel::allocate_gptq_weight(arm_neon::GPTQInt8Weight& w, int K, int N, int group_size) {
    w.K = K;
    w.N = N;
    w.group_size = group_size;
    w.num_groups = (K + group_size - 1) / group_size;
    w.has_zero = true;
    w.has_g_idx = false;
    w.pack_kind = arm_neon::GPTQPackKind::W8A16_FP16_PANEL;

    int np = (N + arm_neon::NR_F16 - 1) / arm_neon::NR_F16;
    int K_pad = ((K + 7) / 8) * 8;
    allocate_tensor(w.qweight_pack, {np, K_pad, arm_neon::NR_F16}, DataType::INT8);
    allocate_tensor(w.scales_pack, {np, w.num_groups, arm_neon::NR_F16}, DataType::FP16);
    allocate_tensor(w.zeros_pack, {np, w.num_groups, arm_neon::NR_F16}, DataType::INT8);
}

bool QwenModel::load_tensor_from_bin(const std::string& filepath, Tensor& tensor) {
    return read_exact_file(filepath, tensor.data, tensor.bytes());
}

bool QwenModel::load_gptq_weight_from_bins(const std::string& prefix, arm_neon::GPTQInt8Weight& w) {
    if (!load_tensor_from_bin(prefix + ".qweight.s8pack.bin", w.qweight_pack)) return false;
    if (!load_tensor_from_bin(prefix + ".scales.f16pack.bin", w.scales_pack)) return false;
    if (!load_tensor_from_bin(prefix + ".qzeros.s8pack.bin", w.zeros_pack)) return false;

    std::string gidx_path = prefix + ".g_idx.i32.bin";
    if (file_exists(gidx_path)) {
        allocate_tensor(w.g_idx, {w.K}, DataType::INT32);
        if (!load_tensor_from_bin(gidx_path, w.g_idx)) return false;
        w.has_g_idx = true;
    }
    return true;
}

bool QwenModel::load_weights(const std::string& weights_dir) {
    if (!load_tensor_from_bin(join_path(weights_dir, "embed_tokens_w.f16.bin"), embed_tokens_w)) return false;
    if (!load_tensor_from_bin(join_path(weights_dir, "final_norm_w.f16.bin"), final_norm_w)) return false;

    for (int i = 0; i < config.num_layers; ++i) {
        auto& layer = layers[i];
        std::string p = "layer" + std::to_string(i) + "_";
        if (!load_tensor_from_bin(join_path(weights_dir, p + "norm1_w.f16.bin"), layer.norm1_w)) return false;
        if (!load_tensor_from_bin(join_path(weights_dir, p + "norm2_w.f16.bin"), layer.norm2_w)) return false;
        if (!load_tensor_from_bin(join_path(weights_dir, p + "b_q.f16.bin"), layer.b_q)) return false;
        if (!load_tensor_from_bin(join_path(weights_dir, p + "b_k.f16.bin"), layer.b_k)) return false;
        if (!load_tensor_from_bin(join_path(weights_dir, p + "b_v.f16.bin"), layer.b_v)) return false;

        if (!load_gptq_weight_from_bins(join_path(weights_dir, p + "q_proj"), layer.q_proj)) return false;
        if (!load_gptq_weight_from_bins(join_path(weights_dir, p + "k_proj"), layer.k_proj)) return false;
        if (!load_gptq_weight_from_bins(join_path(weights_dir, p + "v_proj"), layer.v_proj)) return false;
        if (!load_gptq_weight_from_bins(join_path(weights_dir, p + "o_proj"), layer.o_proj)) return false;
        if (!load_gptq_weight_from_bins(join_path(weights_dir, p + "gate_proj"), layer.gate_proj)) return false;
        if (!load_gptq_weight_from_bins(join_path(weights_dir, p + "up_proj"), layer.up_proj)) return false;
        if (!load_gptq_weight_from_bins(join_path(weights_dir, p + "down_proj"), layer.down_proj)) return false;
    }

    return load_gptq_weight_from_bins(join_path(weights_dir, "lm_head"), lm_head);
}

void QwenModel::init_rope_cache() {
    fp16_t* cos_ptr = cos_cache.ptr<fp16_t>();
    fp16_t* sin_ptr = sin_cache.ptr<fp16_t>();
    int half = config.head_dim / 2;

    for (int pos = 0; pos < config.max_seq_len; ++pos) {
        for (int i = 0; i < half; ++i) {
            float inv_freq = std::pow(config.rope_theta, -static_cast<float>(i) / half);
            float freq = static_cast<float>(pos) * inv_freq;
            fp16_t c = (fp16_t)std::cos(freq);
            fp16_t s = (fp16_t)std::sin(freq);
            cos_ptr[(size_t)pos * config.head_dim + i] = c;
            sin_ptr[(size_t)pos * config.head_dim + i] = s;
            cos_ptr[(size_t)pos * config.head_dim + i + half] = c;
            sin_ptr[(size_t)pos * config.head_dim + i + half] = s;
        }
    }
}

void QwenModel::ensure_block_workspace() {
    size_t row_bytes = align_size((size_t)config.hidden_dim * sizeof(fp16_t));
    size_t required_block_workspace = 2 * row_bytes + 64ULL * 1024 * 1024 + 64;
    if (block_workspace_bytes >= required_block_workspace) {
        return;
    }

    if (block_workspace) {
        g_memory_pool->free_block(block_workspace);
        block_workspace = nullptr;
        block_workspace_bytes = 0;
    }

    block_workspace = g_memory_pool->allocate(required_block_workspace);
    if (!block_workspace) {
        throw std::runtime_error("MemoryPool allocation failed: block_workspace");
    }
    block_workspace_bytes = required_block_workspace;
    std::cerr << "[WORKSPACE] persistent_block_workspace_mb="
              << (block_workspace_bytes / 1024.0 / 1024.0) << std::endl;
}

void QwenModel::build_graph(KVCache& cache) {
    ext_hidden_states.assign(config.hidden_dim, (fp16_t)0);
    ext_norm_out.assign(config.hidden_dim, (fp16_t)0);

    t_hidden_states = graph.create_tensor_from_ptr(
        {1, config.hidden_dim}, ext_hidden_states.data(), DataType::FP16);

    t_cos = graph.create_tensor_from_ptr(
        {config.head_dim}, cos_cache.ptr<fp16_t>(), DataType::FP16);
    t_sin = graph.create_tensor_from_ptr(
        {config.head_dim}, sin_cache.ptr<fp16_t>(), DataType::FP16);

    arm_neon::AttentionConfig attn_config{
        config.hidden_dim, config.num_q_heads, config.num_kv_heads, config.head_dim};
    arm_neon::FFNConfig ffn_config{config.hidden_dim, config.intermediate_size};

    ensure_block_workspace();

    for (int i = 0; i < config.num_layers; ++i) {
        auto& layer = layers[i];
        QwenBlockNode* block = graph.add_qwen_block(
            t_hidden_states, &layer.norm1_w,
            &layer.b_q, &layer.b_k, &layer.b_v,
            t_cos, t_sin,
            &layer.norm2_w,
            &layer,
            &cache,
            i,
            &current_pos,
            attn_config,
            ffn_config,
            config.rms_norm_eps);
        block->set_external_workspace(block_workspace, block_workspace_bytes);
    }

    t_norm_out = graph.create_tensor_from_ptr(
        {1, config.hidden_dim}, ext_norm_out.data(), DataType::FP16);
    graph.add_rmsnorm(t_hidden_states, &final_norm_w, t_norm_out, config.rms_norm_eps);

    plan = compiler.compile(graph);
    is_graph_built = true;
}

int QwenModel::prefill_prompt_batch(
    const std::vector<int>& input_tokens,
    int start_pos,
    KVCache& cache
) {
    if (input_tokens.empty()) {
        return -1;
    }
    if (start_pos < 0 ||
        start_pos + static_cast<int>(input_tokens.size()) > config.max_seq_len) {
        return -1;
    }

    ensure_block_workspace();

    const int T = static_cast<int>(input_tokens.size());
    const int H = config.hidden_dim;
    std::vector<fp16_t> hidden((size_t)T * H);

    for (int t = 0; t < T; ++t) {
        int token_id = input_tokens[t];
        if (token_id < 0 || token_id >= config.vocab_size) {
            return -1;
        }
        const fp16_t* embed = embed_tokens_w.ptr<fp16_t>() + (size_t)token_id * H;
        std::memcpy(hidden.data() + (size_t)t * H, embed, (size_t)H * sizeof(fp16_t));
    }

    arm_neon::AttentionConfig attn_config{
        config.hidden_dim, config.num_q_heads, config.num_kv_heads, config.head_dim};
    arm_neon::FFNConfig ffn_config{config.hidden_dim, config.intermediate_size};
    Workspace block_ws(block_workspace, block_workspace_bytes);

    for (int layer_id = 0; layer_id < config.num_layers; ++layer_id) {
        auto& layer = layers[layer_id];
        Tensor batch_hidden({T, H}, hidden.data(), DataType::FP16);
        Status status = arm_neon::qwen_block_f16_gptq_prefill_neon(
            batch_hidden,
            layer.norm1_w,
            layer.q_proj,
            layer.k_proj,
            layer.v_proj,
            layer.o_proj,
            layer.b_q.ptr<fp16_t>(),
            layer.b_k.ptr<fp16_t>(),
            layer.b_v.ptr<fp16_t>(),
            cos_cache.ptr<fp16_t>(),
            sin_cache.ptr<fp16_t>(),
            layer.norm2_w,
            layer.gate_proj,
            layer.up_proj,
            layer.down_proj,
            cache,
            layer_id,
            start_pos,
            attn_config,
            ffn_config,
            config.rms_norm_eps,
            block_ws);
        if (status != Status::SUCCESS) {
            std::cerr << "[ERROR] batch prefill block failed"
                      << " layer=" << layer_id
                      << " start_pos=" << start_pos
                      << " tokens=" << T
                      << " status=" << StatusToString(status)
                      << std::endl;
            return -1;
        }
    }

    current_pos = start_pos + T - 1;
    const fp16_t* last_hidden = hidden.data() + (size_t)(T - 1) * H;
    std::vector<fp16_t> norm_out(H);
    arm_neon::rmsnorm_f16_neon(
        last_hidden,
        final_norm_w.ptr<fp16_t>(),
        norm_out.data(),
        H,
        config.rms_norm_eps);

    arm_neon::ArgmaxResult result = arm_neon::linear_gptq_int8_decode_argmax_neon(
        norm_out.data(),
        lm_head,
        nullptr,
        0);
    forward_debug_last_result.token_id = result.index;
    forward_debug_last_result.logit = result.value;
    if (result.index < 0 || result.index >= config.vocab_size) {
        std::cerr << "[ERROR] batch prefill lm_head argmax failed"
                  << " index=" << result.index
                  << " value=" << result.value
                  << std::endl;
        return -1;
    }
    return result.index;
}

int QwenModel::forward(int token_id, int pos, KVCache& cache) {
    if (!is_graph_built) {
        build_graph(cache);
    }

    current_pos = pos;
    fp16_t* embed = embed_tokens_w.ptr<fp16_t>() + (size_t)token_id * config.hidden_dim;
    std::memcpy(t_hidden_states->data, embed, (size_t)config.hidden_dim * sizeof(fp16_t));

    t_cos->data = cos_cache.ptr<fp16_t>() + (size_t)pos * config.head_dim;
    t_sin->data = sin_cache.ptr<fp16_t>() + (size_t)pos * config.head_dim;

    Status status = runtime.run(plan);
    if (status != Status::SUCCESS) {
        std::cerr << "[ERROR] graph runtime failed: " << StatusToString(status) << std::endl;
        return -1;
    }

    if (env_flag("LLM_DEBUG_NUMERIC")) {
        const fp16_t* norm = t_norm_out->ptr<fp16_t>();
        int finite_count = 0;
        int nan_count = 0;
        int inf_count = 0;
        float min_v = std::numeric_limits<float>::infinity();
        float max_v = -std::numeric_limits<float>::infinity();
        for (int i = 0; i < config.hidden_dim; ++i) {
            float v = (float)norm[i];
            if (std::isnan(v)) {
                nan_count++;
            } else if (!std::isfinite(v)) {
                inf_count++;
            } else {
                finite_count++;
                min_v = std::min(min_v, v);
                max_v = std::max(max_v, v);
            }
        }
        std::cerr << "[NUMERIC] final_norm_out"
                  << " finite=" << finite_count
                  << " nan=" << nan_count
                  << " inf=" << inf_count
                  << " min=" << min_v
                  << " max=" << max_v
                  << std::endl;
    }

    arm_neon::ArgmaxResult result = arm_neon::linear_gptq_int8_decode_argmax_neon(
        t_norm_out->ptr<fp16_t>(),
        lm_head,
        nullptr,
        0);
    forward_debug_last_result.token_id = result.index;
    forward_debug_last_result.logit = result.value;
    if (result.index < 0 || result.index >= config.vocab_size) {
        std::cerr << "[ERROR] lm_head argmax failed"
                  << " index=" << result.index
                  << " value=" << result.value
                  << " K=" << lm_head.K
                  << " N=" << lm_head.N
                  << " qweight=" << lm_head.qweight_pack.data
                  << " scales=" << lm_head.scales_pack.data
                  << " zeros=" << lm_head.zeros_pack.data
                  << " norm_out=" << t_norm_out->data
                  << std::endl;
        return -1;
    }
    return result.index;
}

void QwenModel::generate(
    const std::vector<int>& input_tokens,
    int max_new_tokens,
    std::function<bool(int)> callback
) {
    if (!kv_cache) return;

    int next_token = -1;
    if (!input_tokens.empty()) {
        if (env_flag("LLM_DISABLE_BATCH_PREFILL")) {
            for (int tok : input_tokens) {
                if (history_pos >= config.max_seq_len) return;
                next_token = forward(tok, history_pos, *kv_cache);
                history_pos++;
                if (next_token < 0) {
                    std::cerr << "[ERROR] prompt forward failed"
                              << " token=" << tok
                              << " pos=" << (history_pos - 1)
                              << std::endl;
                    return;
                }
            }
        } else {
            if (history_pos + static_cast<int>(input_tokens.size()) > config.max_seq_len) {
                return;
            }
            next_token = prefill_prompt_batch(input_tokens, history_pos, *kv_cache);
            if (next_token < 0) {
                std::cerr << "[ERROR] batch prefill failed"
                          << " start_pos=" << history_pos
                          << " tokens=" << input_tokens.size()
                          << std::endl;
                return;
            }
            history_pos += static_cast<int>(input_tokens.size());
        }
    }

    int current_token = next_token;
    for (int i = 0; i < max_new_tokens && current_token >= 0; ++i) {
        if (is_stop_token(current_token)) {
            if (history_pos < config.max_seq_len) {
                forward(current_token, history_pos, *kv_cache);
                history_pos++;
            }
            break;
        }

        if (!callback(current_token)) break;
        if (history_pos >= config.max_seq_len) break;
        current_token = forward(current_token, history_pos, *kv_cache);
        history_pos++;
    }
}

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
    return end == v ? default_value : static_cast<int>(x);
}

float QwenModel::env_float(const char* name, float default_value) {
    const char* v = std::getenv(name);
    if (!v) return default_value;
    char* end = nullptr;
    float x = std::strtof(v, &end);
    return end == v ? default_value : x;
}

} // namespace llm_engine
