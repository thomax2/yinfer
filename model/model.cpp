#include "model.h"
#include "qwen_execution_plan.h"
#include "backends/cpu/arm_neon/kernel_common.h"
#include "llm_engine/metrics/metrics.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <sstream>
#include <unordered_set>

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

uint64_t model_now_us() {
    using clock = std::chrono::steady_clock;
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::microseconds>(
            clock::now().time_since_epoch()).count());
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
    if (mixed_batch_workspace) {
        g_memory_pool->free_block(mixed_batch_workspace);
        mixed_batch_workspace = nullptr;
        mixed_batch_workspace_bytes = 0;
        mixed_batch_workspace_max_rows = 0;
        mixed_batch_row_capacity = 0;
    }
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
        std::vector<int32_t> g_idx((size_t)w.K);
        if (!read_exact_file(gidx_path, g_idx.data(), g_idx.size() * sizeof(int32_t))) {
            return false;
        }
        bool canonical = true;
        for (int k = 0; k < w.K; ++k) {
            if (g_idx[(size_t)k] != k / w.group_size) {
                canonical = false;
                break;
            }
        }
        if (canonical) {
            // A canonical g_idx carries no permutation information. Treating it
            // as the implicit group mapping preserves the exact quantization
            // semantics and keeps the true batch kernel on its optimized path.
            w.has_g_idx = false;
        } else {
            allocate_tensor(w.g_idx, {w.K}, DataType::INT32);
            std::memcpy(w.g_idx.data, g_idx.data(), g_idx.size() * sizeof(int32_t));
            w.has_g_idx = true;
        }
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

bool QwenModel::ensure_mixed_batch_workspace(int max_rows) {
    // 256 是保护性上限，不代表 RK3588 的推荐值；实际默认 token budget 为 64。
    if (max_rows <= 0 || max_rows > 256) return false;
    if (mixed_batch_workspace && mixed_batch_workspace_max_rows >= max_rows) {
        return true;
    }

    const int H = config.hidden_dim;
    const int q_size = config.num_q_heads * config.head_dim;
    const int kv_size = config.num_kv_heads * config.head_dim;
    const int packed_a_max_k = std::max({H, q_size, config.intermediate_size});
    const int num_rep = config.num_q_heads / config.num_kv_heads;
    size_t required = 64;
    auto add_fp16 = [&](size_t elements) {
        required += align_size(elements * sizeof(fp16_t));
    };
    auto add_f32 = [&](size_t elements) {
        required += align_size(elements * sizeof(float));
    };
    add_fp16((size_t)max_rows * H);       // hidden
    add_fp16((size_t)max_rows * H);       // residual
    add_fp16((size_t)max_rows * H);       // norm/final norm
    // Mixed 主体中所有普通 Linear 共享这一块 Packed A：
    // norm1->Q/K/V、attention output->O、norm2->Gate/Up、SwiGLU output->Down。
    add_fp16((size_t)arm_neon::align_up_int(max_rows, 8) * packed_a_max_k);
    add_fp16((size_t)max_rows * q_size);  // q
    add_fp16((size_t)max_rows * kv_size); // k
    add_fp16((size_t)max_rows * kv_size); // v
    add_fp16((size_t)max_rows * q_size);  // attention output
    add_fp16((size_t)max_rows * H);       // projection output
    add_fp16((size_t)max_rows * config.intermediate_size); // gate/activation
    add_fp16((size_t)max_rows * config.intermediate_size); // up
    // Decode 每次只有一个 Query，只需 [num_rep, max_seq_len] Score。
    // Online Paged Prefill 不再占用 [max_rows, num_rep, max_seq_len] 完整矩阵。
    add_fp16((size_t)num_rep * config.max_seq_len);
    // 4x16 QK^T 内核把一个 16-token K 子块转置为 [head_dim, 16]。
    add_fp16((size_t)config.head_dim * 16);
    // Online Softmax 的 O 累加器。一次只处理 4 个 Query 和一个 Q Head，
    // 所以不同 Query Tile、KV Head、模型层都可以复用同一块固定内存。
    add_f32((size_t)4 * config.head_dim);
    // total_rows 可以达到 Scheduler token budget，但真正进入 LM Head 的
    // logit_rows 受 Batch Argmax 内核上限约束。固定 Workspace 应按该独立上限
    // 预留，不能把 max_rows=64 直接传入 helper（超上限时 helper 会返回 0）。
    const int argmax_rows = std::min(
        max_rows, arm_neon::GPTQ_BATCH_ARGMAX_MAX_ROWS);
    required += align_size(arm_neon::linear_gptq_int8_batch_argmax_workspace_bytes(
        argmax_rows, lm_head));

    void* replacement = g_memory_pool->allocate(required);
    if (!replacement) return false;
    if (mixed_batch_workspace) {
        g_memory_pool->free_block(mixed_batch_workspace);
    }
    mixed_batch_workspace = replacement;
    mixed_batch_workspace_bytes = required;
    mixed_batch_workspace_max_rows = max_rows;
    mixed_batch_workspace_reallocations++;
    std::cerr << "[MIXED_BATCH] workspace_mb="
              << (required / 1024.0 / 1024.0)
              << " workspace_max_rows=" << max_rows << std::endl;
    return true;
}

QwenModel::MixedBatchWorkspaceView
QwenModel::mixed_batch_workspace_view(int total_rows) {
    MixedBatchWorkspaceView view;
    if (!mixed_batch_workspace || total_rows <= 0 ||
        total_rows > mixed_batch_workspace_max_rows) {
        return view;
    }
    const int H = config.hidden_dim;
    const int q_size = config.num_q_heads * config.head_dim;
    const int kv_size = config.num_kv_heads * config.head_dim;
    const int packed_a_max_k = std::max({H, q_size, config.intermediate_size});
    const int num_rep = config.num_q_heads / config.num_kv_heads;
    char* cursor = static_cast<char*>(mixed_batch_workspace);
    auto take_fp16 = [&](size_t elements) {
        fp16_t* result = reinterpret_cast<fp16_t*>(cursor);
        cursor += align_size(elements * sizeof(fp16_t));
        return result;
    };
    auto take_f32 = [&](size_t elements) {
        float* result = reinterpret_cast<float*>(cursor);
        cursor += align_size(elements * sizeof(float));
        return result;
    };
    view.hidden = take_fp16((size_t)mixed_batch_workspace_max_rows * H);
    view.residual = take_fp16((size_t)mixed_batch_workspace_max_rows * H);
    view.norm = take_fp16((size_t)mixed_batch_workspace_max_rows * H);
    view.packed_a = take_fp16(
        (size_t)arm_neon::align_up_int(mixed_batch_workspace_max_rows, 8) *
        packed_a_max_k);
    view.q = take_fp16((size_t)mixed_batch_workspace_max_rows * q_size);
    view.k = take_fp16((size_t)mixed_batch_workspace_max_rows * kv_size);
    view.v = take_fp16((size_t)mixed_batch_workspace_max_rows * kv_size);
    view.attn_out = take_fp16((size_t)mixed_batch_workspace_max_rows * q_size);
    view.proj_out = take_fp16((size_t)mixed_batch_workspace_max_rows * H);
    view.gate = take_fp16(
        (size_t)mixed_batch_workspace_max_rows * config.intermediate_size);
    view.up = take_fp16(
        (size_t)mixed_batch_workspace_max_rows * config.intermediate_size);
    view.score = take_fp16((size_t)num_rep * config.max_seq_len);
    view.attention_k_tile = take_fp16((size_t)config.head_dim * 16);
    view.attention_online_out = take_f32((size_t)4 * config.head_dim);
    view.argmax_partials = cursor;
    view.argmax_partials_bytes =
        mixed_batch_workspace_bytes - (size_t)(cursor - static_cast<char*>(mixed_batch_workspace));
    view.bytes = mixed_batch_workspace_bytes;
    return view;
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

    // Both the fine ExecutablePlan and the legacy GraphRuntime bind their
    // final RMSNorm result to the same stable model-owned boundary buffer.
    t_norm_out = graph.create_tensor_from_ptr(
        {1, config.hidden_dim}, ext_norm_out.data(), DataType::FP16);

    arm_neon::AttentionConfig attn_config{
        config.hidden_dim, config.num_q_heads, config.num_kv_heads, config.head_dim};
    arm_neon::FFNConfig ffn_config{config.hidden_dim, config.intermediate_size};

    use_fine_decode_plan = !env_flag("LLM_USE_COARSE_QWEN_BLOCK");
    if (use_fine_decode_plan) {
        QwenPlanArtifacts artifacts = build_qwen_single_decode_plan(
            layers, final_norm_w, lm_head,
            attn_config, ffn_config,
            config.max_seq_len, config.rms_norm_eps);
        fine_hidden_input = artifacts.hidden_input;
        fine_norm_output = artifacts.norm_output;
        fine_argmax_output = artifacts.argmax_output;
        coarse_decode_workspace_bytes = artifacts.coarse_workspace_bytes;
        fine_decode_plan = std::move(artifacts.plan);

        std::cerr << "[GRAPH] mode=fine_decode"
                  << " nodes=" << fine_decode_plan->memory_stats().node_count
                  << " arena_mb="
                  << fine_decode_plan->memory_stats().arena_bytes / 1024.0 / 1024.0
                  << " coarse_workspace_mb="
                  << coarse_decode_workspace_bytes / 1024.0 / 1024.0
                  << std::endl;
        is_graph_built = true;
        return;
    }

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

    int capacity = 1;
    while (capacity < T && capacity < config.max_seq_len) capacity *= 2;
    capacity = std::min(capacity, config.max_seq_len);

    auto plan_it = fine_prefill_plans.find(capacity);
    if (plan_it == fine_prefill_plans.end()) {
        arm_neon::AttentionConfig attn_config{
            config.hidden_dim, config.num_q_heads, config.num_kv_heads, config.head_dim};
        arm_neon::FFNConfig ffn_config{config.hidden_dim, config.intermediate_size};
        QwenPlanArtifacts artifacts = build_qwen_prefill_plan(
            layers, final_norm_w, lm_head,
            attn_config, ffn_config,
            config.max_seq_len, capacity, config.rms_norm_eps);
        BoundExecutablePlan stored;
        stored.hidden_input = artifacts.hidden_input;
        stored.norm_output = artifacts.norm_output;
        stored.argmax_output = artifacts.argmax_output;
        stored.plan = std::move(artifacts.plan);
        plan_it = fine_prefill_plans.emplace(capacity, std::move(stored)).first;
        std::cerr << "[GRAPH] mode=prefill"
                  << " capacity=" << capacity
                  << " nodes=" << plan_it->second.plan->memory_stats().node_count
                  << " arena_mb="
                  << plan_it->second.plan->memory_stats().arena_bytes / 1024.0 / 1024.0
                  << std::endl;
    }

    BoundExecutablePlan& stored = plan_it->second;
    last_batch_norm_out_.assign(H, (fp16_t)0);
    arm_neon::ArgmaxResult result{-1, 0.0f};
    QwenPlanRuntime plan_runtime;
    plan_runtime.kv_cache = &cache;
    plan_runtime.current_pos = start_pos + T - 1;
    plan_runtime.start_pos = start_pos;
    plan_runtime.cos = cos_cache.ptr<fp16_t>();
    plan_runtime.sin = sin_cache.ptr<fp16_t>();

    ExecutionContext context;
    context.actual_rows = T;
    context.current_pos = plan_runtime.current_pos;
    context.start_pos = start_pos;
    context.user_data = &plan_runtime;
    context.bind_external(stored.hidden_input, hidden.data());
    context.bind_external(stored.norm_output, last_batch_norm_out_.data());
    context.bind_external(stored.argmax_output, &result);
    Status status = stored.plan->run(context, fine_prefill_workspace);
    if (status != Status::SUCCESS) {
        std::cerr << "[ERROR] fine prefill plan failed"
                  << " start_pos=" << start_pos
                  << " tokens=" << T
                  << " capacity=" << capacity
                  << " status=" << StatusToString(status)
                  << std::endl;
        return -1;
    }

    current_pos = start_pos + T - 1;
    last_batch_norm_out_valid_ = true;
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
    last_batch_norm_out_valid_ = false;
    if (!is_graph_built) {
        build_graph(cache);
    }

    current_pos = pos;
    fp16_t* embed = embed_tokens_w.ptr<fp16_t>() + (size_t)token_id * config.hidden_dim;
    std::memcpy(t_hidden_states->data, embed, (size_t)config.hidden_dim * sizeof(fp16_t));

    t_cos->data = cos_cache.ptr<fp16_t>() + (size_t)pos * config.head_dim;
    t_sin->data = sin_cache.ptr<fp16_t>() + (size_t)pos * config.head_dim;

    arm_neon::ArgmaxResult fine_result{-1, 0.0f};
    Status status = Status::SUCCESS;
    if (use_fine_decode_plan) {
        if (!fine_decode_plan) return -1;
        QwenPlanRuntime plan_runtime;
        plan_runtime.kv_cache = &cache;
        plan_runtime.current_pos = pos;
        plan_runtime.start_pos = pos;
        plan_runtime.cos = t_cos->ptr<fp16_t>();
        plan_runtime.sin = t_sin->ptr<fp16_t>();

        ExecutionContext context;
        context.actual_rows = 1;
        context.current_pos = pos;
        context.start_pos = pos;
        context.user_data = &plan_runtime;
        context.bind_external(fine_hidden_input, t_hidden_states->data);
        context.bind_external(fine_norm_output, t_norm_out->data);
        context.bind_external(fine_argmax_output, &fine_result);
        status = fine_decode_plan->run(context, fine_execution_workspace);
    } else {
        status = runtime.run(plan);
    }
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

    arm_neon::ArgmaxResult result = fine_result;
    if (!use_fine_decode_plan) {
        result = arm_neon::linear_gptq_int8_decode_argmax_neon(
            t_norm_out->ptr<fp16_t>(),
            lm_head,
            nullptr,
            0);
    }
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

int QwenModel::sample_next_token_from_last_logits(
    const SamplingParams& sampling,
    std::mt19937_64& rng,
    SamplingRuntimeStats* stats
) {
    const fp16_t* norm = nullptr;
    if (last_batch_norm_out_valid_ &&
        static_cast<int>(last_batch_norm_out_.size()) == config.hidden_dim) {
        norm = last_batch_norm_out_.data();
    } else if (t_norm_out) {
        norm = t_norm_out->ptr<fp16_t>();
    }
    if (!norm) {
        return -1;
    }

    auto greedy_argmax = [&]() -> int {
        arm_neon::ArgmaxResult result = arm_neon::linear_gptq_int8_decode_argmax_neon(
            norm,
            lm_head,
            nullptr,
            0);
        if (stats) {
            stats->greedy_tokens++;
        }
        forward_debug_last_result.token_id = result.index;
        forward_debug_last_result.logit = result.value;
        return result.index;
    };

    float temperature = sampling.temperature;
    if (temperature < 0.0f) {
        temperature = 0.0f;
    }
    if (sampling.greedy || temperature <= 0.0f) {
        return greedy_argmax();
    }

    uint64_t begin_us = model_now_us();
    std::vector<fp16_t> logits((size_t)config.vocab_size);
    Status status = arm_neon::linear_gptq_int8_decode_neon(
        norm,
        lm_head,
        logits.data(),
        nullptr,
        nullptr,
        0);
    if (status != Status::SUCCESS) {
        std::cerr << "[SAMPLING] logits failed, fallback greedy status="
                  << StatusToString(status)
                  << std::endl;
        if (stats) {
            stats->sampling_ms += (double)(model_now_us() - begin_us) / 1000.0;
        }
        return greedy_argmax();
    }

    struct Candidate {
        int id;
        float logit;
        double weight;
    };
    std::vector<Candidate> candidates;
    candidates.reserve((size_t)config.vocab_size);
    int greedy_id = -1;
    float greedy_logit = -std::numeric_limits<float>::infinity();
    for (int id = 0; id < config.vocab_size; ++id) {
        float logit = static_cast<float>(logits[(size_t)id]);
        if (!std::isfinite(logit)) {
            continue;
        }
        if (logit > greedy_logit) {
            greedy_logit = logit;
            greedy_id = id;
        }
        candidates.push_back(Candidate{id, logit, 0.0});
    }
    if (candidates.empty()) {
        if (stats) {
            stats->sampling_ms += (double)(model_now_us() - begin_us) / 1000.0;
        }
        return greedy_argmax();
    }

    int top_k = sampling.top_k < 0 ? 0 : sampling.top_k;
    if (top_k > 0 && top_k < static_cast<int>(candidates.size())) {
        std::nth_element(
            candidates.begin(),
            candidates.begin() + top_k,
            candidates.end(),
            [](const Candidate& a, const Candidate& b) {
                return a.logit > b.logit;
            });
        candidates.resize((size_t)top_k);
    }

    float top_p = sampling.top_p;
    if (top_p <= 0.0f) {
        top_p = 1.0f;
    }
    if (top_p > 1.0f) {
        top_p = 1.0f;
    }
    if (top_p < 1.0f) {
        std::sort(
            candidates.begin(),
            candidates.end(),
            [](const Candidate& a, const Candidate& b) {
                return a.logit > b.logit;
            });
    }

    float max_logit = -std::numeric_limits<float>::infinity();
    for (const Candidate& c : candidates) {
        max_logit = std::max(max_logit, c.logit);
    }

    double total_weight = 0.0;
    const double inv_temp = 1.0 / std::max<double>(temperature, 1e-6);
    for (Candidate& c : candidates) {
        c.weight = std::exp(((double)c.logit - (double)max_logit) * inv_temp);
        if (std::isfinite(c.weight) && c.weight > 0.0) {
            total_weight += c.weight;
        } else {
            c.weight = 0.0;
        }
    }
    if (total_weight <= 0.0) {
        if (stats) {
            stats->sampling_ms += (double)(model_now_us() - begin_us) / 1000.0;
        }
        return greedy_argmax();
    }

    if (top_p < 1.0f) {
        double cumulative = 0.0;
        size_t keep = 0;
        for (; keep < candidates.size(); ++keep) {
            cumulative += candidates[keep].weight / total_weight;
            if (cumulative >= top_p) {
                ++keep;
                break;
            }
        }
        keep = std::max<size_t>(1, std::min(keep, candidates.size()));
        candidates.resize(keep);
        total_weight = 0.0;
        for (const Candidate& c : candidates) {
            total_weight += c.weight;
        }
    }

    std::uniform_real_distribution<double> dist(0.0, total_weight);
    double pick = dist(rng);
    double cumulative = 0.0;
    int sampled_id = -1;
    float sampled_logit = 0.0f;
    for (const Candidate& c : candidates) {
        cumulative += c.weight;
        if (pick <= cumulative) {
            sampled_id = c.id;
            sampled_logit = c.logit;
            break;
        }
    }
    if (sampled_id < 0 || sampled_id >= config.vocab_size) {
        sampled_id = greedy_id;
        sampled_logit = greedy_logit;
    }

    if (stats) {
        stats->sampled_tokens++;
        stats->sampling_ms += (double)(model_now_us() - begin_us) / 1000.0;
    }
    forward_debug_last_result.token_id = sampled_id;
    forward_debug_last_result.logit = sampled_logit;
    return sampled_id;
}

void QwenModel::generate(
    const std::vector<int>& input_tokens,
    int max_new_tokens,
    std::function<bool(int)> callback
) {
    if (!kv_cache) return;

    int next_token = -1;
    if (!input_tokens.empty()) {
        bool force_sequential_prefill = env_flag("LLM_DISABLE_BATCH_PREFILL");

        if (force_sequential_prefill) {
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

int QwenModel::prefill_one_for_sequence(
    SequenceState& seq,
    const std::vector<int>& prompt_tokens,
    int prompt_index,
    KVCacheManager& kv_manager,
    PrefixCache* prefix_cache
) {
    if (!kv_cache) return -1;
    if (prompt_index < 0 || prompt_index >= static_cast<int>(prompt_tokens.size())) {
        seq.status = SequenceStatus::FAILED;
        seq.error_message = "prompt cursor out of range";
        return -1;
    }
    if (seq.history_pos >= config.max_seq_len) {
        seq.status = SequenceStatus::FAILED;
        seq.error_message = "sequence length exceeded";
        return -1;
    }
    if (!kv_manager.ensure_block_for_position(seq, seq.history_pos)) {
        seq.status = SequenceStatus::FAILED;
        if (seq.error_message.empty()) {
            seq.error_message = "KV block pool exhausted";
        }
        return -1;
    }

    int token_id = prompt_tokens[(size_t)prompt_index];
    kv_cache->set_active_sequence(&seq.block_table, &seq.max_written_pos);
    int next_token = forward(token_id, seq.history_pos, *kv_cache);
    kv_cache->clear_active_sequence();
    if (next_token < 0) {
        seq.status = SequenceStatus::FAILED;
        seq.error_message = "prompt forward failed";
        return -1;
    }

    seq.all_tokens.push_back(token_id);
    seq.history_pos++;
    seq.num_computed_tokens++;

    if (prefix_cache && kv_cache->block_size() > 0 &&
        seq.history_pos % kv_cache->block_size() == 0) {
        int block_size = kv_cache->block_size();
        int logical_block = (seq.history_pos / block_size) - 1;
        int block_begin = logical_block * block_size;
        if (logical_block >= 0 &&
            block_begin >= 0 &&
            block_begin + block_size <= static_cast<int>(seq.all_tokens.size()) &&
            logical_block < static_cast<int>(seq.block_table.size())) {
            int physical_block = seq.block_table[(size_t)logical_block];
            if (physical_block >= 0 && !kv_manager.block_has_hash(physical_block)) {
                HashValue parent_hash;
                if (logical_block > 0) {
                    int prev_block = seq.block_table[(size_t)(logical_block - 1)];
                    if (prev_block < 0 || !kv_manager.block_has_hash(prev_block)) {
                        return next_token;
                    }
                    parent_hash = kv_manager.block_hash(prev_block);
                }

                HashValue current_hash = hash_token_block(
                    parent_hash,
                    seq.all_tokens,
                    block_begin,
                    block_size,
                    prefix_cache->config());
                if (prefix_cache->insert(
                        current_hash,
                        parent_hash,
                        seq.all_tokens,
                        block_begin,
                        block_size,
                        physical_block)) {
                    kv_manager.attach_hash_to_block(
                        physical_block,
                        current_hash,
                        parent_hash,
                        block_size);
                    seq.last_prefix_hash = current_hash;
                }
            }
        }
    }

    return next_token;
}

int QwenModel::prefill_chunk_for_sequence(
    SequenceState& seq,
    const std::vector<int>& prompt_tokens,
    int prompt_begin,
    int prompt_end,
    KVCacheManager& kv_manager,
    PrefixCache* prefix_cache
) {
    if (prompt_begin < 0 ||
        prompt_end < prompt_begin ||
        prompt_end > static_cast<int>(prompt_tokens.size())) {
        seq.status = SequenceStatus::FAILED;
        seq.error_message = "prefill chunk range out of bounds";
        return -1;
    }
    if (prompt_begin == prompt_end) {
        return -1;
    }

    last_prefill_chunk_stats = PrefillChunkStats{};
    const int chunk_len = prompt_end - prompt_begin;

    auto register_completed_block = [&](int logical_block) {
        if (!prefix_cache || !kv_cache || logical_block < 0) {
            return;
        }
        int block_size = kv_cache->block_size();
        int block_begin = logical_block * block_size;
        if (block_size <= 0 ||
            block_begin < 0 ||
            block_begin + block_size > static_cast<int>(seq.all_tokens.size()) ||
            logical_block >= static_cast<int>(seq.block_table.size())) {
            return;
        }

        int physical_block = seq.block_table[(size_t)logical_block];
        if (physical_block < 0 || kv_manager.block_has_hash(physical_block)) {
            return;
        }

        HashValue parent_hash;
        if (logical_block > 0) {
            int prev_block = seq.block_table[(size_t)(logical_block - 1)];
            if (prev_block < 0 || !kv_manager.block_has_hash(prev_block)) {
                return;
            }
            parent_hash = kv_manager.block_hash(prev_block);
        }

        HashValue current_hash = hash_token_block(
            parent_hash,
            seq.all_tokens,
            block_begin,
            block_size,
            prefix_cache->config());
        if (prefix_cache->insert(
                current_hash,
                parent_hash,
                seq.all_tokens,
                block_begin,
                block_size,
                physical_block)) {
            kv_manager.attach_hash_to_block(
                physical_block,
                current_hash,
                parent_hash,
                block_size);
            seq.last_prefix_hash = current_hash;
        }
    };

    auto token_loop = [&]() -> int {
        uint64_t begin_us = model_now_us();
        last_prefill_chunk_stats.token_loop_used = true;
        int next_token = -1;
        for (int index = prompt_begin; index < prompt_end; ++index) {
            next_token = prefill_one_for_sequence(
                seq,
                prompt_tokens,
                index,
                kv_manager,
                prefix_cache);
            if (seq.status == SequenceStatus::FAILED || next_token < 0) {
                last_prefill_chunk_stats.token_loop_ms +=
                    (double)(model_now_us() - begin_us) / 1000.0;
                return -1;
            }
        }
        last_prefill_chunk_stats.token_loop_ms +=
            (double)(model_now_us() - begin_us) / 1000.0;
        return next_token;
    };

    if (chunk_len == 1 || env_flag("LLM_DISABLE_BATCH_PREFILL")) {
        return token_loop();
    }
    if (!kv_cache) {
        seq.status = SequenceStatus::FAILED;
        seq.error_message = "missing kv cache";
        return -1;
    }

    const int start_pos = seq.history_pos;
    if (start_pos < 0 || start_pos + chunk_len > config.max_seq_len) {
        seq.status = SequenceStatus::FAILED;
        seq.error_message = "sequence length exceeded";
        return -1;
    }
    if (!kv_manager.ensure_blocks_for_range(seq, start_pos, start_pos + chunk_len)) {
        seq.status = SequenceStatus::FAILED;
        if (seq.error_message.empty()) {
            seq.error_message = "KV block pool exhausted";
        }
        return -1;
    }

    std::vector<int> chunk_tokens(
        prompt_tokens.begin() + prompt_begin,
        prompt_tokens.begin() + prompt_end);

    uint64_t batch_begin_us = model_now_us();
    kv_cache->set_active_sequence(&seq.block_table, &seq.max_written_pos);
    int next_token = prefill_prompt_batch(chunk_tokens, start_pos, *kv_cache);
    kv_cache->clear_active_sequence();
    last_prefill_chunk_stats.batch_ms +=
        (double)(model_now_us() - batch_begin_us) / 1000.0;

    if (next_token >= 0) {
        last_prefill_chunk_stats.real_batch_used = true;
        seq.all_tokens.insert(seq.all_tokens.end(), chunk_tokens.begin(), chunk_tokens.end());
        seq.history_pos += chunk_len;
        seq.max_written_pos = std::max(seq.max_written_pos, seq.history_pos - 1);
        seq.num_computed_tokens += chunk_len;

        if (kv_cache->block_size() > 0) {
            int block_size = kv_cache->block_size();
            for (int pos = start_pos + 1; pos <= seq.history_pos; ++pos) {
                if (pos % block_size == 0) {
                    register_completed_block((pos / block_size) - 1);
                }
            }
        }

        if (env_flag("LLM_DEBUG_BATCH_PREFILL")) {
            std::cerr << "[BATCH_PREFILL] real_batch"
                      << " begin=" << prompt_begin
                      << " end=" << prompt_end
                      << " start_pos=" << start_pos
                      << " chunk_len=" << chunk_len
                      << " next_token=" << next_token
                      << " batch_ms=" << last_prefill_chunk_stats.batch_ms
                      << std::endl;
        }
        if (env_flag("LLM_BATCH_PREFILL_COMPARE")) {
            std::cerr << "[BATCH_PREFILL_COMPARE]"
                      << " begin=" << prompt_begin
                      << " end=" << prompt_end
                      << " batch_next=" << next_token
                      << " ref_next=-1"
                      << " match=skip"
                      << " reason=reference_kv_clone_not_available"
                      << std::endl;
        }
        return next_token;
    }

    last_prefill_chunk_stats.fallback = true;
    if (env_flag("LLM_DEBUG_BATCH_PREFILL")) {
        std::cerr << "[BATCH_PREFILL] fallback"
                  << " begin=" << prompt_begin
                  << " end=" << prompt_end
                  << " start_pos=" << start_pos
                  << " reason=batch_failed"
                  << std::endl;
    }
    if (env_flag("LLM_REAL_BATCH_PREFILL_STRICT")) {
        seq.status = SequenceStatus::FAILED;
        seq.error_message = "real batch prefill failed";
        return -1;
    }

    seq.status = SequenceStatus::RUNNING;
    seq.error_message.clear();
    return token_loop();
}

int QwenModel::decode_one_for_sequence(
    SequenceState& seq,
    int input_token,
    KVCacheManager& kv_manager,
    PrefixCache* /*prefix_cache*/
) {
    if (!kv_cache) return -1;
    if (seq.history_pos >= config.max_seq_len) {
        seq.status = SequenceStatus::FAILED;
        seq.error_message = "sequence length exceeded";
        return -1;
    }
    if (!kv_manager.ensure_block_for_position(seq, seq.history_pos)) {
        seq.status = SequenceStatus::FAILED;
        if (seq.error_message.empty()) {
            seq.error_message = "KV block pool exhausted";
        }
        return -1;
    }

    kv_cache->set_active_sequence(&seq.block_table, &seq.max_written_pos);
    int next_token = forward(input_token, seq.history_pos, *kv_cache);
    kv_cache->clear_active_sequence();
    if (next_token < 0) {
        seq.status = SequenceStatus::FAILED;
        seq.error_message = "decode forward failed";
        return -1;
    }

    seq.all_tokens.push_back(input_token);
    seq.history_pos++;
    return next_token;
}

void QwenModel::build_mixed_batch_plan(int row_capacity) {
    if (row_capacity <= 0 || row_capacity > 256) {
        throw std::invalid_argument(
            "mixed batch row capacity must be in range [1, 256]");
    }
    if (fine_mixed_batch_plan) {
        if (mixed_batch_row_capacity != row_capacity) {
            throw std::logic_error(
                "mixed batch plan is already initialized with a different capacity");
        }
        return;
    }

    ExecutablePlanBuilder builder({ExecutionMode::MIXED_SELECTIVE_BATCH, row_capacity});
    PlanValueId previous = builder.add_value({"mixed.input", 1, 1, true});
    for (int layer_id = 0; layer_id < config.num_layers; ++layer_id) {
        const std::string prefix = "layer." + std::to_string(layer_id) + ".";
        auto add_step = [&](const std::string& name, CallbackKernelNode::Callback callback,
                            ParallelismPolicy policy = ParallelismPolicy::SERIAL) {
            PlanValueId next = builder.add_value({name + ".done", 1, 1, false});
            builder.add_node(std::make_unique<CallbackKernelNode>(
                name, std::vector<PlanValueId>{previous},
                std::vector<PlanValueId>{next}, std::move(callback), policy));
            previous = next;
        };

        add_step(prefix + "rmsnorm1", [this, layer_id](ExecutionContext& context) {
            auto& s = *static_cast<MixedPlanRunState*>(context.user_data);
            auto& layer = layers[(size_t)layer_id];
            std::memcpy(s.ws.residual, s.ws.hidden,
                        (size_t)s.total_rows * config.hidden_dim * sizeof(fp16_t));
            Status status = arm_neon::rmsnorm_f16_batch_neon(
                s.ws.hidden, layer.norm1_w.ptr<fp16_t>(), s.ws.norm,
                s.total_rows, config.hidden_dim, config.rms_norm_eps);
            if (status != Status::SUCCESS) s.error = "mixed batch norm1 failed";
            if (status == Status::SUCCESS) {
                // norm1 只 Pack 一次；后续 Q/K/V 三个 Linear 共享相同 Packed A。
                status = arm_neon::pack_gptq_batch_a_f16_neon(
                    s.ws.norm, s.total_rows, config.hidden_dim, s.ws.packed_a);
                if (status != Status::SUCCESS) s.error = "mixed batch norm1 Pack A failed";
            }
            return status;
        });

        add_step(prefix + "attention", [this, layer_id](ExecutionContext& context) {
            auto& s = *static_cast<MixedPlanRunState*>(context.user_data);
            auto& layer = layers[(size_t)layer_id];
            const int T = s.total_rows;
            const int q_size = config.num_q_heads * config.head_dim;
            const int kv_size = config.num_kv_heads * config.head_dim;
            const int num_rep = config.num_q_heads / config.num_kv_heads;
            auto fail = [&](const char* message, Status status) {
                s.error = message;
                return status;
            };
            Status status = arm_neon::linear_gptq_int8_batch_packed_a_neon(
                s.ws.packed_a, T, layer.q_proj, s.ws.q,
                layer.b_q.ptr<fp16_t>(), nullptr, 0);
            if (status != Status::SUCCESS) return fail("selective q projection failed", status);
            status = arm_neon::linear_gptq_int8_batch_packed_a_neon(
                s.ws.packed_a, T, layer.k_proj, s.ws.k,
                layer.b_k.ptr<fp16_t>(), nullptr, 0);
            if (status != Status::SUCCESS) return fail("selective k projection failed", status);
            status = arm_neon::linear_gptq_int8_batch_packed_a_neon(
                s.ws.packed_a, T, layer.v_proj, s.ws.v,
                layer.b_v.ptr<fp16_t>(), nullptr, 0);
            if (status != Status::SUCCESS) return fail("selective v projection failed", status);
            if (s.stats) s.stats->linear_batch_rows += T * 3;

            status = arm_neon::rope_qk_f16_batch_neon(
                s.ws.q, s.ws.k, s.positions->data(), T,
                config.num_q_heads, config.num_kv_heads, config.head_dim,
                cos_cache.ptr<fp16_t>(), sin_cache.ptr<fp16_t>());
            if (status != Status::SUCCESS) return fail("selective batch rope failed", status);

            // 先把本层所有新 K/V 写入各自 Sequence。Prefill 的后续 query
            // 才能看到同一 chunk 中位于它之前的 token。
            for (const MixedBatchItem& item : *s.items) {
                SequenceState* seq = item.seq;
                kv_cache->set_active_sequence(&seq->block_table, &seq->max_written_pos);
                for (int local_row = 0; local_row < item.row_count; ++local_row) {
                    const int row = item.row_begin + local_row;
                    const int pos = (*s.positions)[(size_t)row];
                    kv_cache->update(
                        layer_id, pos,
                        s.ws.k + (size_t)row * kv_size,
                        s.ws.v + (size_t)row * kv_size);
                }
                kv_cache->clear_active_sequence();
                if (s.stats) s.stats->state_modified = true;
            }

            const float scale = 1.0f / std::sqrt((float)config.head_dim);
            // Attention 不能按 total_rows 直接混在一起：每个 item 都有自己的
            // 历史长度和 Paged KV 页表。这里按 Sequence 分段；Prefill 在序列内部
            // 用 Multi-Query Tile 保持因果性，Decode 则仍是单 Query。
            for (const MixedBatchItem& item : *s.items) {
                SequenceState* seq = item.seq;

                // Prefill 一个 item 就是一段连续 Query。Paged KV 可用时，整段走
                // Multi-Query Online-Softmax 内核，不再逐 token 复用 Decode Kernel，
                // 也不再为 Prefill 落地完整 Score 矩阵。
                // Attention 仍以 Sequence 为边界，不与其他会话共享 Block Table。
                if (item.kind == MixedBatchItemKind::PREFILL && item.row_count > 1) {
                    const int final_seq_len = item.start_position + item.row_count;
                    kv_cache->set_active_sequence(
                        &seq->block_table, &seq->max_written_pos);

                    PagedKVView paged_view;
                    const bool use_paged_prefill =
                        s.paged_attention_requested && kv_cache->is_paged() &&
                        kv_cache->get_active_paged_view(&paged_view) &&
                        paged_view.block_table && paged_view.seq_len >= final_seq_len &&
                        paged_view.head_dim == config.head_dim &&
                        paged_view.num_kv_heads == config.num_kv_heads &&
                        paged_view.num_layers > layer_id &&
                        paged_view.block_size > 0 &&
                        kv_cache->raw_k_pages() && kv_cache->raw_v_pages();

                    Status prefill_status = Status::INVALID_ARGUMENT;
                    if (use_paged_prefill) {
                        const int* table = paged_view.block_table->data();
                        const int table_size = static_cast<int>(
                            paged_view.block_table->size());
                        for (int kv_head = 0;
                             kv_head < config.num_kv_heads;
                             ++kv_head) {
                            const fp16_t* q_chunk = s.ws.q +
                                (size_t)item.row_begin * q_size +
                                (size_t)kv_head * num_rep * config.head_dim;
                            fp16_t* out_chunk = s.ws.attn_out +
                                (size_t)item.row_begin * q_size +
                                (size_t)kv_head * num_rep * config.head_dim;

                            prefill_status =
                                arm_neon::attention_prefill_paged_online_f16_neon_public(
                                    q_chunk, out_chunk, item.row_count,
                                    item.start_position, q_size, num_rep,
                                    config.head_dim, scale,
                                    kv_cache->raw_k_pages(),
                                    kv_cache->raw_v_pages(), table, table_size,
                                    paged_view.block_size, layer_id, kv_head,
                                    paged_view.num_layers,
                                    paged_view.num_kv_heads,
                                    paged_view.num_physical_blocks,
                                    s.ws.attention_k_tile,
                                    (size_t)config.head_dim * 16,
                                    s.ws.attention_online_out,
                                    (size_t)4 * config.head_dim);
                            if (prefill_status != Status::SUCCESS) break;
                            record_paged_attention_call();
                        }
                    }
                    kv_cache->clear_active_sequence();

                    if (prefill_status == Status::SUCCESS) {
                        if (s.stats) s.stats->attention_sequence_segments++;
                        continue;
                    }
                    record_paged_attention_fallback();
                    if (s.strict_paged_attention) {
                        return fail(
                            "strict paged prefill attention failed",
                            prefill_status);
                    }
                    // 非 strict 模式保留原逐 Query 路径，仅用于异常兼容回退。
                }

                for (int local_row = 0; local_row < item.row_count; ++local_row) {
                    const int row = item.row_begin + local_row;
                    int current_seq_len = (*s.positions)[(size_t)row] + 1;
                    fp16_t* q_row = s.ws.q + (size_t)row * q_size;
                    fp16_t* out_row = s.ws.attn_out + (size_t)row * q_size;
                kv_cache->set_active_sequence(&seq->block_table, &seq->max_written_pos);

                PagedKVView paged_view;
                bool use_paged = false;
                const fp16_t* raw_k = nullptr;
                const fp16_t* raw_v = nullptr;
                const int* table = nullptr;
                int table_size = 0;
                if (s.paged_attention_requested && kv_cache->is_paged() &&
                    kv_cache->get_active_paged_view(&paged_view) && paged_view.block_table &&
                    paged_view.seq_len >= current_seq_len &&
                    paged_view.head_dim == config.head_dim &&
                    paged_view.num_kv_heads == config.num_kv_heads &&
                    paged_view.num_layers > layer_id && paged_view.block_size > 0) {
                    raw_k = kv_cache->raw_k_pages();
                    raw_v = kv_cache->raw_v_pages();
                    if (raw_k && raw_v) {
                        table = paged_view.block_table->data();
                        table_size = static_cast<int>(paged_view.block_table->size());
                        use_paged = true;
                    }
                }
                if (s.paged_attention_requested && !use_paged) {
                    record_paged_attention_fallback();
                }

                for (int kv_head = 0; kv_head < config.num_kv_heads; ++kv_head) {
                    fp16_t* q_group = q_row + kv_head * num_rep * config.head_dim;
                    fp16_t* out_group = out_row + kv_head * num_rep * config.head_dim;
                    bool used_paged = false;
                    if (use_paged) {
                        Status ps = arm_neon::attention_decode_score_paged_f16_neon_public(
                            q_group, s.ws.score, num_rep, current_seq_len,
                            config.head_dim, scale, raw_k, table, table_size,
                            paged_view.block_size, layer_id, kv_head,
                            paged_view.num_layers, paged_view.num_kv_heads,
                            paged_view.num_physical_blocks);
                        if (ps == Status::SUCCESS) {
                            ps = arm_neon::softmax_f16_inplace_neon(
                                s.ws.score, num_rep, current_seq_len);
                        }
                        if (ps == Status::SUCCESS) {
                            ps = arm_neon::attention_decode_value_paged_f16_neon_public(
                                s.ws.score, out_group, num_rep, current_seq_len,
                                config.head_dim, raw_v, table, table_size,
                                paged_view.block_size, layer_id, kv_head,
                                paged_view.num_layers, paged_view.num_kv_heads,
                                paged_view.num_physical_blocks);
                        }
                        if (ps == Status::SUCCESS) {
                            used_paged = true;
                            record_paged_attention_call();
                        } else {
                            record_paged_attention_fallback();
                            if (s.strict_paged_attention) {
                                kv_cache->clear_active_sequence();
                                return fail("strict paged attention failed", ps);
                            }
                        }
                    }
                    if (!used_paged) {
                        fp16_t* k_ptr = kv_cache->get_k_head_ptr(layer_id, kv_head);
                        fp16_t* v_ptr = kv_cache->get_v_head_ptr(layer_id, kv_head);
                        arm_neon::attention_decode_score_f16_neon_public(
                            q_group, k_ptr, s.ws.score, num_rep,
                            current_seq_len, config.head_dim, scale);
                        status = arm_neon::softmax_f16_inplace_neon(
                            s.ws.score, num_rep, current_seq_len);
                        if (status != Status::SUCCESS) {
                            kv_cache->clear_active_sequence();
                            return fail("selective attention softmax failed", status);
                        }
                        arm_neon::attention_decode_value_f16_neon_public(
                            s.ws.score, v_ptr, out_group, num_rep,
                            current_seq_len, config.head_dim);
                    }
                }
                kv_cache->clear_active_sequence();
                }
                if (s.stats) s.stats->attention_sequence_segments++;
            }

            // Attention 输出是 O Projection 的新输入。统一先覆盖复用 Packed A，
            // 再走与 Q/K/V 相同的 Packed-A Batch Linear。
            status = arm_neon::pack_gptq_batch_a_f16_neon(
                s.ws.attn_out, T, q_size, s.ws.packed_a);
            if (status != Status::SUCCESS) {
                return fail("mixed o projection Pack A failed", status);
            }
            status = arm_neon::linear_gptq_int8_batch_packed_a_neon(
                s.ws.packed_a, T, layer.o_proj, s.ws.proj_out,
                nullptr, nullptr, 0);
            if (status != Status::SUCCESS) return fail("mixed o projection failed", status);
            if (s.stats) s.stats->linear_batch_rows += T;
            return Status::SUCCESS;
        }, ParallelismPolicy::INTERNAL_THREAD_POOL);

        add_step(prefix + "residual.attention", [this](ExecutionContext& context) {
            auto& s = *static_cast<MixedPlanRunState*>(context.user_data);
            arm_neon::add_f16_batch_neon(
                s.ws.residual, s.ws.proj_out, s.ws.hidden,
                (size_t)s.total_rows * config.hidden_dim);
            return Status::SUCCESS;
        });

        add_step(prefix + "rmsnorm2", [this, layer_id](ExecutionContext& context) {
            auto& s = *static_cast<MixedPlanRunState*>(context.user_data);
            auto& layer = layers[(size_t)layer_id];
            std::memcpy(s.ws.residual, s.ws.hidden,
                        (size_t)s.total_rows * config.hidden_dim * sizeof(fp16_t));
            Status status = arm_neon::rmsnorm_f16_batch_neon(
                s.ws.hidden, layer.norm2_w.ptr<fp16_t>(), s.ws.norm,
                s.total_rows, config.hidden_dim, config.rms_norm_eps);
            if (status != Status::SUCCESS) s.error = "mixed batch norm2 failed";
            if (status == Status::SUCCESS) {
                // norm2 只 Pack 一次；后续 Gate/Up 两个 Linear 共享相同 Packed A。
                status = arm_neon::pack_gptq_batch_a_f16_neon(
                    s.ws.norm, s.total_rows, config.hidden_dim, s.ws.packed_a);
                if (status != Status::SUCCESS) s.error = "mixed batch norm2 Pack A failed";
            }
            return status;
        });

        add_step(prefix + "ffn.batch", [this, layer_id](ExecutionContext& context) {
            auto& s = *static_cast<MixedPlanRunState*>(context.user_data);
            auto& layer = layers[(size_t)layer_id];
            const int T = s.total_rows;
            auto fail = [&](const char* message, Status status) {
                s.error = message;
                return status;
            };
            Status status = arm_neon::linear_gptq_int8_batch_packed_a_neon(
                s.ws.packed_a, T, layer.gate_proj, s.ws.gate, nullptr, nullptr, 0);
            if (status != Status::SUCCESS) return fail("selective gate projection failed", status);
            status = arm_neon::linear_gptq_int8_batch_packed_a_neon(
                s.ws.packed_a, T, layer.up_proj, s.ws.up, nullptr, nullptr, 0);
            if (status != Status::SUCCESS) return fail("selective up projection failed", status);
            arm_neon::swiglu_f16_batch_neon(
                s.ws.gate, s.ws.up, T, config.intermediate_size);
            // SwiGLU 原地写回 gate，成为 Down Projection 的新输入。
            // 与其他普通 Linear 一样先 Pack，再调用统一 Packed-A 内核。
            status = arm_neon::pack_gptq_batch_a_f16_neon(
                s.ws.gate, T, config.intermediate_size, s.ws.packed_a);
            if (status != Status::SUCCESS) {
                return fail("selective down projection Pack A failed", status);
            }
            status = arm_neon::linear_gptq_int8_batch_packed_a_neon(
                s.ws.packed_a, T, layer.down_proj, s.ws.proj_out,
                nullptr, nullptr, 0);
            if (status != Status::SUCCESS) return fail("selective down projection failed", status);
            if (s.stats) s.stats->linear_batch_rows += T * 3;
            return Status::SUCCESS;
        }, ParallelismPolicy::INTERNAL_THREAD_POOL);

        add_step(prefix + "residual.ffn", [this](ExecutionContext& context) {
            auto& s = *static_cast<MixedPlanRunState*>(context.user_data);
            arm_neon::add_f16_batch_neon(
                s.ws.residual, s.ws.proj_out, s.ws.hidden,
                (size_t)s.total_rows * config.hidden_dim);
            return Status::SUCCESS;
        });
    }

    PlanValueId final_done = builder.add_value({"mixed.final.done", 1, 1, false});
    builder.add_node(std::make_unique<CallbackKernelNode>(
        "final_norm_and_lm_head_argmax",
        std::vector<PlanValueId>{previous}, std::vector<PlanValueId>{final_done},
        [this](ExecutionContext& context) {
            auto& s = *static_cast<MixedPlanRunState*>(context.user_data);
            const int logit_rows = static_cast<int>(s.logit_rows->size());
            if (logit_rows == 0) {
                return Status::SUCCESS;
            }

            // LM Head 很大，只收集真正需要 next_token 的行：
            // Decode 行，以及“本 chunk 正好结束 Prompt”的 Prefill 最后一行。
            for (int i = 0; i < logit_rows; ++i) {
                const int source_row = (*s.logit_rows)[(size_t)i];
                std::memcpy(
                    s.ws.norm + (size_t)i * config.hidden_dim,
                    s.ws.hidden + (size_t)source_row * config.hidden_dim,
                    (size_t)config.hidden_dim * sizeof(fp16_t));
            }
            Status status = arm_neon::rmsnorm_f16_batch_neon(
                s.ws.norm, final_norm_w.ptr<fp16_t>(), s.ws.residual,
                logit_rows, config.hidden_dim, config.rms_norm_eps);
            if (status != Status::SUCCESS) {
                s.error = "mixed final norm failed";
                return status;
            }
            status = arm_neon::linear_gptq_int8_decode_argmax_batch_neon(
                s.ws.residual, logit_rows, lm_head, s.argmax_results->data(),
                s.ws.argmax_partials, s.ws.argmax_partials_bytes);
            if (status != Status::SUCCESS) {
                s.error = "mixed batch lm_head argmax failed";
                return status;
            }
            if (s.stats) {
                s.stats->linear_batch_rows += logit_rows;
                s.stats->lm_head_rows = logit_rows;
            }
            return Status::SUCCESS;
        }, ParallelismPolicy::INTERNAL_THREAD_POOL));
    // 先完成编译，再一次性申请与 Scheduler token budget 等大的 Workspace。
    // run_mixed_batch_for_sequences() 只取视图，绝不在热路径扩容。
    auto compiled_plan = std::make_unique<ExecutablePlan>(builder.compile());
    if (!ensure_mixed_batch_workspace(row_capacity)) {
        throw std::runtime_error("failed to allocate fixed mixed batch workspace");
    }
    mixed_batch_row_capacity = row_capacity;
    fine_mixed_batch_plan = std::move(compiled_plan);
    std::cerr << "[GRAPH] mode=mixed_selective_batch capacity="
              << mixed_batch_row_capacity << " nodes="
              << fine_mixed_batch_plan->memory_stats().node_count
              << std::endl;
}

bool QwenModel::run_mixed_batch_for_sequences(
    const std::vector<int>& token_ids,
    const std::vector<MixedBatchItem>& items,
    KVCacheManager& kv_manager,
    PrefixCache* prefix_cache,
    std::vector<MixedBatchOutput>* outputs,
    MixedBatchStats* stats
) {
    const uint64_t begin_us = model_now_us();
    const arm_neon::GPTQBatchKernelStats kernel_begin =
        arm_neon::snapshot_gptq_batch_kernel_stats();
    if (stats) {
        *stats = MixedBatchStats{};
        stats->item_count = static_cast<int>(items.size());
        stats->total_rows = static_cast<int>(token_ids.size());
    }
    if (!outputs || !kv_cache || items.empty() || token_ids.empty() ||
        token_ids.size() > 256) {
        return false;
    }
    outputs->resize(items.size());
    for (auto& output : *outputs) output = MixedBatchOutput{};

    const int T = static_cast<int>(token_ids.size());
    const int H = config.hidden_dim;
    std::vector<int> positions((size_t)T);
    std::vector<int> logit_rows;
    std::vector<arm_neon::ArgmaxResult> argmax_results;
    logit_rows.reserve(items.size());
    argmax_results.reserve(items.size());

    auto finish_stats = [&]() {
        if (!stats) return;
        arm_neon::GPTQBatchKernelStats kernel_end =
            arm_neon::snapshot_gptq_batch_kernel_stats();
        arm_neon::GPTQBatchKernelStats delta =
            arm_neon::diff_gptq_batch_kernel_stats(kernel_begin, kernel_end);
        stats->gptq_batch_kernel_calls = delta.kernel_calls;
        stats->gptq_batch_rows_total = delta.rows_total;
        stats->gptq_batch_output_panel_tasks = delta.output_panel_tasks;
        stats->gptq_batch_row_gemv_fallbacks = delta.row_gemv_fallbacks;
        stats->gptq_batch_weight_vector_loads = delta.weight_vector_loads;
        stats->gptq_batch_dequant_vector_ops = delta.dequant_vector_ops;
        stats->gptq_batch_argmax_calls = delta.argmax_calls;
        stats->gptq_batch_argmax_rows = delta.argmax_rows;
        stats->gptq_batch_full_logits_elements_written = delta.full_logits_elements_written;
        stats->gptq_batch_compare_mismatches = delta.compare_mismatches;
        stats->model_ms = static_cast<double>(model_now_us() - begin_us) / 1000.0;
    };
    auto fail_all = [&](const char* message) {
        for (auto& output : *outputs) output.error_message = message;
        finish_stats();
        return false;
    };

    // 运行热路径禁止现场建图或扩容。Scheduler 必须在接收请求前完成
    // 固定容量计划的构建、编译和 Workspace 分配。
    if (!fine_mixed_batch_plan) {
        return fail_all("mixed batch plan is not initialized");
    }
    if (mixed_batch_row_capacity <= 0 ||
        mixed_batch_workspace_max_rows != mixed_batch_row_capacity ||
        !mixed_batch_workspace) {
        return fail_all("mixed batch fixed workspace is not initialized");
    }
    if (T > mixed_batch_row_capacity) {
        return fail_all("mixed batch rows exceed compiled capacity");
    }

    auto valid_batch_weight = [](const arm_neon::GPTQInt8Weight& weight) {
        return weight.K > 0 && weight.N > 0 &&
               weight.qweight_pack.data && weight.scales_pack.data &&
               !weight.has_g_idx;
    };
    for (const auto& layer : layers) {
        if (!valid_batch_weight(layer.q_proj) || !valid_batch_weight(layer.k_proj) ||
            !valid_batch_weight(layer.v_proj) || !valid_batch_weight(layer.o_proj) ||
            !valid_batch_weight(layer.gate_proj) || !valid_batch_weight(layer.up_proj) ||
            !valid_batch_weight(layer.down_proj)) {
            return fail_all("mixed batch does not support g_idx or missing weights");
        }
    }

    // 先验证所有 item，并建立每一行的绝对 position。只有全部检查通过后才能写 KV。
    int expected_row_begin = 0;
    std::unordered_set<SequenceState*> seen_sequences;
    for (size_t item_index = 0; item_index < items.size(); ++item_index) {
        const MixedBatchItem& item = items[item_index];
        if (!item.seq || item.row_begin != expected_row_begin || item.row_count <= 0 ||
            item.start_position != item.seq->history_pos ||
            item.start_position < 0 ||
            item.start_position + item.row_count > config.max_seq_len ||
            item.row_begin + item.row_count > T ||
            (item.kind == MixedBatchItemKind::DECODE && item.row_count != 1)) {
            (*outputs)[item_index].error_message = "invalid mixed batch item";
            finish_stats();
            return false;
        }
        // 同一个 Session 不能在同一批里出现两段，否则二者会从相同 history_pos
        // 写入重叠 KV。并发请求必须使用不同 SessionId，同一 Session 应串行提交。
        if (!seen_sequences.insert(item.seq).second) {
            (*outputs)[item_index].error_message =
                "duplicate sequence in one mixed batch";
            finish_stats();
            return false;
        }
        for (int local_row = 0; local_row < item.row_count; ++local_row) {
            const int row = item.row_begin + local_row;
            const int token_id = token_ids[(size_t)row];
            if (token_id < 0 || token_id >= config.vocab_size) {
                (*outputs)[item_index].error_message = "mixed batch token id out of range";
                finish_stats();
                return false;
            }
            positions[(size_t)row] = item.start_position + local_row;
        }
        if (item.requires_logits) {
            logit_rows.push_back(item.row_begin + item.row_count - 1);
        }
        if (stats) {
            if (item.kind == MixedBatchItemKind::DECODE) {
                stats->decode_rows += item.row_count;
            } else {
                stats->prefill_rows += item.row_count;
            }
        }
        expected_row_begin += item.row_count;
    }
    if (expected_row_begin != T) {
        return fail_all("mixed batch rows are not contiguous");
    }
    if (logit_rows.size() >
        static_cast<size_t>(arm_neon::GPTQ_BATCH_ARGMAX_MAX_ROWS)) {
        return fail_all("mixed batch logit rows exceed batch argmax capacity");
    }
    if (!logit_rows.empty() && !valid_batch_weight(lm_head)) {
        return fail_all("mixed batch does not support lm_head g_idx or missing weights");
    }
    argmax_results.resize(logit_rows.size());

    // Workspace 已按 LLM_MAX_BATCHED_TOKENS 固定分配。本轮 T 可以更小，
    // 但不得触发申请或扩容，因此运行期 reallocations 始终为 0。
    MixedBatchWorkspaceView ws = mixed_batch_workspace_view(T);
    if (!ws.hidden || !ws.score || !ws.attention_k_tile ||
        !ws.attention_online_out || !ws.argmax_partials) {
        return fail_all("invalid mixed batch workspace");
    }

    // 一次性为所有 Sequence 预留页，避免处理中途才发现 KV block 不足。
    for (size_t item_index = 0; item_index < items.size(); ++item_index) {
        const MixedBatchItem& item = items[item_index];
        if (!kv_manager.ensure_blocks_for_range(
                *item.seq, item.start_position, item.start_position + item.row_count)) {
            item.seq->status = SequenceStatus::FAILED;
            item.seq->error_message = "failed to allocate mixed batch KV blocks";
            (*outputs)[item_index].error_message = item.seq->error_message;
            finish_stats();
            return false;
        }
    }

    last_batch_norm_out_valid_ = false;

    for (int row = 0; row < T; ++row) {
        const fp16_t* embed =
            embed_tokens_w.ptr<fp16_t>() + (size_t)token_ids[(size_t)row] * H;
        std::memcpy(
            ws.hidden + (size_t)row * H,
            embed,
            (size_t)H * sizeof(fp16_t));
    }

    MixedPlanRunState run_state;
    run_state.items = &items;
    run_state.outputs = outputs;
    run_state.stats = stats;
    run_state.ws = ws;
    run_state.positions = &positions;
    run_state.logit_rows = &logit_rows;
    run_state.argmax_results = &argmax_results;
    run_state.total_rows = T;
    run_state.paged_attention_requested = kv_cache->is_paged();
    run_state.strict_paged_attention = env_flag("LLM_PAGED_ATTENTION_STRICT");

    unsigned char trigger = 0;
    ExecutionContext mixed_context;
    mixed_context.actual_rows = T;
    mixed_context.user_data = &run_state;
    mixed_context.bind_external(0, &trigger);
    Status status = fine_mixed_batch_plan->run(
        mixed_context, fine_mixed_control_workspace);
    if (status != Status::SUCCESS) {
        return fail_all(run_state.error.empty()
            ? "mixed execution plan failed" : run_state.error.c_str());
    }

    auto register_completed_blocks = [&](SequenceState& seq, int old_pos, int new_pos) {
        if (!prefix_cache || !kv_cache || kv_cache->block_size() <= 0) return;
        const int block_size = kv_cache->block_size();
        for (int pos = old_pos + 1; pos <= new_pos; ++pos) {
            if (pos % block_size != 0) continue;
            const int logical_block = (pos / block_size) - 1;
            const int block_begin = logical_block * block_size;
            if (logical_block < 0 ||
                logical_block >= static_cast<int>(seq.block_table.size()) ||
                block_begin + block_size > static_cast<int>(seq.all_tokens.size())) {
                continue;
            }
            const int physical_block = seq.block_table[(size_t)logical_block];
            if (physical_block < 0 || kv_manager.block_has_hash(physical_block)) continue;

            HashValue parent_hash;
            if (logical_block > 0) {
                const int previous_block = seq.block_table[(size_t)(logical_block - 1)];
                if (previous_block < 0 || !kv_manager.block_has_hash(previous_block)) continue;
                parent_hash = kv_manager.block_hash(previous_block);
            }
            HashValue current_hash = hash_token_block(
                parent_hash, seq.all_tokens, block_begin, block_size, prefix_cache->config());
            if (prefix_cache->insert(
                    current_hash, parent_hash, seq.all_tokens,
                    block_begin, block_size, physical_block)) {
                kv_manager.attach_hash_to_block(
                    physical_block, current_hash, parent_hash, block_size);
                seq.last_prefix_hash = current_hash;
            }
        }
    };

    size_t logit_index = 0;
    for (size_t item_index = 0; item_index < items.size(); ++item_index) {
        const MixedBatchItem& item = items[item_index];
        MixedBatchOutput& output = (*outputs)[item_index];
        if (item.requires_logits) {
            const int best_id = argmax_results[logit_index++].index;
            if (best_id < 0 || best_id >= config.vocab_size) {
                output.error_message = "mixed lm_head argmax failed";
                finish_stats();
                return false;
            }
            output.next_token = best_id;
        }

        // 只有整轮模型成功后才提交 token history/cursor；KV 在计算中已写好。
        SequenceState& seq = *item.seq;
        const int old_pos = seq.history_pos;
        seq.all_tokens.insert(
            seq.all_tokens.end(),
            token_ids.begin() + item.row_begin,
            token_ids.begin() + item.row_begin + item.row_count);
        seq.history_pos += item.row_count;
        seq.max_written_pos = std::max(seq.max_written_pos, seq.history_pos - 1);
        seq.num_computed_tokens += item.row_count;
        register_completed_blocks(seq, old_pos, seq.history_pos);
        output.success = true;
    }
    finish_stats();
    return true;
}

bool QwenModel::decode_selective_batch_for_sequences(
    const std::vector<SelectiveDecodeItem>& items,
    KVCacheManager& kv_manager,
    PrefixCache* prefix_cache,
    std::vector<SelectiveDecodeOutput>* outputs,
    SelectiveDecodeStats* stats) {
    if (!outputs || items.empty()) return false;

    std::vector<int> token_ids;
    std::vector<MixedBatchItem> mixed_items;
    token_ids.reserve(items.size());
    mixed_items.reserve(items.size());
    for (size_t i = 0; i < items.size(); ++i) {
        token_ids.push_back(items[i].input_token);
        mixed_items.push_back(MixedBatchItem{
            MixedBatchItemKind::DECODE,
            items[i].seq,
            static_cast<int>(i),
            1,
            items[i].seq ? items[i].seq->history_pos : 0,
            true});
    }

    std::vector<MixedBatchOutput> mixed_outputs;
    MixedBatchStats mixed_stats;
    const bool ok = run_mixed_batch_for_sequences(
        token_ids, mixed_items, kv_manager, prefix_cache, &mixed_outputs, &mixed_stats);

    outputs->resize(mixed_outputs.size());
    for (size_t i = 0; i < mixed_outputs.size(); ++i) {
        (*outputs)[i].next_token = mixed_outputs[i].next_token;
        (*outputs)[i].success = mixed_outputs[i].success;
        (*outputs)[i].error_message = mixed_outputs[i].error_message;
    }
    if (stats) {
        *stats = SelectiveDecodeStats{};
        stats->batch_size = static_cast<int>(items.size());
        stats->linear_batch_rows = mixed_stats.linear_batch_rows;
        stats->attention_per_sequence_calls = mixed_stats.attention_sequence_segments;
        stats->lm_head_rows = mixed_stats.lm_head_rows;
        stats->state_modified = mixed_stats.state_modified;
        stats->gptq_batch_kernel_calls = mixed_stats.gptq_batch_kernel_calls;
        stats->gptq_batch_rows_total = mixed_stats.gptq_batch_rows_total;
        stats->gptq_batch_output_panel_tasks = mixed_stats.gptq_batch_output_panel_tasks;
        stats->gptq_batch_row_gemv_fallbacks = mixed_stats.gptq_batch_row_gemv_fallbacks;
        stats->gptq_batch_weight_vector_loads = mixed_stats.gptq_batch_weight_vector_loads;
        stats->gptq_batch_dequant_vector_ops = mixed_stats.gptq_batch_dequant_vector_ops;
        stats->gptq_batch_argmax_calls = mixed_stats.gptq_batch_argmax_calls;
        stats->gptq_batch_argmax_rows = mixed_stats.gptq_batch_argmax_rows;
        stats->gptq_batch_full_logits_elements_written =
            mixed_stats.gptq_batch_full_logits_elements_written;
        stats->gptq_batch_compare_mismatches = mixed_stats.gptq_batch_compare_mismatches;
        stats->selective_decode_hotpath_allocations = mixed_stats.mixed_batch_hotpath_allocations;
        stats->selective_decode_workspace_reallocations =
            mixed_stats.mixed_batch_workspace_reallocations;
        stats->model_ms = mixed_stats.model_ms;
    }
    return ok;
}

int QwenModel::decode_one_for_sequence_sampled(
    SequenceState& seq,
    int input_token,
    KVCacheManager& kv_manager,
    PrefixCache* prefix_cache,
    const SamplingParams& sampling,
    std::mt19937_64& rng,
    SamplingRuntimeStats* stats
) {
    if (!sampling_enabled(sampling)) {
        int next = decode_one_for_sequence(seq, input_token, kv_manager, prefix_cache);
        if (stats && next >= 0) {
            stats->greedy_tokens++;
        }
        return next;
    }

    if (!kv_cache) return -1;
    if (seq.history_pos >= config.max_seq_len) {
        seq.status = SequenceStatus::FAILED;
        seq.error_message = "sequence length exceeded";
        return -1;
    }
    if (!kv_manager.ensure_block_for_position(seq, seq.history_pos)) {
        seq.status = SequenceStatus::FAILED;
        if (seq.error_message.empty()) {
            seq.error_message = "KV block pool exhausted";
        }
        return -1;
    }

    kv_cache->set_active_sequence(&seq.block_table, &seq.max_written_pos);
    int greedy_next = forward(input_token, seq.history_pos, *kv_cache);
    kv_cache->clear_active_sequence();
    if (greedy_next < 0) {
        seq.status = SequenceStatus::FAILED;
        seq.error_message = "decode forward failed";
        return -1;
    }

    int sampled_next = sample_next_token_from_last_logits(sampling, rng, stats);
    if (sampled_next < 0) {
        sampled_next = greedy_next;
        if (stats) {
            stats->greedy_tokens++;
        }
    }

    seq.all_tokens.push_back(input_token);
    seq.history_pos++;
    return sampled_next;
}

void QwenModel::generate_for_sequence(
    SequenceState& seq,
    const std::vector<int>& input_tokens,
    int max_new_tokens,
    KVCacheManager& kv_manager,
    PrefixCache* prefix_cache,
    std::function<bool(int)> callback
) {
    if (!kv_cache) return;

    seq.status = SequenceStatus::RUNNING;
    seq.error_message.clear();
    kv_manager.init_sequence(seq);

    auto fail = [&](const std::string& message) {
        seq.status = SequenceStatus::FAILED;
        seq.error_message = message;
        kv_cache->clear_active_sequence();
    };

    auto maybe_register_completed_block = [&](int logical_block) {
        if (!prefix_cache || logical_block < 0) {
            return;
        }
        int block_size = kv_cache->block_size();
        int block_begin = logical_block * block_size;
        if (block_begin < 0 ||
            block_begin + block_size > static_cast<int>(seq.all_tokens.size()) ||
            logical_block >= static_cast<int>(seq.block_table.size())) {
            return;
        }

        int physical_block = seq.block_table[(size_t)logical_block];
        if (physical_block < 0 ||
            kv_manager.block_has_hash(physical_block)) {
            return;
        }

        HashValue parent_hash;
        if (logical_block > 0) {
            int prev_block = seq.block_table[(size_t)(logical_block - 1)];
            if (prev_block < 0 || !kv_manager.block_has_hash(prev_block)) {
                return;
            }
            parent_hash = kv_manager.block_hash(prev_block);
        }

        HashValue current_hash = hash_token_block(
            parent_hash,
            seq.all_tokens,
            block_begin,
            block_size,
            prefix_cache->config());
        if (prefix_cache->insert(
                current_hash,
                parent_hash,
                seq.all_tokens,
                block_begin,
                block_size,
                physical_block)) {
            kv_manager.attach_hash_to_block(
                physical_block,
                current_hash,
                parent_hash,
                block_size);
            seq.last_prefix_hash = current_hash;
        }
    };

    int prefill_begin = (seq.cached_prefix_tokens > 0 &&
                         seq.history_pos == seq.cached_prefix_tokens)
        ? seq.cached_prefix_tokens
        : 0;
    if (prefill_begin < 0) {
        prefill_begin = 0;
    }
    if (prefill_begin > static_cast<int>(input_tokens.size())) {
        prefill_begin = static_cast<int>(input_tokens.size());
    }
    if (prefill_begin == static_cast<int>(input_tokens.size()) && !input_tokens.empty()) {
        prefill_begin = static_cast<int>(input_tokens.size()) - 1;
        seq.history_pos = prefill_begin;
        if (seq.all_tokens.size() > (size_t)prefill_begin) {
            seq.all_tokens.resize((size_t)prefill_begin);
        }
    }

    int next_token = -1;
    for (int token_index = prefill_begin; token_index < static_cast<int>(input_tokens.size()); ++token_index) {
        int tok = input_tokens[(size_t)token_index];
        if (seq.history_pos >= config.max_seq_len) {
            fail("sequence length exceeded");
            return;
        }
        if (!kv_manager.ensure_block_for_position(seq, seq.history_pos)) {
            fail(seq.error_message.empty() ? "KV block pool exhausted" : seq.error_message);
            return;
        }

        kv_cache->set_active_sequence(&seq.block_table, &seq.max_written_pos);
        next_token = forward(tok, seq.history_pos, *kv_cache);
        kv_cache->clear_active_sequence();

        if (next_token < 0) {
            fail("prompt forward failed");
            return;
        }
        seq.all_tokens.push_back(tok);
        seq.history_pos++;
        seq.num_computed_tokens++;
        if (prefix_cache && seq.history_pos % kv_cache->block_size() == 0) {
            maybe_register_completed_block((seq.history_pos / kv_cache->block_size()) - 1);
        }
    }

    int current_token = next_token;
    for (int i = 0; i < max_new_tokens && current_token >= 0; ++i) {
        if (is_stop_token(current_token)) {
            if (seq.history_pos < config.max_seq_len) {
                if (!kv_manager.ensure_block_for_position(seq, seq.history_pos)) {
                    fail(seq.error_message.empty() ? "KV block pool exhausted" : seq.error_message);
                    return;
                }
                kv_cache->set_active_sequence(&seq.block_table, &seq.max_written_pos);
                int ignored = forward(current_token, seq.history_pos, *kv_cache);
                kv_cache->clear_active_sequence();
                if (ignored < 0) {
                    fail("stop token forward failed");
                    return;
                }
                seq.all_tokens.push_back(current_token);
                seq.history_pos++;
            }
            break;
        }

        seq.generated_tokens.push_back(current_token);
        if (!callback(current_token)) {
            seq.status = SequenceStatus::ABORTED;
            kv_cache->clear_active_sequence();
            return;
        }

        if (seq.history_pos >= config.max_seq_len) {
            break;
        }
        if (!kv_manager.ensure_block_for_position(seq, seq.history_pos)) {
            fail(seq.error_message.empty() ? "KV block pool exhausted" : seq.error_message);
            return;
        }

        kv_cache->set_active_sequence(&seq.block_table, &seq.max_written_pos);
        int produced = forward(current_token, seq.history_pos, *kv_cache);
        kv_cache->clear_active_sequence();

        if (produced < 0) {
            fail("decode forward failed");
            return;
        }
        seq.all_tokens.push_back(current_token);
        seq.history_pos++;
        current_token = produced;
    }

    seq.status = SequenceStatus::FINISHED;
    kv_cache->clear_active_sequence();
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
