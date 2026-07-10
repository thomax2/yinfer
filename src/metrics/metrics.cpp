#include "llm_engine/metrics/metrics.h"

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <mutex>
#include <sstream>

namespace llm_engine {

namespace {

std::atomic<int64_t> g_paged_attention_calls{0};
std::atomic<int64_t> g_paged_attention_fallbacks{0};
std::atomic<int64_t> g_paged_attention_compare_warnings{0};
std::mutex g_metrics_file_mu;

bool env_flag(const char* name) {
    const char* v = std::getenv(name);
    if (!v) return false;
    std::string s(v);
    return s == "1" || s == "true" || s == "TRUE" || s == "on" || s == "ON";
}

std::string json_escape(const std::string& s) {
    std::ostringstream os;
    for (unsigned char c : s) {
        switch (c) {
            case '\\': os << "\\\\"; break;
            case '"': os << "\\\""; break;
            case '\b': os << "\\b"; break;
            case '\f': os << "\\f"; break;
            case '\n': os << "\\n"; break;
            case '\r': os << "\\r"; break;
            case '\t': os << "\\t"; break;
            default:
                if (c < 0x20) {
                    static const char* hex = "0123456789abcdef";
                    os << "\\u00" << hex[(c >> 4) & 0xf] << hex[c & 0xf];
                } else {
                    os << c;
                }
                break;
        }
    }
    return os.str();
}

} // namespace

ScopedTimer::ScopedTimer(double* out_ms)
    : out_ms_(out_ms), start_us_(now_us()) {}

ScopedTimer::~ScopedTimer() {
    if (out_ms_) {
        *out_ms_ += elapsed_ms(start_us_, now_us());
    }
}

uint64_t now_us() {
    using clock = std::chrono::steady_clock;
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::microseconds>(
            clock::now().time_since_epoch()).count());
}

double elapsed_ms(uint64_t begin_us, uint64_t end_us) {
    if (end_us <= begin_us) {
        return 0.0;
    }
    return static_cast<double>(end_us - begin_us) / 1000.0;
}

void record_paged_attention_call() {
    g_paged_attention_calls.fetch_add(1, std::memory_order_relaxed);
}

void record_paged_attention_fallback() {
    g_paged_attention_fallbacks.fetch_add(1, std::memory_order_relaxed);
}

void record_paged_attention_compare_warning() {
    g_paged_attention_compare_warnings.fetch_add(1, std::memory_order_relaxed);
}

PagedAttentionStats snapshot_paged_attention_stats() {
    PagedAttentionStats stats;
    stats.calls = g_paged_attention_calls.load(std::memory_order_relaxed);
    stats.fallbacks = g_paged_attention_fallbacks.load(std::memory_order_relaxed);
    stats.compare_warnings = g_paged_attention_compare_warnings.load(std::memory_order_relaxed);
    return stats;
}

PagedAttentionStats diff_paged_attention_stats(
    const PagedAttentionStats& begin,
    const PagedAttentionStats& end
) {
    PagedAttentionStats diff;
    diff.calls = end.calls - begin.calls;
    diff.fallbacks = end.fallbacks - begin.fallbacks;
    diff.compare_warnings = end.compare_warnings - begin.compare_warnings;
    return diff;
}

std::string request_metrics_to_json(const RequestMetrics& m) {
    std::ostringstream os;
    os << '{'
       << "\"request_id\":" << m.request_id
       << ",\"session_id\":" << m.session_id
       << ",\"prompt_tokens\":" << m.prompt_tokens
       << ",\"generated_tokens\":" << m.generated_tokens
       << ",\"max_new_tokens\":" << m.max_new_tokens
       << ",\"cached_prefix_tokens\":" << m.cached_prefix_tokens
       << ",\"cached_prefix_blocks\":" << m.cached_prefix_blocks
       << ",\"computed_prefill_tokens\":" << m.computed_prefill_tokens
       << ",\"prefill_chunks\":" << m.prefill_chunks
       << ",\"real_batch_prefill_chunks\":" << m.real_batch_prefill_chunks
       << ",\"token_loop_prefill_chunks\":" << m.token_loop_prefill_chunks
       << ",\"batch_prefill_fallbacks\":" << m.batch_prefill_fallbacks
       << ",\"batch_prefill_compare_mismatches\":" << m.batch_prefill_compare_mismatches
       << ",\"kv_total_blocks\":" << m.kv_total_blocks
       << ",\"kv_free_blocks\":" << m.kv_free_blocks
       << ",\"kv_active_blocks\":" << m.kv_active_blocks
       << ",\"kv_cached_blocks\":" << m.kv_cached_blocks
       << ",\"kv_lru_size\":" << m.kv_lru_size
       << ",\"paged_attention_calls\":" << m.paged_attention_calls
       << ",\"paged_attention_fallbacks\":" << m.paged_attention_fallbacks
       << ",\"paged_attention_compare_warnings\":" << m.paged_attention_compare_warnings
       << ",\"continuous_batching_enabled\":" << (m.continuous_batching_enabled ? "true" : "false")
       << ",\"decode_batch_steps\":" << m.decode_batch_steps
       << ",\"decode_batch_size_sum\":" << m.decode_batch_size_sum
       << ",\"decode_batch_size_max\":" << m.decode_batch_size_max
       << ",\"decode_batch_size_avg\":" << m.decode_batch_size_avg
       << ",\"prefill_chunk_steps\":" << m.prefill_chunk_steps
       << ",\"prefill_scheduler_yield_count\":" << m.prefill_scheduler_yield_count
       << ",\"scheduler_v2_steps\":" << m.scheduler_v2_steps
       << ",\"scheduler_v2_decode_steps\":" << m.scheduler_v2_decode_steps
       << ",\"scheduler_v2_prefill_steps\":" << m.scheduler_v2_prefill_steps
       << ",\"active_decode_batch_size_at_finish\":" << m.active_decode_batch_size_at_finish
       << ",\"selective_decode_enabled\":" << (m.selective_decode_enabled ? "true" : "false")
       << ",\"selective_decode_steps\":" << m.selective_decode_steps
       << ",\"selective_decode_size_sum\":" << m.selective_decode_size_sum
       << ",\"selective_decode_size_max\":" << m.selective_decode_size_max
       << ",\"selective_decode_size_avg\":" << m.selective_decode_size_avg
       << ",\"selective_decode_linear_batch_rows\":" << m.selective_decode_linear_batch_rows
       << ",\"selective_decode_attention_per_sequence_calls\":"
       << m.selective_decode_attention_per_sequence_calls
       << ",\"selective_decode_lm_head_rows\":" << m.selective_decode_lm_head_rows
       << ",\"selective_decode_fallbacks\":" << m.selective_decode_fallbacks
       << ",\"selective_decode_model_ms\":" << m.selective_decode_model_ms
       << ",\"gptq_batch_kernel_calls\":" << m.gptq_batch_kernel_calls
       << ",\"gptq_batch_rows_total\":" << m.gptq_batch_rows_total
       << ",\"gptq_batch_output_panel_tasks\":" << m.gptq_batch_output_panel_tasks
       << ",\"gptq_batch_row_gemv_fallbacks\":" << m.gptq_batch_row_gemv_fallbacks
       << ",\"gptq_batch_weight_vector_loads\":" << m.gptq_batch_weight_vector_loads
       << ",\"gptq_batch_dequant_vector_ops\":" << m.gptq_batch_dequant_vector_ops
       << ",\"gptq_batch_argmax_calls\":" << m.gptq_batch_argmax_calls
       << ",\"gptq_batch_argmax_rows\":" << m.gptq_batch_argmax_rows
       << ",\"gptq_batch_full_logits_elements_written\":"
       << m.gptq_batch_full_logits_elements_written
       << ",\"gptq_batch_compare_mismatches\":" << m.gptq_batch_compare_mismatches
       << ",\"selective_decode_hotpath_allocations\":"
       << m.selective_decode_hotpath_allocations
       << ",\"selective_decode_workspace_reallocations\":"
       << m.selective_decode_workspace_reallocations
       << ",\"selective_decode_mode\":\"" << json_escape(m.selective_decode_mode) << "\""
       << ",\"prefill_batching_enabled\":" << (m.prefill_batching_enabled ? "true" : "false")
       << ",\"prefill_microbatch_steps\":" << m.prefill_microbatch_steps
       << ",\"prefill_microbatch_size_sum\":" << m.prefill_microbatch_size_sum
       << ",\"prefill_microbatch_size_max\":" << m.prefill_microbatch_size_max
       << ",\"prefill_microbatch_size_avg\":" << m.prefill_microbatch_size_avg
       << ",\"prefill_full_chunk_steps\":" << m.prefill_full_chunk_steps
       << ",\"prefill_tail_chunk_steps\":" << m.prefill_tail_chunk_steps
       << ",\"prefill_tail_wait_steps\":" << m.prefill_tail_wait_steps
       << ",\"prefill_requeue_count\":" << m.prefill_requeue_count
       << ",\"prefill_microbatch_items_total\":" << m.prefill_microbatch_items_total
       << ",\"prefill_microbatch_tokens_total\":" << m.prefill_microbatch_tokens_total
       << ",\"prefill_microbatch_executor\":\"" << json_escape(m.prefill_microbatch_executor) << "\""
       << ",\"queue_wait_ms\":" << m.queue_wait_ms
       << ",\"prefill_ms\":" << m.prefill_ms
       << ",\"batch_prefill_ms\":" << m.batch_prefill_ms
       << ",\"token_loop_prefill_ms\":" << m.token_loop_prefill_ms
       << ",\"decode_ms\":" << m.decode_ms
       << ",\"sampling_ms\":" << m.sampling_ms
       << ",\"total_ms\":" << m.total_ms
       << ",\"first_token_ms\":" << m.first_token_ms
       << ",\"tokens_per_second\":" << m.tokens_per_second
       << ",\"sampling_enabled\":" << (m.sampling_enabled ? "true" : "false")
       << ",\"temperature\":" << m.temperature
       << ",\"top_k\":" << m.top_k
       << ",\"top_p\":" << m.top_p
       << ",\"seed\":" << m.seed
       << ",\"sampled_tokens\":" << m.sampled_tokens
       << ",\"greedy_tokens\":" << m.greedy_tokens
       << ",\"final_status\":\"" << json_escape(m.final_status) << "\""
       << ",\"error_message\":\"" << json_escape(m.error_message) << "\""
       << '}';
    return os.str();
}

void emit_request_metrics(const RequestMetrics& metrics) {
    const bool debug = env_flag("LLM_DEBUG_METRICS");
    const char* jsonl_path = std::getenv("LLM_METRICS_JSONL");
    if (!debug && (!jsonl_path || !*jsonl_path)) {
        return;
    }

    std::string line = request_metrics_to_json(metrics);
    if (debug) {
        std::cerr << "[METRICS] " << line << std::endl;
    }
    if (jsonl_path && *jsonl_path) {
        std::lock_guard<std::mutex> lk(g_metrics_file_mu);
        std::ofstream out(jsonl_path, std::ios::app);
        if (!out) {
            std::cerr << "[METRICS] failed to open jsonl path=" << jsonl_path << std::endl;
            return;
        }
        out << line << '\n';
    }
}

} // namespace llm_engine
