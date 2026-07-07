#pragma once

#include <cstdint>
#include <string>

#include "llm_engine/engine/sequence_state.h"

namespace llm_engine {

struct PagedAttentionStats {
    int64_t calls = 0;
    int64_t fallbacks = 0;
    int64_t compare_warnings = 0;
};

struct RequestMetrics {
    uint64_t request_id = 0;
    SessionId session_id = 0;

    int prompt_tokens = 0;
    int generated_tokens = 0;
    int max_new_tokens = 0;

    int cached_prefix_tokens = 0;
    int cached_prefix_blocks = 0;
    int computed_prefill_tokens = 0;
    int prefill_chunks = 0;
    int real_batch_prefill_chunks = 0;
    int token_loop_prefill_chunks = 0;
    int batch_prefill_fallbacks = 0;
    int batch_prefill_compare_mismatches = 0;

    int kv_total_blocks = 0;
    int kv_free_blocks = 0;
    int kv_active_blocks = 0;
    int kv_cached_blocks = 0;
    int kv_lru_size = 0;

    int64_t paged_attention_calls = 0;
    int64_t paged_attention_fallbacks = 0;
    int64_t paged_attention_compare_warnings = 0;

    bool continuous_batching_enabled = false;
    int decode_batch_steps = 0;
    int decode_batch_size_sum = 0;
    int decode_batch_size_max = 0;
    double decode_batch_size_avg = 0.0;
    int prefill_chunk_steps = 0;
    int prefill_scheduler_yield_count = 0;
    int scheduler_v2_steps = 0;
    int scheduler_v2_decode_steps = 0;
    int scheduler_v2_prefill_steps = 0;
    int active_decode_batch_size_at_finish = 0;

    bool prefill_batching_enabled = false;
    int prefill_microbatch_steps = 0;
    int prefill_microbatch_size_sum = 0;
    int prefill_microbatch_size_max = 0;
    double prefill_microbatch_size_avg = 0.0;
    int prefill_full_chunk_steps = 0;
    int prefill_tail_chunk_steps = 0;
    int prefill_tail_wait_steps = 0;
    int prefill_requeue_count = 0;
    int prefill_microbatch_items_total = 0;
    int prefill_microbatch_tokens_total = 0;
    std::string prefill_microbatch_executor = "none";

    double queue_wait_ms = 0.0;
    double prefill_ms = 0.0;
    double batch_prefill_ms = 0.0;
    double token_loop_prefill_ms = 0.0;
    double decode_ms = 0.0;
    double sampling_ms = 0.0;
    double total_ms = 0.0;
    double first_token_ms = 0.0;
    double tokens_per_second = 0.0;

    bool sampling_enabled = false;
    float temperature = 0.0f;
    int top_k = 0;
    float top_p = 1.0f;
    uint64_t seed = 0;
    int sampled_tokens = 0;
    int greedy_tokens = 0;

    std::string final_status;
    std::string error_message;
};

class ScopedTimer {
public:
    explicit ScopedTimer(double* out_ms);
    ~ScopedTimer();

    ScopedTimer(const ScopedTimer&) = delete;
    ScopedTimer& operator=(const ScopedTimer&) = delete;

private:
    double* out_ms_ = nullptr;
    uint64_t start_us_ = 0;
};

uint64_t now_us();
double elapsed_ms(uint64_t begin_us, uint64_t end_us);

void record_paged_attention_call();
void record_paged_attention_fallback();
void record_paged_attention_compare_warning();
PagedAttentionStats snapshot_paged_attention_stats();
PagedAttentionStats diff_paged_attention_stats(
    const PagedAttentionStats& begin,
    const PagedAttentionStats& end);

std::string request_metrics_to_json(const RequestMetrics& metrics);
void emit_request_metrics(const RequestMetrics& metrics);

} // namespace llm_engine
