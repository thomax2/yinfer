#pragma once

#include <cstdint>
#include <mutex>
#include <string>

#include "llm_engine/metrics/metrics.h"

namespace llm_engine {

class ServerMetrics {
public:
    void connection_opened();
    void connection_closed();
    void request_started(int prompt_tokens);
    void request_finished(const RequestMetrics& metrics);
    void request_failed();
    void request_aborted();
    void set_pending_requests(int pending);
    void set_active_requests(int active);

    int active_connections() const;
    int active_requests() const;
    int pending_requests() const;
    std::string to_json() const;

private:
    mutable std::mutex mu_;
    int64_t requests_total_ = 0;
    int64_t requests_finished_ = 0;
    int64_t requests_failed_ = 0;
    int64_t requests_aborted_ = 0;
    int64_t tokens_generated_total_ = 0;
    int64_t prompt_tokens_total_ = 0;
    int active_requests_ = 0;
    int pending_requests_ = 0;
    int active_connections_ = 0;
    int64_t prefix_cached_tokens_total_ = 0;
    int64_t prefix_cached_blocks_total_ = 0;
    int64_t prefix_hit_requests_ = 0;
    int64_t real_batch_prefill_chunks_total_ = 0;
    int64_t paged_attention_calls_ = 0;
    int64_t paged_attention_fallbacks_ = 0;
    bool scheduler_enabled_ = false;
    int64_t decode_batch_steps_total_ = 0;
    int64_t decode_batch_size_sum_ = 0;
    int decode_batch_size_max_ = 0;
    int64_t prefill_chunk_steps_total_ = 0;
    int64_t scheduler_decode_steps_total_ = 0;
    int64_t scheduler_prefill_steps_total_ = 0;
    int decode_queue_size_ = 0;
    bool mixed_batch_enabled_ = false;
    int64_t mixed_batch_steps_total_ = 0;
    int64_t mixed_batch_total_rows_sum_ = 0;
    int mixed_batch_total_rows_max_ = 0;
    int64_t mixed_batch_decode_rows_total_ = 0;
    int64_t mixed_batch_prefill_rows_total_ = 0;
    int64_t mixed_batch_attention_segments_total_ = 0;
    int64_t mixed_batch_lm_head_rows_total_ = 0;
    int64_t mixed_batch_fallbacks_total_ = 0;
    double mixed_batch_model_ms_total_ = 0.0;
    bool selective_decode_enabled_ = false;
    int64_t selective_decode_steps_total_ = 0;
    int64_t selective_decode_size_sum_ = 0;
    int selective_decode_size_max_ = 0;
    int64_t selective_decode_linear_batch_rows_total_ = 0;
    int64_t selective_decode_attention_per_sequence_calls_total_ = 0;
    int64_t selective_decode_lm_head_rows_total_ = 0;
    int64_t selective_decode_fallbacks_total_ = 0;
    double selective_decode_model_ms_total_ = 0.0;
    uint64_t gptq_batch_kernel_calls_ = 0;
    uint64_t gptq_batch_rows_total_ = 0;
    uint64_t gptq_batch_output_panel_tasks_ = 0;
    uint64_t gptq_batch_row_gemv_fallbacks_ = 0;
    uint64_t gptq_batch_weight_vector_loads_ = 0;
    uint64_t gptq_batch_dequant_vector_ops_ = 0;
    uint64_t gptq_batch_argmax_calls_ = 0;
    uint64_t gptq_batch_argmax_rows_ = 0;
    uint64_t gptq_batch_full_logits_elements_written_ = 0;
    uint64_t gptq_batch_compare_mismatches_ = 0;
    uint64_t selective_decode_hotpath_allocations_ = 0;
    uint64_t selective_decode_workspace_reallocations_ = 0;
    double tokens_per_second_sum_ = 0.0;
    double first_token_ms_sum_ = 0.0;
    int64_t latency_samples_ = 0;
};

} // namespace llm_engine
