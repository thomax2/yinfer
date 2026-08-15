#include "llm_engine/server/server_metrics.h"

#include <sstream>

namespace llm_engine {

void ServerMetrics::connection_opened() {
    std::lock_guard<std::mutex> lk(mu_);
    active_connections_++;
}

void ServerMetrics::connection_closed() {
    std::lock_guard<std::mutex> lk(mu_);
    if (active_connections_ > 0) {
        active_connections_--;
    }
}

void ServerMetrics::request_started(int prompt_tokens) {
    std::lock_guard<std::mutex> lk(mu_);
    requests_total_++;
    prompt_tokens_total_ += prompt_tokens;
    active_requests_++;
}

void ServerMetrics::request_finished(const RequestMetrics& metrics) {
    std::lock_guard<std::mutex> lk(mu_);
    requests_finished_++;
    if (active_requests_ > 0) {
        active_requests_--;
    }
    tokens_generated_total_ += metrics.generated_tokens;
    prefix_cached_tokens_total_ += metrics.cached_prefix_tokens;
    prefix_cached_blocks_total_ += metrics.cached_prefix_blocks;
    if (metrics.cached_prefix_tokens > 0 || metrics.cached_prefix_blocks > 0) {
        prefix_hit_requests_++;
    }
    real_batch_prefill_chunks_total_ += metrics.real_batch_prefill_chunks;
    paged_attention_calls_ += metrics.paged_attention_calls;
    paged_attention_fallbacks_ += metrics.paged_attention_fallbacks;
    if (metrics.scheduler_enabled) {
        scheduler_enabled_ = true;
    }
    decode_batch_steps_total_ += metrics.decode_batch_steps;
    decode_batch_size_sum_ += metrics.decode_batch_size_sum;
    if (metrics.decode_batch_size_max > decode_batch_size_max_) {
        decode_batch_size_max_ = metrics.decode_batch_size_max;
    }
    prefill_chunk_steps_total_ += metrics.prefill_chunk_steps;
    scheduler_decode_steps_total_ += metrics.scheduler_decode_steps;
    scheduler_prefill_steps_total_ += metrics.scheduler_prefill_steps;
    decode_queue_size_ = metrics.decode_queue_size_at_finish;
    if (metrics.mixed_batch_enabled) mixed_batch_enabled_ = true;
    mixed_batch_steps_total_ += metrics.mixed_batch_steps;
    mixed_batch_total_rows_sum_ += metrics.mixed_batch_total_rows_sum;
    if (metrics.mixed_batch_total_rows_max > mixed_batch_total_rows_max_) {
        mixed_batch_total_rows_max_ = metrics.mixed_batch_total_rows_max;
    }
    mixed_batch_decode_rows_total_ += metrics.mixed_batch_decode_rows;
    mixed_batch_prefill_rows_total_ += metrics.mixed_batch_prefill_rows;
    mixed_batch_attention_segments_total_ += metrics.mixed_batch_attention_segments;
    mixed_batch_lm_head_rows_total_ += metrics.mixed_batch_lm_head_rows;
    mixed_batch_fallbacks_total_ += metrics.mixed_batch_fallbacks;
    mixed_batch_model_ms_total_ += metrics.mixed_batch_model_ms;
    if (metrics.selective_decode_enabled) {
        selective_decode_enabled_ = true;
    }
    selective_decode_steps_total_ += metrics.selective_decode_steps;
    selective_decode_size_sum_ += metrics.selective_decode_size_sum;
    if (metrics.selective_decode_size_max > selective_decode_size_max_) {
        selective_decode_size_max_ = metrics.selective_decode_size_max;
    }
    selective_decode_linear_batch_rows_total_ += metrics.selective_decode_linear_batch_rows;
    selective_decode_attention_per_sequence_calls_total_ +=
        metrics.selective_decode_attention_per_sequence_calls;
    selective_decode_lm_head_rows_total_ += metrics.selective_decode_lm_head_rows;
    selective_decode_fallbacks_total_ += metrics.selective_decode_fallbacks;
    selective_decode_model_ms_total_ += metrics.selective_decode_model_ms;
    gptq_batch_kernel_calls_ += metrics.gptq_batch_kernel_calls;
    gptq_batch_rows_total_ += metrics.gptq_batch_rows_total;
    gptq_batch_output_panel_tasks_ += metrics.gptq_batch_output_panel_tasks;
    gptq_batch_row_gemv_fallbacks_ += metrics.gptq_batch_row_gemv_fallbacks;
    gptq_batch_weight_vector_loads_ += metrics.gptq_batch_weight_vector_loads;
    gptq_batch_dequant_vector_ops_ += metrics.gptq_batch_dequant_vector_ops;
    gptq_batch_argmax_calls_ += metrics.gptq_batch_argmax_calls;
    gptq_batch_argmax_rows_ += metrics.gptq_batch_argmax_rows;
    gptq_batch_full_logits_elements_written_ += metrics.gptq_batch_full_logits_elements_written;
    gptq_batch_compare_mismatches_ += metrics.gptq_batch_compare_mismatches;
    selective_decode_hotpath_allocations_ += metrics.selective_decode_hotpath_allocations;
    selective_decode_workspace_reallocations_ +=
        metrics.selective_decode_workspace_reallocations;
    if (metrics.tokens_per_second > 0.0) {
        tokens_per_second_sum_ += metrics.tokens_per_second;
    }
    if (metrics.first_token_ms > 0.0) {
        first_token_ms_sum_ += metrics.first_token_ms;
    }
    latency_samples_++;
}

void ServerMetrics::request_failed() {
    std::lock_guard<std::mutex> lk(mu_);
    requests_failed_++;
    if (active_requests_ > 0) {
        active_requests_--;
    }
}

void ServerMetrics::request_aborted() {
    std::lock_guard<std::mutex> lk(mu_);
    requests_aborted_++;
    if (active_requests_ > 0) {
        active_requests_--;
    }
}

void ServerMetrics::set_pending_requests(int pending) {
    std::lock_guard<std::mutex> lk(mu_);
    pending_requests_ = pending;
}

void ServerMetrics::set_active_requests(int active) {
    std::lock_guard<std::mutex> lk(mu_);
    active_requests_ = active;
}

int ServerMetrics::active_connections() const {
    std::lock_guard<std::mutex> lk(mu_);
    return active_connections_;
}

int ServerMetrics::active_requests() const {
    std::lock_guard<std::mutex> lk(mu_);
    return active_requests_;
}

int ServerMetrics::pending_requests() const {
    std::lock_guard<std::mutex> lk(mu_);
    return pending_requests_;
}

std::string ServerMetrics::to_json() const {
    std::lock_guard<std::mutex> lk(mu_);
    double avg_tps = latency_samples_ > 0
        ? tokens_per_second_sum_ / static_cast<double>(latency_samples_)
        : 0.0;
    double avg_first = latency_samples_ > 0
        ? first_token_ms_sum_ / static_cast<double>(latency_samples_)
        : 0.0;
    double avg_decode_batch = decode_batch_steps_total_ > 0
        ? static_cast<double>(decode_batch_size_sum_) /
              static_cast<double>(decode_batch_steps_total_)
        : 0.0;
    double avg_selective_decode_batch = selective_decode_steps_total_ > 0
        ? static_cast<double>(selective_decode_size_sum_) /
              static_cast<double>(selective_decode_steps_total_)
        : 0.0;
    double avg_mixed_batch_rows = mixed_batch_steps_total_ > 0
        ? static_cast<double>(mixed_batch_total_rows_sum_) /
              static_cast<double>(mixed_batch_steps_total_)
        : 0.0;
    std::ostringstream os;
    os << '{'
       << "\"requests_total\":" << requests_total_
       << ",\"requests_finished\":" << requests_finished_
       << ",\"requests_failed\":" << requests_failed_
       << ",\"requests_aborted\":" << requests_aborted_
       << ",\"tokens_generated_total\":" << tokens_generated_total_
       << ",\"prompt_tokens_total\":" << prompt_tokens_total_
       << ",\"active_requests\":" << active_requests_
       << ",\"pending_requests\":" << pending_requests_
       << ",\"active_connections\":" << active_connections_
       << ",\"prefix_cached_tokens_total\":" << prefix_cached_tokens_total_
       << ",\"prefix_cached_blocks_total\":" << prefix_cached_blocks_total_
       << ",\"prefix_hit_requests\":" << prefix_hit_requests_
       << ",\"real_batch_prefill_chunks_total\":" << real_batch_prefill_chunks_total_
       << ",\"paged_attention_calls\":" << paged_attention_calls_
       << ",\"paged_attention_fallbacks\":" << paged_attention_fallbacks_
       << ",\"scheduler_enabled\":" << (scheduler_enabled_ ? "true" : "false")
       << ",\"decode_batch_steps_total\":" << decode_batch_steps_total_
       << ",\"decode_batch_size_sum\":" << decode_batch_size_sum_
       << ",\"decode_batch_size_max\":" << decode_batch_size_max_
       << ",\"avg_decode_batch_size\":" << avg_decode_batch
       << ",\"prefill_chunk_steps_total\":" << prefill_chunk_steps_total_
       << ",\"decode_queue_size\":" << decode_queue_size_
       << ",\"scheduler_decode_steps\":" << scheduler_decode_steps_total_
       << ",\"scheduler_prefill_steps\":" << scheduler_prefill_steps_total_
       << ",\"mixed_batch_enabled\":" << (mixed_batch_enabled_ ? "true" : "false")
       << ",\"mixed_batch_steps_total\":" << mixed_batch_steps_total_
       << ",\"mixed_batch_total_rows_sum\":" << mixed_batch_total_rows_sum_
       << ",\"mixed_batch_total_rows_max\":" << mixed_batch_total_rows_max_
       << ",\"avg_mixed_batch_rows\":" << avg_mixed_batch_rows
       << ",\"mixed_batch_decode_rows_total\":" << mixed_batch_decode_rows_total_
       << ",\"mixed_batch_prefill_rows_total\":" << mixed_batch_prefill_rows_total_
       << ",\"mixed_batch_attention_segments_total\":"
       << mixed_batch_attention_segments_total_
       << ",\"mixed_batch_lm_head_rows_total\":" << mixed_batch_lm_head_rows_total_
       << ",\"mixed_batch_fallbacks_total\":" << mixed_batch_fallbacks_total_
       << ",\"mixed_batch_model_ms_total\":" << mixed_batch_model_ms_total_
       << ",\"selective_decode_enabled\":" << (selective_decode_enabled_ ? "true" : "false")
       << ",\"selective_decode_steps_total\":" << selective_decode_steps_total_
       << ",\"selective_decode_size_sum\":" << selective_decode_size_sum_
       << ",\"selective_decode_size_max\":" << selective_decode_size_max_
       << ",\"avg_selective_decode_size\":" << avg_selective_decode_batch
       << ",\"selective_decode_linear_batch_rows_total\":"
       << selective_decode_linear_batch_rows_total_
       << ",\"selective_decode_attention_per_sequence_calls_total\":"
       << selective_decode_attention_per_sequence_calls_total_
       << ",\"selective_decode_lm_head_rows_total\":" << selective_decode_lm_head_rows_total_
       << ",\"selective_decode_fallbacks_total\":" << selective_decode_fallbacks_total_
       << ",\"selective_decode_model_ms_total\":" << selective_decode_model_ms_total_
       << ",\"gptq_batch_kernel_calls\":" << gptq_batch_kernel_calls_
       << ",\"gptq_batch_rows_total\":" << gptq_batch_rows_total_
       << ",\"gptq_batch_output_panel_tasks\":" << gptq_batch_output_panel_tasks_
       << ",\"gptq_batch_row_gemv_fallbacks\":" << gptq_batch_row_gemv_fallbacks_
       << ",\"gptq_batch_weight_vector_loads\":" << gptq_batch_weight_vector_loads_
       << ",\"gptq_batch_dequant_vector_ops\":" << gptq_batch_dequant_vector_ops_
       << ",\"gptq_batch_argmax_calls\":" << gptq_batch_argmax_calls_
       << ",\"gptq_batch_argmax_rows\":" << gptq_batch_argmax_rows_
       << ",\"gptq_batch_full_logits_elements_written\":"
       << gptq_batch_full_logits_elements_written_
       << ",\"gptq_batch_compare_mismatches\":" << gptq_batch_compare_mismatches_
       << ",\"selective_decode_hotpath_allocations\":"
       << selective_decode_hotpath_allocations_
       << ",\"selective_decode_workspace_reallocations\":"
       << selective_decode_workspace_reallocations_
       << ",\"avg_tokens_per_second\":" << avg_tps
       << ",\"avg_first_token_ms\":" << avg_first
       << '}';
    return os.str();
}

} // namespace llm_engine
