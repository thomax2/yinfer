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
    if (metrics.continuous_batching_enabled) {
        continuous_batching_enabled_ = true;
    }
    decode_batch_steps_total_ += metrics.decode_batch_steps;
    decode_batch_size_sum_ += metrics.decode_batch_size_sum;
    if (metrics.decode_batch_size_max > decode_batch_size_max_) {
        decode_batch_size_max_ = metrics.decode_batch_size_max;
    }
    prefill_chunk_steps_total_ += metrics.prefill_chunk_steps;
    scheduler_v2_decode_steps_total_ += metrics.scheduler_v2_decode_steps;
    scheduler_v2_prefill_steps_total_ += metrics.scheduler_v2_prefill_steps;
    active_decode_batch_size_ = metrics.active_decode_batch_size_at_finish;
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
    if (metrics.prefill_batching_enabled) {
        prefill_batching_enabled_ = true;
    }
    prefill_microbatch_steps_total_ += metrics.prefill_microbatch_steps;
    prefill_microbatch_size_sum_ += metrics.prefill_microbatch_size_sum;
    if (metrics.prefill_microbatch_size_max > prefill_microbatch_size_max_) {
        prefill_microbatch_size_max_ = metrics.prefill_microbatch_size_max;
    }
    prefill_full_chunk_steps_total_ += metrics.prefill_full_chunk_steps;
    prefill_tail_chunk_steps_total_ += metrics.prefill_tail_chunk_steps;
    prefill_microbatch_tokens_total_ += metrics.prefill_microbatch_tokens_total;
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
    double avg_prefill_microbatch = prefill_microbatch_steps_total_ > 0
        ? static_cast<double>(prefill_microbatch_size_sum_) /
              static_cast<double>(prefill_microbatch_steps_total_)
        : 0.0;
    double avg_selective_decode_batch = selective_decode_steps_total_ > 0
        ? static_cast<double>(selective_decode_size_sum_) /
              static_cast<double>(selective_decode_steps_total_)
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
       << ",\"continuous_batching_enabled\":" << (continuous_batching_enabled_ ? "true" : "false")
       << ",\"decode_batch_steps_total\":" << decode_batch_steps_total_
       << ",\"decode_batch_size_sum\":" << decode_batch_size_sum_
       << ",\"decode_batch_size_max\":" << decode_batch_size_max_
       << ",\"avg_decode_batch_size\":" << avg_decode_batch
       << ",\"prefill_chunk_steps_total\":" << prefill_chunk_steps_total_
       << ",\"active_decode_batch_size\":" << active_decode_batch_size_
       << ",\"scheduler_v2_decode_steps\":" << scheduler_v2_decode_steps_total_
       << ",\"scheduler_v2_prefill_steps\":" << scheduler_v2_prefill_steps_total_
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
       << ",\"prefill_batching_enabled\":" << (prefill_batching_enabled_ ? "true" : "false")
       << ",\"prefill_microbatch_steps_total\":" << prefill_microbatch_steps_total_
       << ",\"prefill_microbatch_size_sum\":" << prefill_microbatch_size_sum_
       << ",\"prefill_microbatch_size_max\":" << prefill_microbatch_size_max_
       << ",\"avg_prefill_microbatch_size\":" << avg_prefill_microbatch
       << ",\"prefill_full_chunk_steps_total\":" << prefill_full_chunk_steps_total_
       << ",\"prefill_tail_chunk_steps_total\":" << prefill_tail_chunk_steps_total_
       << ",\"prefill_microbatch_tokens_total\":" << prefill_microbatch_tokens_total_
       << ",\"avg_tokens_per_second\":" << avg_tps
       << ",\"avg_first_token_ms\":" << avg_first
       << '}';
    return os.str();
}

} // namespace llm_engine
