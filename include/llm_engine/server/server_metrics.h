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
    bool continuous_batching_enabled_ = false;
    int64_t decode_batch_steps_total_ = 0;
    int64_t decode_batch_size_sum_ = 0;
    int decode_batch_size_max_ = 0;
    int64_t prefill_chunk_steps_total_ = 0;
    int64_t scheduler_v2_decode_steps_total_ = 0;
    int64_t scheduler_v2_prefill_steps_total_ = 0;
    int active_decode_batch_size_ = 0;
    bool prefill_batching_enabled_ = false;
    int64_t prefill_microbatch_steps_total_ = 0;
    int64_t prefill_microbatch_size_sum_ = 0;
    int prefill_microbatch_size_max_ = 0;
    int64_t prefill_full_chunk_steps_total_ = 0;
    int64_t prefill_tail_chunk_steps_total_ = 0;
    int64_t prefill_microbatch_tokens_total_ = 0;
    double tokens_per_second_sum_ = 0.0;
    double first_token_ms_sum_ = 0.0;
    int64_t latency_samples_ = 0;
};

} // namespace llm_engine
