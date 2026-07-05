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
       << ",\"avg_tokens_per_second\":" << avg_tps
       << ",\"avg_first_token_ms\":" << avg_first
       << '}';
    return os.str();
}

} // namespace llm_engine
