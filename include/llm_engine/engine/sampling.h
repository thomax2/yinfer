#pragma once

#include <cstdint>
#include <vector>

namespace llm_engine {

struct SamplingParams {
    int max_new_tokens = 512;
    float temperature = 0.0f;
    int top_k = 0;
    float top_p = 1.0f;
    bool greedy = true;
    uint64_t seed = 0;
    bool has_seed = false;
    std::vector<int> stop_token_ids;
};

struct SamplingRuntimeStats {
    int sampled_tokens = 0;
    int greedy_tokens = 0;
    double sampling_ms = 0.0;
};

inline bool sampling_enabled(const SamplingParams& sampling) {
    return !sampling.greedy || sampling.temperature > 0.0f;
}

} // namespace llm_engine
