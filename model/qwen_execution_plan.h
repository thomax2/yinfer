#pragma once

#include <memory>
#include <vector>

#include "llm_engine/graph/executable_plan.h"
#include "llm_engine/graph/graph.h"

namespace llm_engine {

struct QwenPlanRuntime {
    KVCache* kv_cache = nullptr;
    int current_pos = 0;
    int start_pos = 0;
    const int* positions = nullptr;
    const fp16_t* cos = nullptr;
    const fp16_t* sin = nullptr;
};

struct QwenPlanArtifacts {
    std::unique_ptr<ExecutablePlan> plan;
    PlanValueId hidden_input = kInvalidPlanValue;
    PlanValueId norm_output = kInvalidPlanValue;
    PlanValueId argmax_output = kInvalidPlanValue;
    size_t coarse_workspace_bytes = 0;
};

QwenPlanArtifacts build_qwen_single_decode_plan(
    const std::vector<QwenBlockWeights>& layers,
    const Tensor& final_norm_weight,
    const arm_neon::GPTQInt8Weight& lm_head,
    const arm_neon::AttentionConfig& attention_config,
    const arm_neon::FFNConfig& ffn_config,
    int max_seq_len,
    float rms_norm_eps);

QwenPlanArtifacts build_qwen_prefill_plan(
    const std::vector<QwenBlockWeights>& layers,
    const Tensor& final_norm_weight,
    const arm_neon::GPTQInt8Weight& lm_head,
    const arm_neon::AttentionConfig& attention_config,
    const arm_neon::FFNConfig& ffn_config,
    int max_seq_len,
    int row_capacity,
    float rms_norm_eps);

} // namespace llm_engine
