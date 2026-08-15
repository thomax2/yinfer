#pragma once

namespace llm_engine {

// 一轮混合批次的 token 预算分配结果。
// Decode 永远先占行，剩余行才交给 Chunked Prefill。
struct MixedBatchBudget {
    int decode_rows = 0;
    int prefill_rows = 0;
    int total_rows = 0;
};

// 这是一个纯调度函数，不访问模型和 KV Cache，便于在非 ARM 主机上验证调度规则。
// max_prefill_rows_with_decode <= 0 表示 Decode 活跃时不额外限制 Prefill 行数。
MixedBatchBudget plan_mixed_batch_budget(
    int max_batched_tokens,
    int ready_decode_rows,
    int available_prefill_rows,
    int max_prefill_rows_with_decode);

} // namespace llm_engine
