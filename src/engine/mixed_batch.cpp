#include "llm_engine/engine/mixed_batch.h"

#include <algorithm>

namespace llm_engine {

MixedBatchBudget plan_mixed_batch_budget(
    int max_batched_tokens,
    int ready_decode_rows,
    int available_prefill_rows,
    int max_prefill_rows_with_decode) {
    MixedBatchBudget result;
    if (max_batched_tokens <= 0) {
        return result;
    }

    const int safe_decode_rows = std::max(0, ready_decode_rows);
    const int safe_prefill_rows = std::max(0, available_prefill_rows);

    // Decode 一行代表一个请求本轮要消费的 token。先占满 Decode，保证生成阶段优先。
    result.decode_rows = std::min(safe_decode_rows, max_batched_tokens);
    int remaining = max_batched_tokens - result.decode_rows;

    // 可选的 Prefill 上限用于控制 ITL。配置为 0 时，把剩余 token budget 全部填满。
    if (result.decode_rows > 0 && max_prefill_rows_with_decode > 0) {
        remaining = std::min(remaining, max_prefill_rows_with_decode);
    }
    result.prefill_rows = std::min(safe_prefill_rows, remaining);
    result.total_rows = result.decode_rows + result.prefill_rows;
    return result;
}

} // namespace llm_engine
