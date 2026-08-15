#include <gtest/gtest.h>

#include "llm_engine/engine/llm_engine.h"
#include "llm_engine/engine/mixed_batch.h"

namespace llm_engine {
namespace {

TEST(MixedBatchBudgetTest, DecodeRowsTakePriorityAndPrefillFillsTheRemainder) {
    MixedBatchBudget budget = plan_mixed_batch_budget(64, 8, 1000, 0);
    EXPECT_EQ(budget.decode_rows, 8);
    EXPECT_EQ(budget.prefill_rows, 56);
    EXPECT_EQ(budget.total_rows, 64);
}

TEST(MixedBatchBudgetTest, DecodeCanConsumeTheWholeBudget) {
    MixedBatchBudget budget = plan_mixed_batch_budget(64, 80, 1000, 0);
    EXPECT_EQ(budget.decode_rows, 64);
    EXPECT_EQ(budget.prefill_rows, 0);
    EXPECT_EQ(budget.total_rows, 64);
}

TEST(MixedBatchBudgetTest, PrefillLatencyCapAppliesOnlyWhenDecodeIsActive) {
    MixedBatchBudget with_decode = plan_mixed_batch_budget(64, 8, 1000, 16);
    EXPECT_EQ(with_decode.decode_rows, 8);
    EXPECT_EQ(with_decode.prefill_rows, 16);

    MixedBatchBudget prefill_only = plan_mixed_batch_budget(64, 0, 1000, 16);
    EXPECT_EQ(prefill_only.prefill_rows, 64);
}

TEST(MixedBatchBudgetTest, EmptyOrInvalidInputsNeverProduceNegativeRows) {
    MixedBatchBudget budget = plan_mixed_batch_budget(0, -1, -1, -1);
    EXPECT_EQ(budget.decode_rows, 0);
    EXPECT_EQ(budget.prefill_rows, 0);
    EXPECT_EQ(budget.total_rows, 0);
}

TEST(SchedulerQueueStateTest, DecodeHasOneQueueMembershipFlag) {
    RequestState request;
    EXPECT_EQ(request.status, RequestStatus::WAITING);
    EXPECT_FALSE(request.queued_decode);

    // Prefill 完成后只需要切换业务状态并加入唯一的 Decode 队列。
    request.status = RequestStatus::RUNNING_DECODE;
    request.queued_decode = true;
    EXPECT_EQ(request.status, RequestStatus::RUNNING_DECODE);
    EXPECT_TRUE(request.queued_decode);
}

} // namespace
} // namespace llm_engine
