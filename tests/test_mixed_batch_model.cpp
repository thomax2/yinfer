#include <gtest/gtest.h>

#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "llm_engine/engine/llm_engine.h"
#include "llm_engine/memory/memory_pool.h"
#include "model.h"

namespace llm_engine {
namespace {

class ScopedEnv {
public:
    ScopedEnv(const char* name, const char* value) : name_(name) {
        const char* old = std::getenv(name);
        if (old) {
            had_old_ = true;
            old_value_ = old;
        }
#if defined(_WIN32)
        _putenv_s(name, value);
#else
        setenv(name, value, 1);
#endif
    }

    ~ScopedEnv() {
#if defined(_WIN32)
        _putenv_s(name_.c_str(), had_old_ ? old_value_.c_str() : "");
#else
        if (had_old_) setenv(name_.c_str(), old_value_.c_str(), 1);
        else unsetenv(name_.c_str());
#endif
    }

private:
    std::string name_;
    std::string old_value_;
    bool had_old_ = false;
};

class ScopedMemoryPool {
public:
    ScopedMemoryPool() {
        if (g_memory_pool == nullptr) {
            owned_pool_ = std::make_unique<MemoryPool>(32ULL * 1024 * 1024);
            g_memory_pool = owned_pool_.get();
        }
    }

    ~ScopedMemoryPool() {
        if (owned_pool_) {
            g_memory_pool = nullptr;
        }
    }

private:
    std::unique_ptr<MemoryPool> owned_pool_;
};

QwenConfig tiny_config() {
    QwenConfig config;
    config.num_layers = 1;
    config.hidden_dim = 8;
    config.intermediate_size = 16;
    config.num_q_heads = 1;
    config.num_kv_heads = 1;
    config.head_dim = 8;
    config.vocab_size = 32;
    config.max_seq_len = 16;
    return config;
}

TEST(MixedBatchDescriptorTest, DecodeAndPrefillUseOneContiguousTokenMajorLayout) {
    SequenceState decode_sequence;
    decode_sequence.history_pos = 100;
    SequenceState prefill_sequence;
    prefill_sequence.history_pos = 32;

    QwenModel::MixedBatchItem decode{
        QwenModel::MixedBatchItemKind::DECODE,
        &decode_sequence,
        0,
        1,
        100,
        true};
    QwenModel::MixedBatchItem prefill{
        QwenModel::MixedBatchItemKind::PREFILL,
        &prefill_sequence,
        1,
        63,
        32,
        false};

    EXPECT_EQ(decode.row_begin + decode.row_count, prefill.row_begin);
    EXPECT_EQ(prefill.row_begin + prefill.row_count, 64);
    EXPECT_TRUE(decode.requires_logits);
    EXPECT_FALSE(prefill.requires_logits);
}

TEST(MixedBatchDescriptorTest, OnlyFinalPrefillChunkRequiresLogits) {
    SequenceState sequence;
    QwenModel::MixedBatchItem middle{
        QwenModel::MixedBatchItemKind::PREFILL, &sequence, 0, 32, 0, false};
    QwenModel::MixedBatchItem final{
        QwenModel::MixedBatchItemKind::PREFILL, &sequence, 0, 7, 32, true};

    EXPECT_FALSE(middle.requires_logits);
    EXPECT_TRUE(final.requires_logits);
}

TEST(MixedBatchPlanInitializationTest, SchedulerBuildsAndCompilesPlanBeforeFirstRun) {
    ScopedMemoryPool memory_pool;
    ScopedEnv scheduler("LLM_ENABLE_SCHEDULER", "1");
    ScopedEnv token_budget("LLM_MAX_BATCHED_TOKENS", "8");
    QwenModel model(tiny_config());
    ASSERT_EQ(model.fine_mixed_batch_plan, nullptr);
    ASSERT_EQ(model.mixed_batch_workspace, nullptr);

    LLMEngine engine(model);

    ASSERT_TRUE(engine.scheduler_enabled());
    ASSERT_NE(model.fine_mixed_batch_plan, nullptr);
    EXPECT_EQ(model.fine_mixed_batch_plan->memory_stats().node_count, 7u);
    EXPECT_EQ(model.fine_mixed_batch_plan->spec().row_capacity, 8);
    EXPECT_EQ(model.fine_mixed_batch_plan->spec().mode,
              ExecutionMode::MIXED_SELECTIVE_BATCH);
    EXPECT_EQ(model.mixed_batch_row_capacity, 8);
    EXPECT_EQ(model.mixed_batch_workspace_max_rows, 8);
    EXPECT_NE(model.mixed_batch_workspace, nullptr);
    EXPECT_GT(model.mixed_batch_workspace_bytes, 0u);
    EXPECT_EQ(model.mixed_batch_workspace_reallocations, 1u);

    // 相同容量的初始化幂等，不能替换计划或重新申请 Workspace。
    ExecutablePlan* compiled = model.fine_mixed_batch_plan.get();
    void* workspace = model.mixed_batch_workspace;
    model.build_mixed_batch_plan(8);
    EXPECT_EQ(model.fine_mixed_batch_plan.get(), compiled);
    EXPECT_EQ(model.mixed_batch_workspace, workspace);
    EXPECT_EQ(model.mixed_batch_workspace_reallocations, 1u);

    // 初始化后不允许悄悄切换容量；需要新容量时应重建整个 Model/Engine。
    EXPECT_THROW(model.build_mixed_batch_plan(16), std::logic_error);

    // 运行阶段只做边界检查：超过固定容量时在写 KV 前失败，且不得扩容。
    KVCacheManager kv_manager(
        *model.kv_cache,
        model.kv_cache->get_max_seq_len(),
        model.kv_cache->block_size(),
        model.kv_cache->allocated_blocks());
    SequenceState sequence;
    std::vector<int> token_ids(9, 0);
    std::vector<QwenModel::MixedBatchItem> items{
        {QwenModel::MixedBatchItemKind::PREFILL, &sequence, 0, 9, 0, false}};
    std::vector<QwenModel::MixedBatchOutput> outputs;
    QwenModel::MixedBatchStats stats;

    EXPECT_FALSE(model.run_mixed_batch_for_sequences(
        token_ids, items, kv_manager, nullptr, &outputs, &stats));
    ASSERT_EQ(outputs.size(), 1u);
    EXPECT_EQ(outputs[0].error_message,
              "mixed batch rows exceed compiled capacity");
    EXPECT_FALSE(stats.state_modified);
    EXPECT_EQ(stats.mixed_batch_workspace_reallocations, 0u);
    EXPECT_EQ(model.mixed_batch_workspace, workspace);
    EXPECT_EQ(model.mixed_batch_workspace_reallocations, 1u);
}

TEST(MixedBatchPlanInitializationTest, RejectsTooManyLogitRowsBeforeKvMutation) {
    ScopedMemoryPool memory_pool;
    ScopedEnv scheduler("LLM_ENABLE_SCHEDULER", "1");
    ScopedEnv token_budget("LLM_MAX_BATCHED_TOKENS", "16");
    QwenModel model(tiny_config());
    LLMEngine engine(model);

    ASSERT_EQ(model.fine_mixed_batch_plan->spec().row_capacity, 16);
    KVCacheManager kv_manager(
        *model.kv_cache,
        model.kv_cache->get_max_seq_len(),
        model.kv_cache->block_size(),
        model.kv_cache->allocated_blocks());

    constexpr int rows = arm_neon::GPTQ_BATCH_ARGMAX_MAX_ROWS + 1;
    std::vector<SequenceState> sequences((size_t)rows);
    std::vector<int> token_ids((size_t)rows, 0);
    std::vector<QwenModel::MixedBatchItem> items;
    items.reserve((size_t)rows);
    for (int row = 0; row < rows; ++row) {
        items.push_back({
            QwenModel::MixedBatchItemKind::DECODE,
            &sequences[(size_t)row], row, 1, 0, true});
    }

    std::vector<QwenModel::MixedBatchOutput> outputs;
    QwenModel::MixedBatchStats stats;
    EXPECT_FALSE(model.run_mixed_batch_for_sequences(
        token_ids, items, kv_manager, nullptr, &outputs, &stats));
    ASSERT_EQ(outputs.size(), (size_t)rows);
    EXPECT_EQ(outputs[0].error_message,
              "mixed batch logit rows exceed batch argmax capacity");
    EXPECT_FALSE(stats.state_modified);
    for (const SequenceState& sequence : sequences) {
        EXPECT_EQ(sequence.history_pos, 0);
        EXPECT_EQ(sequence.max_written_pos, -1);
        EXPECT_TRUE(sequence.block_table.empty());
    }
}

} // namespace
} // namespace llm_engine
