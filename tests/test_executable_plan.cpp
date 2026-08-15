#include <gtest/gtest.h>

#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>

#include "llm_engine/graph/executable_plan.h"
#include "llm_engine/memory/memory_pool.h"
#include "qwen_execution_plan.h"

namespace llm_engine {
namespace {

class ExecutablePlanTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (g_memory_pool == nullptr) {
            owned_pool_ = std::make_unique<MemoryPool>(32 * 1024 * 1024);
            g_memory_pool = owned_pool_.get();
        }
    }

    void TearDown() override {
        if (owned_pool_) g_memory_pool = nullptr;
    }

    std::unique_ptr<MemoryPool> owned_pool_;
};

TEST_F(ExecutablePlanTest, StableTopologicalSortRunsLoweredNodes) {
    ExecutablePlanBuilder builder({ExecutionMode::SINGLE_DECODE, 1});
    PlanValueId input = builder.add_value({"input", sizeof(int), 64, true});
    PlanValueId middle = builder.add_value({"middle", sizeof(int)});
    PlanValueId output = builder.add_value({"output", sizeof(int), 64, true});

    // Consumer is deliberately inserted before its producer.
    builder.add_node(std::make_unique<CallbackKernelNode>(
        "consumer", std::vector<PlanValueId>{middle}, std::vector<PlanValueId>{output},
        [=](ExecutionContext& context) {
            *context.ptr<int>(output) = *context.ptr<int>(middle) + 1;
            return Status::SUCCESS;
        }));
    builder.add_node(std::make_unique<CallbackKernelNode>(
        "producer", std::vector<PlanValueId>{input}, std::vector<PlanValueId>{middle},
        [=](ExecutionContext& context) {
            *context.ptr<int>(middle) = *context.ptr<int>(input) * 2;
            return Status::SUCCESS;
        }));

    ExecutablePlan plan = builder.compile();
    ASSERT_EQ(plan.order().size(), 2u);
    EXPECT_EQ(plan.order()[0]->name(), "producer");
    EXPECT_EQ(plan.order()[1]->name(), "consumer");

    int in = 7;
    int out = 0;
    ExecutionContext context;
    context.bind_external(input, &in);
    context.bind_external(output, &out);
    ExecutionWorkspace workspace;
    ASSERT_EQ(plan.run(context, workspace), Status::SUCCESS);
    EXPECT_EQ(out, 15);
}

TEST(ExecutablePlanCompileTest, RejectsMultipleProducers) {
    ExecutablePlanBuilder builder({ExecutionMode::SINGLE_DECODE, 1});
    PlanValueId input = builder.add_value({"input", 4, 64, true});
    PlanValueId value = builder.add_value({"value", 4});
    auto noop = [](ExecutionContext&) { return Status::SUCCESS; };
    builder.add_node(std::make_unique<CallbackKernelNode>(
        "first", std::vector<PlanValueId>{input}, std::vector<PlanValueId>{value}, noop));
    builder.add_node(std::make_unique<CallbackKernelNode>(
        "second", std::vector<PlanValueId>{input}, std::vector<PlanValueId>{value}, noop));
    EXPECT_THROW((void)builder.compile(), std::runtime_error);
}

TEST(ExecutablePlanCompileTest, RejectsCycle) {
    ExecutablePlanBuilder builder({ExecutionMode::SINGLE_DECODE, 1});
    PlanValueId a = builder.add_value({"a", 4});
    PlanValueId b = builder.add_value({"b", 4});
    auto noop = [](ExecutionContext&) { return Status::SUCCESS; };
    builder.add_node(std::make_unique<CallbackKernelNode>(
        "a_to_b", std::vector<PlanValueId>{a}, std::vector<PlanValueId>{b}, noop));
    builder.add_node(std::make_unique<CallbackKernelNode>(
        "b_to_a", std::vector<PlanValueId>{b}, std::vector<PlanValueId>{a}, noop));
    EXPECT_THROW((void)builder.compile(), std::runtime_error);
}

TEST(ExecutablePlanCompileTest, RejectsDirectInputOutputSelfAlias) {
    ExecutablePlanBuilder builder({ExecutionMode::SINGLE_DECODE, 1});
    PlanValueId value = builder.add_value({"value", 4});
    builder.add_node(std::make_unique<CallbackKernelNode>(
        "self", std::vector<PlanValueId>{value}, std::vector<PlanValueId>{value},
        [](ExecutionContext&) { return Status::SUCCESS; }));
    EXPECT_THROW((void)builder.compile(), std::runtime_error);
}

TEST_F(ExecutablePlanTest, LivenessReusesArenaAndWorkspaceOnlyGrows) {
    ExecutablePlanBuilder builder({ExecutionMode::PREFILL, 8});
    PlanValueId input = builder.add_value({"input", 64, 64, true});
    PlanValueId v1 = builder.add_value({"v1", 128});
    PlanValueId v2 = builder.add_value({"v2", 256});
    PlanValueId v3 = builder.add_value({"v3", 128});
    PlanValueId output = builder.add_value({"output", 64, 64, true});

    auto write = [](PlanValueId destination) {
        return [destination](ExecutionContext& context) {
            std::memset(context.value_data(destination), 1, 1);
            return Status::SUCCESS;
        };
    };
    builder.add_node(std::make_unique<CallbackKernelNode>(
        "n1", std::vector<PlanValueId>{input}, std::vector<PlanValueId>{v1}, write(v1)));
    builder.add_node(std::make_unique<CallbackKernelNode>(
        "n2", std::vector<PlanValueId>{v1}, std::vector<PlanValueId>{v2}, write(v2)));
    builder.add_node(std::make_unique<CallbackKernelNode>(
        "n3", std::vector<PlanValueId>{v2}, std::vector<PlanValueId>{v3}, write(v3)));
    builder.add_node(std::make_unique<CallbackKernelNode>(
        "n4", std::vector<PlanValueId>{v3}, std::vector<PlanValueId>{output},
        [](ExecutionContext&) { return Status::SUCCESS; }));

    ExecutablePlan plan = builder.compile();
    EXPECT_LT(plan.memory_stats().arena_bytes,
              plan.memory_stats().sum_internal_value_bytes);
    EXPECT_EQ(plan.memory_stats().peak_live_value_bytes, 384u);

    char in[64]{};
    char out[64]{};
    ExecutionContext context;
    context.actual_rows = 4;
    context.bind_external(input, in);
    context.bind_external(output, out);
    ExecutionWorkspace workspace;
    ASSERT_EQ(plan.run(context, workspace), Status::SUCCESS);
    size_t first_capacity = workspace.size();
    ASSERT_EQ(plan.run(context, workspace), Status::SUCCESS);
    EXPECT_EQ(workspace.size(), first_capacity);
    EXPECT_EQ(workspace.reallocations(), 1u);

    context.actual_rows = 9;
    EXPECT_EQ(plan.run(context, workspace), Status::INVALID_ARGUMENT);
}

TEST(QwenPlanLayoutTest, FineDecodeEliminatesThe64MiBBlockWorkspace) {
    constexpr int layers_count = 28;
    constexpr int hidden = 1536;
    constexpr int intermediate = 8960;
    std::vector<QwenBlockWeights> layers(static_cast<size_t>(layers_count));
    for (QwenBlockWeights& layer : layers) {
        layer.norm1_w = Tensor({hidden}, DataType::FP16);
        layer.norm2_w = Tensor({hidden}, DataType::FP16);
        layer.b_q = Tensor({hidden}, DataType::FP16);
        layer.b_k = Tensor({256}, DataType::FP16);
        layer.b_v = Tensor({256}, DataType::FP16);
    }
    Tensor final_norm({hidden}, DataType::FP16);
    arm_neon::GPTQInt8Weight lm_head;
    arm_neon::AttentionConfig attention{hidden, 12, 2, 128};
    arm_neon::FFNConfig ffn{hidden, intermediate};

    QwenPlanArtifacts artifacts = build_qwen_single_decode_plan(
        layers, final_norm, lm_head, attention, ffn, 8192, 1e-6f);
    ASSERT_TRUE(artifacts.plan);
    EXPECT_EQ(artifacts.plan->memory_stats().node_count,
              static_cast<size_t>(layers_count * 6 + 2));
    EXPECT_LT(artifacts.plan->memory_stats().arena_bytes,
              artifacts.coarse_workspace_bytes / 100);

    bool has_attention = false;
    bool has_fused_ffn = false;
    for (KernelNode* node : artifacts.plan->order()) {
        has_attention |= node->name().find(".attention") != std::string::npos;
        has_fused_ffn |= node->name().find("fused_gate_up_swiglu") != std::string::npos;
    }
    EXPECT_TRUE(has_attention);
    EXPECT_TRUE(has_fused_ffn);
}

TEST(QwenPlanLayoutTest, PrefillUsesCapacitySpecializedPlan) {
    constexpr int hidden = 1536;
    std::vector<QwenBlockWeights> layers(2);
    for (QwenBlockWeights& layer : layers) {
        layer.norm1_w = Tensor({hidden}, DataType::FP16);
        layer.norm2_w = Tensor({hidden}, DataType::FP16);
        layer.b_q = Tensor({hidden}, DataType::FP16);
        layer.b_k = Tensor({256}, DataType::FP16);
        layer.b_v = Tensor({256}, DataType::FP16);
    }
    Tensor final_norm({hidden}, DataType::FP16);
    arm_neon::GPTQInt8Weight lm_head;
    arm_neon::AttentionConfig attention{hidden, 12, 2, 128};
    arm_neon::FFNConfig ffn{hidden, 8960};

    QwenPlanArtifacts decode = build_qwen_single_decode_plan(
        layers, final_norm, lm_head, attention, ffn, 8192, 1e-6f);
    QwenPlanArtifacts prefill = build_qwen_prefill_plan(
        layers, final_norm, lm_head, attention, ffn, 8192, 128, 1e-6f);
    ASSERT_TRUE(prefill.plan);
    EXPECT_EQ(prefill.plan->spec().mode, ExecutionMode::PREFILL);
    EXPECT_EQ(prefill.plan->spec().row_capacity, 128);
    EXPECT_GT(prefill.plan->memory_stats().arena_bytes,
              decode.plan->memory_stats().arena_bytes);
    EXPECT_LT(prefill.plan->memory_stats().arena_bytes,
              prefill.coarse_workspace_bytes);
}

} // namespace
} // namespace llm_engine
