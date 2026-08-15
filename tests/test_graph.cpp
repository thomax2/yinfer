#include <gtest/gtest.h>

#include <cstring>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "llm_engine/graph/compiler.h"
#include "llm_engine/graph/graph.h"
#include "llm_engine/memory/memory_pool.h"

using namespace llm_engine;

namespace {

void set_test_env(const char* name, const char* value) {
#ifdef _WIN32
    _putenv_s(name, value ? value : "");
#else
    if (value) {
        setenv(name, value, 1);
    } else {
        unsetenv(name);
    }
#endif
}

class ScopedEnv final {
public:
    ScopedEnv(const char* name, const char* value) : name_(name) {
        const char* old = std::getenv(name);
        if (old) {
            had_old_ = true;
            old_ = old;
        }
        set_test_env(name, value);
    }

    ~ScopedEnv() {
        set_test_env(name_.c_str(), had_old_ ? old_.c_str() : nullptr);
    }

private:
    std::string name_;
    std::string old_;
    bool had_old_ = false;
};

void init_memory_pool() {
    if (!g_memory_pool) {
        g_memory_pool = new MemoryPool(512ULL * 1024 * 1024);
    }
}

TEST(KVCacheModeTest, SchedulerSelectsPagedStorageWithoutExtraSwitch) {
    ScopedEnv scheduler("LLM_ENABLE_SCHEDULER", "1");
    ScopedEnv paged("LLM_PAGED_KV", nullptr);
    KVCache cache(1, 8, 1, 4);
    EXPECT_TRUE(cache.is_paged());
}

class CopyNode final : public GraphNode {
public:
    CopyNode(Tensor* input, Tensor* output) {
        inputs = {input};
        outputs = {output};
    }

    Status forward() override {
        if (!inputs[0]->data || !outputs[0]->data ||
            inputs[0]->dtype != outputs[0]->dtype ||
            inputs[0]->shape != outputs[0]->shape) {
            return Status::INVALID_ARGUMENT;
        }
        if (inputs[0]->data != outputs[0]->data) {
            std::memcpy(outputs[0]->data, inputs[0]->data, inputs[0]->bytes());
        }
        return Status::SUCCESS;
    }
};

class DoubleNode final : public GraphNode {
public:
    DoubleNode(Tensor* input, Tensor* output) {
        inputs = {input};
        outputs = {output};
    }

    Status forward() override {
        const float* input = inputs[0]->ptr<float>();
        float* output = outputs[0]->ptr<float>();
        if (!input || !output) return Status::INVALID_ARGUMENT;
        for (size_t i = 0; i < inputs[0]->size(); ++i) output[i] = input[i] * 2.0f;
        return Status::SUCCESS;
    }
};

class ScaleMutationNode final : public GraphNode {
public:
    ScaleMutationNode(Tensor* input, Tensor* output, float scale, bool allow_alias)
        : scale_(scale), allow_alias_(allow_alias) {
        inputs = {input};
        outputs = {output};
    }

    Status forward() override {
        const float* input = inputs[0]->ptr<float>();
        float* output = outputs[0]->ptr<float>();
        if (!input || !output || inputs[0]->shape != outputs[0]->shape) {
            return Status::INVALID_ARGUMENT;
        }
        if (input != output) {
            std::memcpy(output, input, inputs[0]->bytes());
        }
        for (size_t i = 0; i < outputs[0]->size(); ++i) output[i] *= scale_;
        return Status::SUCCESS;
    }

    std::vector<InplaceAliasCandidate> inplace_alias_candidates() const override {
        return allow_alias_ ? std::vector<InplaceAliasCandidate>{{0, 0}}
                            : std::vector<InplaceAliasCandidate>{};
    }

private:
    float scale_;
    bool allow_alias_;
};

class LegacySelfAliasNode final : public GraphNode {
public:
    explicit LegacySelfAliasNode(Tensor* tensor) {
        inputs = {tensor};
        outputs = {tensor};
    }
    Status forward() override { return Status::SUCCESS; }
};

Tensor* add_scale_mutation(
    ComputationGraph& graph,
    Tensor* handle,
    float scale,
    bool allow_alias = true
) {
    Tensor* input = graph.latest_value(handle);
    Tensor* output = graph.create_mutation_version(handle);
    graph.add_node(std::make_unique<ScaleMutationNode>(input, output, scale, allow_alias));
    return output;
}

GraphNode* add_copy(ComputationGraph& graph, Tensor* input, Tensor* output) {
    return graph.add_node(std::make_unique<CopyNode>(graph.latest_value(input), output));
}

} // namespace

TEST(FunctionalizationTest, MutationAndSubsequentUseRemainOrdered) {
    init_memory_pool();
    ComputationGraph graph;
    std::vector<float> external_x = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> external_out(4, 0.0f);
    Tensor* x = graph.create_tensor_from_ptr({2, 2}, external_x.data(), DataType::FP32);
    Tensor* out = graph.create_tensor_from_ptr({2, 2}, external_out.data(), DataType::FP32);

    Tensor* x1 = add_scale_mutation(graph, x, 1.1f);
    GraphNode* consumer = add_copy(graph, x, out);

    ASSERT_NE(x, x1);
    ASSERT_EQ(consumer->inputs[0], x1);

    GraphCompiler compiler;
    std::vector<GraphNode*> plan = compiler.compile(graph);
    ASSERT_EQ(plan.size(), 3u); // mutation, consumer, external copy-back
    EXPECT_EQ(x->data, x1->data); // safe reinplace selected by the planner

    GraphRuntime runtime;
    ASSERT_EQ(runtime.run(plan), Status::SUCCESS);
    EXPECT_FLOAT_EQ(external_out[0], 1.1f);
    EXPECT_FLOAT_EQ(external_out[3], 4.4f);
    EXPECT_FLOAT_EQ(external_x[3], 4.4f);
}

TEST(FunctionalizationTest, MutationAsFinalExternalOutputIsWrittenBack) {
    init_memory_pool();
    ComputationGraph graph;
    std::vector<float> external_x = {10.0f, 11.0f, 12.0f, 13.0f};
    Tensor* x = graph.create_tensor_from_ptr({2, 2}, external_x.data(), DataType::FP32);

    Tensor* x1 = add_scale_mutation(graph, x, 1.1f);
    GraphCompiler compiler;
    std::vector<GraphNode*> plan = compiler.compile(graph);
    ASSERT_EQ(plan.size(), 2u);
    EXPECT_EQ(x1->data, x->data);

    GraphRuntime runtime;
    ASSERT_EQ(runtime.run(plan), Status::SUCCESS);
    EXPECT_FLOAT_EQ(external_x[0], 11.0f);
    EXPECT_FLOAT_EQ(external_x[3], 14.3f);
}

TEST(FunctionalizationTest, ConsecutiveMutationsCreateDistinctValuesAndReuseStorage) {
    init_memory_pool();
    ComputationGraph graph;
    std::vector<float> external_x(16, 1.0f);
    std::vector<float> external_out(16, 0.0f);
    Tensor* x = graph.create_tensor_from_ptr({4, 4}, external_x.data(), DataType::FP32);
    Tensor* out = graph.create_tensor_from_ptr({4, 4}, external_out.data(), DataType::FP32);

    Tensor* x1 = add_scale_mutation(graph, x, 1.1f);
    Tensor* x2 = add_scale_mutation(graph, x, 1.1f);
    Tensor* x3 = add_scale_mutation(graph, x, 1.1f);
    add_copy(graph, x, out);

    ASSERT_NE(x, x1);
    ASSERT_NE(x1, x2);
    ASSERT_NE(x2, x3);
    ASSERT_EQ(graph.latest_value(x), x3);

    GraphCompiler compiler;
    std::vector<GraphNode*> plan = compiler.compile(graph);
    EXPECT_EQ(x->data, x1->data);
    EXPECT_EQ(x1->data, x2->data);
    EXPECT_EQ(x2->data, x3->data);

    GraphRuntime runtime;
    ASSERT_EQ(runtime.run(plan), Status::SUCCESS);
    EXPECT_NEAR(external_out[0], 1.331f, 1e-5f);
    EXPECT_NEAR(external_x[0], 1.331f, 1e-5f);
}

TEST(FunctionalizationTest, CopyBackPreservesSemanticsWhenReinplaceIsForbidden) {
    init_memory_pool();
    ComputationGraph graph;
    std::vector<float> external_x = {2.0f, 4.0f, 6.0f, 8.0f};
    Tensor* x = graph.create_tensor_from_ptr({2, 2}, external_x.data(), DataType::FP32);
    Tensor* x1 = add_scale_mutation(graph, x, 3.0f, false);

    GraphCompiler compiler;
    std::vector<GraphNode*> plan = compiler.compile(graph);
    ASSERT_EQ(plan.size(), 2u);
    EXPECT_NE(x->data, x1->data);
    EXPECT_GE(graph.arena_size, x1->bytes());

    GraphRuntime runtime;
    ASSERT_EQ(runtime.run(plan), Status::SUCCESS);
    EXPECT_FLOAT_EQ(external_x[0], 6.0f);
    EXPECT_FLOAT_EQ(external_x[3], 24.0f);
}

TEST(FunctionalizationTest, RealRoPEBuilderUsesDistinctLogicalOutput) {
    init_memory_pool();
    ComputationGraph graph;
    std::vector<float> external_x = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> cosine = {1.0f, 1.0f};
    std::vector<float> sine = {0.0f, 0.0f};
    std::vector<float> external_out(4, 0.0f);
    Tensor* x = graph.create_tensor_from_ptr({1, 4}, external_x.data(), DataType::FP32);
    Tensor* cos = graph.create_tensor_from_ptr({2}, cosine.data(), DataType::FP32);
    Tensor* sin = graph.create_tensor_from_ptr({2}, sine.data(), DataType::FP32);
    Tensor* out = graph.create_tensor_from_ptr({1, 4}, external_out.data(), DataType::FP32);

    RoPENode* rope = graph.add_rope(x, cos, sin);
    Tensor* x1 = graph.latest_value(x);
    GraphNode* consumer = add_copy(graph, x, out);
    ASSERT_NE(x, x1);
    ASSERT_EQ(rope->inputs[0], x);
    ASSERT_EQ(rope->outputs[0], x1);
    ASSERT_EQ(consumer->inputs[0], x1);

    GraphCompiler compiler;
    std::vector<GraphNode*> plan = compiler.compile(graph);
    EXPECT_EQ(x->data, x1->data);
    GraphRuntime runtime;
    ASSERT_EQ(runtime.run(plan), Status::SUCCESS);
    EXPECT_EQ(external_out, external_x);
}

TEST(FunctionalizationTest, GenericOutputAliasingAnInputCreatesNewVersion) {
    init_memory_pool();
    ComputationGraph graph;
    std::vector<float> external_x = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> external_zero(4, 0.0f);
    Tensor* x = graph.create_tensor_from_ptr({2, 2}, external_x.data(), DataType::FP32);
    Tensor* zero = graph.create_tensor_from_ptr({2, 2}, external_zero.data(), DataType::FP32);

    AddNode* add = graph.add_add(x, zero, x);
    Tensor* x1 = graph.latest_value(x);
    ASSERT_NE(x, x1);
    ASSERT_EQ(add->inputs[0], x);
    ASSERT_EQ(add->outputs[0], x1);

    GraphCompiler compiler;
    std::vector<GraphNode*> plan = compiler.compile(graph);
    EXPECT_NE(x->data, x1->data); // Add has not declared in-place safety.
    GraphRuntime runtime;
    ASSERT_EQ(runtime.run(plan), Status::SUCCESS);
    EXPECT_EQ(external_x, (std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f}));
}

TEST(FunctionalizationTest, QwenBlockBuilderSeparatesHiddenStateVersions) {
    init_memory_pool();
    ComputationGraph graph;
    std::vector<float> hidden_data(4, 1.0f);
    std::vector<float> vector_data(4, 1.0f);
    Tensor* hidden = graph.create_tensor_from_ptr({1, 4}, hidden_data.data(), DataType::FP32);
    Tensor* norm1 = graph.create_tensor_from_ptr({4}, vector_data.data(), DataType::FP32);
    Tensor* bq = graph.create_tensor_from_ptr({4}, vector_data.data(), DataType::FP32);
    Tensor* bk = graph.create_tensor_from_ptr({4}, vector_data.data(), DataType::FP32);
    Tensor* bv = graph.create_tensor_from_ptr({4}, vector_data.data(), DataType::FP32);
    Tensor* cos = graph.create_tensor_from_ptr({4}, vector_data.data(), DataType::FP32);
    Tensor* sin = graph.create_tensor_from_ptr({4}, vector_data.data(), DataType::FP32);
    Tensor* norm2 = graph.create_tensor_from_ptr({4}, vector_data.data(), DataType::FP32);
    QwenBlockWeights weights;
    KVCache cache(1, 4, 1, 4);
    int position = 0;
    arm_neon::AttentionConfig attention{4, 1, 1, 4};
    arm_neon::FFNConfig ffn{4, 8};

    QwenBlockNode* block = graph.add_qwen_block(
        hidden, norm1, bq, bk, bv, cos, sin, norm2, &weights,
        &cache, 0, &position, attention, ffn, 1e-6f);
    Tensor* hidden1 = graph.latest_value(hidden);
    ASSERT_NE(hidden, hidden1);
    ASSERT_EQ(block->inputs[0], hidden);
    ASSERT_EQ(block->outputs[0], hidden1);

    GraphCompiler compiler;
    std::vector<GraphNode*> plan = compiler.compile(graph);
    ASSERT_EQ(plan.size(), 2u);
    EXPECT_EQ(hidden->data, hidden1->data);
}

TEST(ArenaPlannerTest, FunctionalizedInternalMutationAndLaterValueReuseStorage) {
    init_memory_pool();
    ComputationGraph graph;
    std::vector<float> external_input = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> external_output(4, 0.0f);
    Tensor* input = graph.create_tensor_from_ptr({2, 2}, external_input.data(), DataType::FP32);
    Tensor* output = graph.create_tensor_from_ptr({2, 2}, external_output.data(), DataType::FP32);
    Tensor* x = graph.create_tensor({2, 2}, DataType::FP32);
    Tensor* dummy = graph.create_tensor({2, 2}, DataType::FP32);
    Tensor* y = graph.create_tensor({2, 2}, DataType::FP32);

    add_copy(graph, input, x);
    Tensor* x1 = add_scale_mutation(graph, x, 2.0f);
    add_copy(graph, x, dummy);
    graph.add_node(std::make_unique<DoubleNode>(dummy, y));
    add_copy(graph, y, output);

    GraphCompiler compiler;
    std::vector<GraphNode*> plan = compiler.compile(graph);
    EXPECT_EQ(x->data, x1->data);
    EXPECT_EQ(x->data, y->data);

    GraphRuntime runtime;
    ASSERT_EQ(runtime.run(plan), Status::SUCCESS);
    EXPECT_FLOAT_EQ(external_output[0], 4.0f);
    EXPECT_FLOAT_EQ(external_output[3], 16.0f);
}

TEST(GraphCompilerTest, StableTopologicalSortRepairsOutOfOrderInsertion) {
    init_memory_pool();
    ComputationGraph graph;
    std::vector<float> external_input = {3.0f, 4.0f};
    std::vector<float> external_output(2, 0.0f);
    Tensor* input = graph.create_tensor_from_ptr({2}, external_input.data(), DataType::FP32);
    Tensor* intermediate = graph.create_tensor({2}, DataType::FP32);
    Tensor* output = graph.create_tensor_from_ptr({2}, external_output.data(), DataType::FP32);

    GraphNode* consumer = graph.add_node(std::make_unique<CopyNode>(intermediate, output));
    GraphNode* producer = graph.add_node(std::make_unique<CopyNode>(input, intermediate));

    GraphCompiler compiler;
    std::vector<GraphNode*> plan = compiler.compile(graph);
    ASSERT_EQ(plan.size(), 2u);
    EXPECT_EQ(plan[0], producer);
    EXPECT_EQ(plan[1], consumer);

    GraphRuntime runtime;
    ASSERT_EQ(runtime.run(plan), Status::SUCCESS);
    EXPECT_EQ(external_output, external_input);
}

TEST(GraphCompilerTest, RejectsLegacyInputOutputSelfAlias) {
    init_memory_pool();
    ComputationGraph graph;
    std::vector<float> external_x(4, 1.0f);
    Tensor* x = graph.create_tensor_from_ptr({2, 2}, external_x.data(), DataType::FP32);
    graph.nodes.push_back(std::make_unique<LegacySelfAliasNode>(x));

    GraphCompiler compiler;
    EXPECT_THROW(compiler.compile(graph), std::runtime_error);
}

TEST(GraphCompilerTest, RejectsRealCycleInsteadOfReturningPartialPlan) {
    init_memory_pool();
    ComputationGraph graph;
    Tensor* a = graph.create_tensor({1}, DataType::FP32);
    Tensor* b = graph.create_tensor({1}, DataType::FP32);
    graph.add_node(std::make_unique<CopyNode>(a, b));
    graph.add_node(std::make_unique<CopyNode>(b, a));

    GraphCompiler compiler;
    EXPECT_THROW(compiler.compile(graph), std::runtime_error);
}

TEST(GraphCompilerTest, RejectsMultipleProducersForOneLogicalValue) {
    init_memory_pool();
    ComputationGraph graph;
    std::vector<float> input_a_data = {1.0f};
    std::vector<float> input_b_data = {2.0f};
    std::vector<float> output_data = {0.0f};
    Tensor* input_a = graph.create_tensor_from_ptr({1}, input_a_data.data(), DataType::FP32);
    Tensor* input_b = graph.create_tensor_from_ptr({1}, input_b_data.data(), DataType::FP32);
    Tensor* output = graph.create_tensor_from_ptr({1}, output_data.data(), DataType::FP32);
    graph.add_node(std::make_unique<CopyNode>(input_a, output));
    graph.add_node(std::make_unique<CopyNode>(input_b, output));

    GraphCompiler compiler;
    EXPECT_THROW(compiler.compile(graph), std::runtime_error);
}

TEST(GraphCompilerTest, RejectsUnboundGraphBoundary) {
    init_memory_pool();
    ComputationGraph graph;
    Tensor* input = graph.create_tensor({2, 2}, DataType::FP32);
    Tensor* output = graph.create_tensor({2, 2}, DataType::FP32);
    graph.add_node(std::make_unique<CopyNode>(input, output));

    GraphCompiler compiler;
    EXPECT_THROW(compiler.compile(graph), std::runtime_error);
}
