#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <memory>
#include <vector>

#include "llm_engine/graph/executable_plan.h"
#include "llm_engine/graph/graph.h"
#include "llm_engine/memory/memory_pool.h"
#include "qwen_execution_plan.h"

namespace {

using Clock = std::chrono::steady_clock;

llm_engine::ExecutablePlan make_dispatch_plan(int node_count) {
    using namespace llm_engine;
    ExecutablePlanBuilder builder({ExecutionMode::SINGLE_DECODE, 1});
    PlanValueId previous = builder.add_value({"input", 1, 1, true});
    for (int index = 0; index < node_count; ++index) {
        PlanValueId next = builder.add_value(
            {"value." + std::to_string(index), 1, 1, false});
        builder.add_node(std::make_unique<CallbackKernelNode>(
            "noop." + std::to_string(index),
            std::vector<PlanValueId>{previous},
            std::vector<PlanValueId>{next},
            [](ExecutionContext&) { return Status::SUCCESS; }));
        previous = next;
    }
    return builder.compile();
}

double benchmark_dispatch(const llm_engine::ExecutablePlan& plan, int iterations) {
    using namespace llm_engine;
    unsigned char input = 0;
    ExecutionContext context;
    context.bind_external(0, &input);
    ExecutionWorkspace workspace;
    if (plan.run(context, workspace) != Status::SUCCESS) return -1.0;
    auto begin = Clock::now();
    for (int iteration = 0; iteration < iterations; ++iteration) {
        if (plan.run(context, workspace) != Status::SUCCESS) return -1.0;
    }
    auto end = Clock::now();
    return std::chrono::duration<double, std::micro>(end - begin).count() / iterations;
}

} // namespace

int main() {
    using namespace llm_engine;
    MemoryPool pool(64 * 1024 * 1024);
    g_memory_pool = &pool;

    constexpr int layer_count = 28;
    constexpr int hidden = 1536;
    constexpr int intermediate = 8960;
    std::vector<QwenBlockWeights> layers(static_cast<size_t>(layer_count));
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

    auto compile_begin = Clock::now();
    QwenPlanArtifacts artifacts = build_qwen_single_decode_plan(
        layers, final_norm, lm_head, attention, ffn, 8192, 1e-6f);
    QwenPlanArtifacts prefill = build_qwen_prefill_plan(
        layers, final_norm, lm_head, attention, ffn, 8192, 128, 1e-6f);
    auto compile_end = Clock::now();

    const auto& memory = artifacts.plan->memory_stats();
    const double reduction = artifacts.coarse_workspace_bytes == 0 ? 0.0 :
        100.0 * (1.0 - static_cast<double>(memory.arena_bytes) /
            static_cast<double>(artifacts.coarse_workspace_bytes));

    ExecutablePlan coarse_dispatch = make_dispatch_plan(layer_count + 2);
    ExecutablePlan fine_dispatch = make_dispatch_plan(layer_count * 6 + 2);
    constexpr int iterations = 20000;
    double coarse_us = benchmark_dispatch(coarse_dispatch, iterations);
    double fine_us = benchmark_dispatch(fine_dispatch, iterations);

    std::cout << std::fixed << std::setprecision(3)
              << "plan_compile_us="
              << std::chrono::duration<double, std::micro>(
                     compile_end - compile_begin).count() << '\n'
              << "coarse_workspace_bytes=" << artifacts.coarse_workspace_bytes << '\n'
              << "fine_arena_bytes=" << memory.arena_bytes << '\n'
              << "fine_peak_live_bytes=" << memory.peak_live_value_bytes << '\n'
              << "memory_reduction_percent=" << reduction << '\n'
              << "prefill_capacity_128_arena_bytes="
              << prefill.plan->memory_stats().arena_bytes << '\n'
              << "prefill_capacity_128_peak_live_bytes="
              << prefill.plan->memory_stats().peak_live_value_bytes << '\n'
              << "coarse_dispatch_nodes=" << coarse_dispatch.order().size() << '\n'
              << "fine_dispatch_nodes=" << fine_dispatch.order().size() << '\n'
              << "coarse_dispatch_us_per_run=" << coarse_us << '\n'
              << "fine_dispatch_us_per_run=" << fine_us << '\n'
              << "fine_dispatch_ns_per_node="
              << fine_us * 1000.0 / fine_dispatch.order().size() << '\n';

    g_memory_pool = nullptr;
    return 0;
}
