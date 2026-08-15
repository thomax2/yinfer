#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "llm_engine/status.h"

namespace llm_engine {

using PlanValueId = std::uint32_t;

constexpr PlanValueId kInvalidPlanValue = static_cast<PlanValueId>(-1);

enum class ExecutionMode {
    SINGLE_DECODE,
    PREFILL,
    SELECTIVE_DECODE,
    // Prefill 与 Decode 共用一个 token-major 执行计划；Attention 再按 Sequence 分段。
    MIXED_SELECTIVE_BATCH
};

enum class ParallelismPolicy {
    SERIAL,
    INTERNAL_THREAD_POOL,
    GRAPH_PARALLEL
};

struct CompileSpec {
    ExecutionMode mode = ExecutionMode::SINGLE_DECODE;
    int row_capacity = 1;
};

struct PlanValueDesc {
    std::string name;
    size_t bytes = 0;
    size_t alignment = 64;
    bool external = false;
};

struct PlanValueSlot {
    size_t offset = 0;
    size_t bytes = 0;
    bool external = false;
};

class ExecutionContext {
public:
    int actual_rows = 1;
    int current_pos = 0;
    int start_pos = 0;
    const int* positions = nullptr;
    void* user_data = nullptr;

    void bind_external(PlanValueId id, void* data);
    void* value_data(PlanValueId id) const;

    template <typename T>
    T* ptr(PlanValueId id) const {
        return static_cast<T*>(value_data(id));
    }

private:
    friend class ExecutablePlan;
    std::unordered_map<PlanValueId, void*> external_bindings_;
    std::vector<void*> resolved_values_;
};

class ExecutionWorkspace {
public:
    ExecutionWorkspace() = default;
    ~ExecutionWorkspace();

    ExecutionWorkspace(const ExecutionWorkspace&) = delete;
    ExecutionWorkspace& operator=(const ExecutionWorkspace&) = delete;

    bool ensure_capacity(size_t required_bytes);
    void reset();

    void* data() const { return data_; }
    size_t size() const { return bytes_; }
    size_t reallocations() const { return reallocations_; }

private:
    void* data_ = nullptr;
    size_t bytes_ = 0;
    size_t reallocations_ = 0;
};

class KernelNode {
public:
    KernelNode(
        std::string name,
        std::vector<PlanValueId> inputs,
        std::vector<PlanValueId> outputs,
        ParallelismPolicy policy = ParallelismPolicy::SERIAL);
    virtual ~KernelNode() = default;

    virtual Status run(ExecutionContext& context) = 0;

    const std::string& name() const { return name_; }
    const std::vector<PlanValueId>& inputs() const { return inputs_; }
    const std::vector<PlanValueId>& outputs() const { return outputs_; }
    ParallelismPolicy parallelism_policy() const { return policy_; }

private:
    std::string name_;
    std::vector<PlanValueId> inputs_;
    std::vector<PlanValueId> outputs_;
    ParallelismPolicy policy_;
};

class CallbackKernelNode final : public KernelNode {
public:
    using Callback = std::function<Status(ExecutionContext&)>;

    CallbackKernelNode(
        std::string name,
        std::vector<PlanValueId> inputs,
        std::vector<PlanValueId> outputs,
        Callback callback,
        ParallelismPolicy policy = ParallelismPolicy::SERIAL);

    Status run(ExecutionContext& context) override;

private:
    Callback callback_;
};

struct PlanMemoryStats {
    size_t arena_bytes = 0;
    size_t sum_internal_value_bytes = 0;
    size_t peak_live_value_bytes = 0;
    size_t external_value_bytes = 0;
    size_t node_count = 0;
};

class ExecutablePlan {
public:
    ExecutablePlan() = default;
    ExecutablePlan(ExecutablePlan&&) noexcept = default;
    ExecutablePlan& operator=(ExecutablePlan&&) noexcept = default;

    ExecutablePlan(const ExecutablePlan&) = delete;
    ExecutablePlan& operator=(const ExecutablePlan&) = delete;

    Status run(ExecutionContext& context, ExecutionWorkspace& workspace) const; // 按 order_ 顺序执行所有节点

    const CompileSpec& spec() const { return spec_; } // 编译配置快照（模式、容量等）
    const std::vector<PlanValueDesc>& values() const { return values_; } // 张量元数据表（名称、大小、对齐）
    const std::vector<PlanValueSlot>& slots() const { return slots_; } // 物理内存布局表（Arena 偏移量、生命周期区间）
    const std::vector<KernelNode*>& order() const { return order_; } // 拓扑排序后的执行时间表（指针数组）
    const PlanMemoryStats& memory_stats() const { return memory_stats_; } // 内存统计报告（arena_bytes、node_count 等）

private:
    friend class ExecutablePlanBuilder;

    CompileSpec spec_; // 编译配置快照
    std::vector<PlanValueDesc> values_; // 张量元数据表
    std::vector<PlanValueSlot> slots_; // 物理内存布局表
    std::vector<std::unique_ptr<KernelNode>> nodes_; // 节点对象池（持有所有权）
    std::vector<KernelNode*> order_; // 执行时间表
    PlanMemoryStats memory_stats_; // 内存统计报告
};

class ExecutablePlanBuilder {
public:
    explicit ExecutablePlanBuilder(CompileSpec spec);

    PlanValueId add_value(PlanValueDesc desc);
    KernelNode* add_node(std::unique_ptr<KernelNode> node);
    ExecutablePlan compile();

private:
    CompileSpec spec_;
    std::vector<PlanValueDesc> values_;
    std::vector<std::unique_ptr<KernelNode>> nodes_;
};

const char* execution_mode_name(ExecutionMode mode);

} // namespace llm_engine
