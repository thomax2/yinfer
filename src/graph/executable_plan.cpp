#include "llm_engine/graph/executable_plan.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <queue>
#include <stdexcept>
#include <unordered_set>

#include "llm_engine/memory/memory_pool.h"

namespace llm_engine {
namespace {

size_t align_up(size_t value, size_t alignment) {
    if (alignment == 0) return value;
    const size_t remainder = value % alignment;
    return remainder == 0 ? value : value + alignment - remainder;
}

struct FreeBlock {
    size_t offset;
    size_t bytes;
};

void coalesce(std::vector<FreeBlock>& blocks) {
    if (blocks.empty()) return;
    std::sort(blocks.begin(), blocks.end(), [](const FreeBlock& a, const FreeBlock& b) {
        return a.offset < b.offset;
    });
    size_t write = 0;
    for (size_t read = 1; read < blocks.size(); ++read) {
        FreeBlock& current = blocks[write];
        const FreeBlock& next = blocks[read];
        if (current.offset + current.bytes == next.offset) {
            current.bytes += next.bytes;
        } else {
            ++write;
            blocks[write] = next;
        }
    }
    blocks.resize(write + 1);
}

} // namespace

void ExecutionContext::bind_external(PlanValueId id, void* data) {
    external_bindings_[id] = data;
}

void* ExecutionContext::value_data(PlanValueId id) const {
    if (id >= resolved_values_.size()) return nullptr;
    return resolved_values_[id];
}

ExecutionWorkspace::~ExecutionWorkspace() {
    reset();
}

bool ExecutionWorkspace::ensure_capacity(size_t required_bytes) {
    if (required_bytes <= bytes_) return true;
    if (g_memory_pool == nullptr) return false;

    void* replacement = g_memory_pool->allocate(required_bytes);
    if (replacement == nullptr) return false;
    std::memset(replacement, 0, required_bytes);

    if (data_ != nullptr) g_memory_pool->free_block(data_);
    data_ = replacement;
    bytes_ = required_bytes;
    ++reallocations_;
    return true;
}

void ExecutionWorkspace::reset() {
    if (data_ != nullptr && g_memory_pool != nullptr) {
        g_memory_pool->free_block(data_);
    }
    data_ = nullptr;
    bytes_ = 0;
}

KernelNode::KernelNode(
    std::string name,
    std::vector<PlanValueId> inputs,
    std::vector<PlanValueId> outputs,
    ParallelismPolicy policy)
    : name_(std::move(name)),
      inputs_(std::move(inputs)),
      outputs_(std::move(outputs)),
      policy_(policy) {}

CallbackKernelNode::CallbackKernelNode(
    std::string name,
    std::vector<PlanValueId> inputs,
    std::vector<PlanValueId> outputs,
    Callback callback,
    ParallelismPolicy policy)
    : KernelNode(std::move(name), std::move(inputs), std::move(outputs), policy),
      callback_(std::move(callback)) {
    if (!callback_) throw std::invalid_argument("kernel callback is empty");
}

Status CallbackKernelNode::run(ExecutionContext& context) {
    return callback_(context);
}

Status ExecutablePlan::run(
    ExecutionContext& context,
    ExecutionWorkspace& workspace) const {
    if (context.actual_rows <= 0 || context.actual_rows > spec_.row_capacity) {
        return Status::INVALID_ARGUMENT;
    }
    if (!workspace.ensure_capacity(memory_stats_.arena_bytes)) {
        return Status::OUT_OF_MEMORY;
    }

    context.resolved_values_.assign(values_.size(), nullptr);
    char* arena = static_cast<char*>(workspace.data());
    for (PlanValueId id = 0; id < values_.size(); ++id) {
        const PlanValueSlot& slot = slots_[id];
        if (slot.external) {
            auto binding = context.external_bindings_.find(id);
            if (binding == context.external_bindings_.end() || binding->second == nullptr) {
                return Status::INVALID_ARGUMENT;
            }
            context.resolved_values_[id] = binding->second;
        } else {
            context.resolved_values_[id] = arena + slot.offset;
        }
    }

    for (KernelNode* node : order_) {
        if (node == nullptr) return Status::INVALID_ARGUMENT;
        Status status = node->run(context);
        if (status != Status::SUCCESS) return status;
    }
    return Status::SUCCESS;
}

ExecutablePlanBuilder::ExecutablePlanBuilder(CompileSpec spec) : spec_(spec) {
    if (spec_.row_capacity <= 0) {
        throw std::invalid_argument("row_capacity must be positive");
    }
}

PlanValueId ExecutablePlanBuilder::add_value(PlanValueDesc desc) {
    if (desc.name.empty()) throw std::invalid_argument("plan value name is empty");
    if (desc.bytes == 0) throw std::invalid_argument("plan value has zero bytes");
    if (desc.alignment == 0) desc.alignment = 1;
    if ((desc.alignment & (desc.alignment - 1)) != 0) {
        throw std::invalid_argument("plan value alignment must be a power of two");
    }
    if (values_.size() >= static_cast<size_t>(kInvalidPlanValue)) {
        throw std::overflow_error("too many plan values");
    }
    values_.push_back(std::move(desc));
    return static_cast<PlanValueId>(values_.size() - 1);
}

KernelNode* ExecutablePlanBuilder::add_node(std::unique_ptr<KernelNode> node) {
    if (!node) throw std::invalid_argument("cannot add a null kernel node");
    KernelNode* raw = node.get();
    nodes_.push_back(std::move(node));
    return raw;
}

ExecutablePlan ExecutablePlanBuilder::compile() {
    ExecutablePlan result;
    result.spec_ = spec_;
    result.values_ = values_;
    result.nodes_ = std::move(nodes_);
    result.slots_.resize(values_.size()); // 初始化物理内存布局表

    // === 第一阶段：构建依赖图（数据流分析） ===
    const size_t node_count = result.nodes_.size();
    std::vector<int> producer(values_.size(), -1); // 记录每个 Value 是由哪个 Node 生产的
    std::vector<std::vector<size_t>> consumers(values_.size()); // 记录每个 Value 被哪些 Node 消费
    std::vector<int> indegree(node_count, 0); // 拓扑排序用的入度表
    std::vector<std::vector<size_t>> edges(node_count); // 拓扑排序用的边表

    // 遍历所有节点，建立"值"与"节点"的关系网
    for (size_t index = 0; index < node_count; ++index) {
        const KernelNode& node = *result.nodes_[index];
        // 建立生产者关系：遍历节点的 outputs，记录谁生产了这个 Value
        for (PlanValueId id : node.outputs()) {
            if (id >= values_.size()) throw std::runtime_error("kernel output value is invalid");
            if (producer[id] != -1) throw std::runtime_error("plan value has multiple producers"); // 数据流不能有歧义
            producer[id] = static_cast<int>(index);
        }
        // 建立消费者关系：遍历节点的 inputs，记录谁消费了这个 Value
        for (PlanValueId id : node.inputs()) {
            if (id >= values_.size()) throw std::runtime_error("kernel input value is invalid");
            consumers[id].push_back(index);
        }
    }

    // 校验：检查是否有内部变量没有生产者（悬空输入），或者节点直接消费自己的输出
    for (size_t id = 0; id < values_.size(); ++id) {
        if (producer[id] == -1 && !values_[id].external && !consumers[id].empty()) {
            throw std::runtime_error("internal plan input has no producer");
        }
        if (producer[id] == -1) continue;
        std::unordered_set<size_t> unique_consumers;
        for (size_t consumer : consumers[id]) {
            if (consumer == static_cast<size_t>(producer[id])) {
                throw std::runtime_error("kernel directly consumes its own output"); // 死循环
            }
            if (!unique_consumers.insert(consumer).second) continue;
            edges[static_cast<size_t>(producer[id])].push_back(consumer);
            ++indegree[consumer];
        }
    }

    // === 第二阶段：拓扑排序（Kahn算法） ===
    std::priority_queue<size_t, std::vector<size_t>, std::greater<size_t>> ready; // 优先队列使 Node ID 小的先执行，方便调试
    for (size_t index = 0; index < node_count; ++index) {
        if (indegree[index] == 0) ready.push(index); // 找出入度为0的节点（没有依赖的节点）
    }
    std::vector<size_t> ordered_indices;
    while (!ready.empty()) {
        size_t index = ready.top();
        ready.pop();
        ordered_indices.push_back(index); // 加入执行序列
        // 相当于"删除"当前节点，将后继节点的入度减1
        for (size_t next : edges[index]) {
            if (--indegree[next] == 0) ready.push(next);
        }
    }
    if (ordered_indices.size() != node_count) {
        throw std::runtime_error("executable plan contains a cycle"); // 存在循环依赖
    }
    result.order_.reserve(node_count);
    for (size_t index : ordered_indices) result.order_.push_back(result.nodes_[index].get()); // 生成执行时间表

    // === 第三阶段：生命周期分析（内存优化的关键前提） ===
    std::vector<int> order_position(node_count, -1);
    for (size_t position = 0; position < ordered_indices.size(); ++position) {
        order_position[ordered_indices[position]] = static_cast<int>(position);
    }

    struct Lifetime { int first = -1; int last = -1; };
    std::vector<Lifetime> life(values_.size());
    for (size_t id = 0; id < values_.size(); ++id) {
        if (values_[id].external) continue;
        // 出生时刻 = 生产者节点的执行序号
        if (producer[id] >= 0) {
            life[id].first = order_position[static_cast<size_t>(producer[id])];
            life[id].last = life[id].first;
        }
        // 逝去时刻 = 最后一个消费者节点的执行序号
        for (size_t consumer : consumers[id]) {
            int position = order_position[consumer];
            if (life[id].first == -1) life[id].first = position;
            life[id].last = std::max(life[id].last, position);
        }
    }

    // 按生命周期将变量分到对应的执行步（starts: 出生, ends: 死亡）
    std::vector<std::vector<PlanValueId>> starts(node_count);
    std::vector<std::vector<PlanValueId>> ends(node_count);
    for (PlanValueId id = 0; id < values_.size(); ++id) {
        PlanValueSlot& slot = result.slots_[id];
        slot.bytes = values_[id].bytes;
        slot.external = values_[id].external;
        if (values_[id].external) {
            result.memory_stats_.external_value_bytes += values_[id].bytes;
            continue;
        }
        result.memory_stats_.sum_internal_value_bytes += values_[id].bytes;
        if (life[id].first >= 0) {
            starts[static_cast<size_t>(life[id].first)].push_back(id);
            ends[static_cast<size_t>(life[id].last)].push_back(id);
        }
    }

    // === 第四阶段：Arena 内存分配（First-Fit 算法） ===
    std::vector<FreeBlock> free_blocks; // 空闲内存块列表
    size_t arena_end = 0; // 当前分配到的最大末尾
    size_t live_bytes = 0;
    for (size_t step = 0; step < node_count; ++step) {
        // 1. 分配内存：取出这一步需要"出生"的变量，按大小从大到小排序以减少碎片
        std::sort(starts[step].begin(), starts[step].end(), [&](PlanValueId a, PlanValueId b) {
            if (values_[a].bytes != values_[b].bytes) {
                return values_[a].bytes > values_[b].bytes;
            }
            return values_[a].alignment > values_[b].alignment;
        });
        for (PlanValueId id : starts[step]) {
            const PlanValueDesc& desc = values_[id];
            bool placed = false;
            // 尝试在 free_blocks 中找第一个能放下的空闲块（First-Fit）
            for (size_t block_index = 0; block_index < free_blocks.size(); ++block_index) {
                FreeBlock block = free_blocks[block_index];
                size_t aligned = align_up(block.offset, desc.alignment);
                size_t padding = aligned - block.offset;
                if (padding + desc.bytes > block.bytes) continue;

                // 找到了，切分空闲块
                free_blocks.erase(free_blocks.begin() + static_cast<std::ptrdiff_t>(block_index));
                if (padding != 0) free_blocks.push_back({block.offset, padding});
                size_t used_end = aligned + desc.bytes;
                size_t block_end = block.offset + block.bytes;
                if (used_end < block_end) free_blocks.push_back({used_end, block_end - used_end});
                result.slots_[id].offset = aligned;
                placed = true;
                break;
            }
            if (!placed) {
                // 没找到空闲块，向 Arena 末尾追加新内存
                size_t aligned = align_up(arena_end, desc.alignment);
                result.slots_[id].offset = aligned;
                arena_end = aligned + desc.bytes;
            }
            live_bytes += desc.bytes;
        }
        result.memory_stats_.peak_live_value_bytes =
            std::max(result.memory_stats_.peak_live_value_bytes, live_bytes);
        
        // 2. 回收内存：取出这一步需要"死亡"的变量，归还为碎片
        for (PlanValueId id : ends[step]) {
            free_blocks.push_back({result.slots_[id].offset, result.slots_[id].bytes});
            live_bytes -= result.slots_[id].bytes;
        }
        coalesce(free_blocks); // 碎片整理：合并相邻的空闲块
    }

    result.memory_stats_.arena_bytes = align_up(arena_end, 64); // 64 字节对齐
    result.memory_stats_.node_count = node_count;
    return result;
}

const char* execution_mode_name(ExecutionMode mode) {
    switch (mode) {
        case ExecutionMode::SINGLE_DECODE: return "single_decode";
        case ExecutionMode::PREFILL: return "prefill";
        case ExecutionMode::SELECTIVE_DECODE: return "selective_decode";
        case ExecutionMode::MIXED_SELECTIVE_BATCH: return "mixed_selective_batch";
    }
    return "unknown";
}

} // namespace llm_engine
