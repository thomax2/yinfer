#include "llm_engine/graph/compiler.h"

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <queue>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace llm_engine {

namespace {

bool env_flag(const char* name) {
    const char* value = std::getenv(name);
    if (!value) return false;
    const std::string text(value);
    return text == "1" || text == "true" || text == "TRUE" ||
           text == "on" || text == "ON";
}

size_t align_up_64(size_t size) {
    return (size + 63) & ~size_t(63);
}

struct TensorLife {
    int start = -1;
    int end = -1;
};

struct FreeBlock {
    size_t offset;
    size_t size;
};

struct StorageGroup {
    bool external = false;
    void* external_data = nullptr;
    Tensor* external_owner = nullptr;
    bool owner_owns_data = false;
    size_t offset = 0;
    size_t size = 0;
    int end = -1;
    bool released = false;
};

bool tensors_are_alias_compatible(const Tensor* input, const Tensor* output) {
    return input && output && input->shape == output->shape &&
           input->dtype == output->dtype && input->device == output->device &&
           input->bytes() == output->bytes();
}

void coalesce_free_blocks(std::vector<FreeBlock>& blocks) {
    if (blocks.empty()) return;
    std::sort(blocks.begin(), blocks.end(), [](const FreeBlock& lhs, const FreeBlock& rhs) {
        return lhs.offset < rhs.offset;
    });
    for (size_t i = 0; i + 1 < blocks.size();) {
        if (blocks[i].offset + blocks[i].size == blocks[i + 1].offset) {
            blocks[i].size += blocks[i + 1].size;
            blocks.erase(blocks.begin() + static_cast<std::ptrdiff_t>(i + 1));
        } else {
            ++i;
        }
    }
}

size_t allocate_first_fit(
    size_t requested,
    std::vector<FreeBlock>& free_blocks,
    size_t& peak_memory
) {
    for (auto it = free_blocks.begin(); it != free_blocks.end(); ++it) {
        if (it->size < requested) continue;
        const size_t offset = it->offset;
        if (it->size == requested) {
            free_blocks.erase(it);
        } else {
            it->offset += requested;
            it->size -= requested;
        }
        return offset;
    }
    const size_t offset = peak_memory;
    peak_memory += requested;
    return offset;
}

} // namespace

std::vector<GraphNode*> GraphCompiler::compile(ComputationGraph& graph) {
    graph.materialize_mutation_fixups();

    const size_t node_count = graph.nodes.size();
    std::unordered_map<GraphNode*, size_t> node_index;
    std::unordered_map<Tensor*, GraphNode*> producer;
    std::vector<std::unordered_set<size_t>> successors(node_count);
    std::vector<int> in_degree(node_count, 0);

    for (size_t i = 0; i < node_count; ++i) {
        GraphNode* node = graph.nodes[i].get();
        if (!node) throw std::runtime_error("graph contains a null node");
        node_index[node] = i;
        for (Tensor* input : node->inputs) {
            if (!input) throw std::runtime_error("graph node contains a null input tensor");
        }
        for (Tensor* output : node->outputs) {
            if (!output) throw std::runtime_error("graph node contains a null output tensor");
            if (std::find(node->inputs.begin(), node->inputs.end(), output) != node->inputs.end()) {
                throw std::runtime_error(
                    "graph contains an input/output self-alias; create a mutation version before compiling");
            }
            auto [it, inserted] = producer.emplace(output, node);
            if (!inserted && it->second != node) {
                throw std::runtime_error("a logical tensor value has more than one producer");
            }
        }
    }

    for (size_t consumer_index = 0; consumer_index < node_count; ++consumer_index) {
        GraphNode* consumer = graph.nodes[consumer_index].get();
        std::unordered_set<size_t> predecessors;
        for (Tensor* input : consumer->inputs) {
            auto producer_it = producer.find(input);
            if (producer_it == producer.end()) continue;
            const size_t producer_index = node_index.at(producer_it->second);
            if (producer_index == consumer_index) {
                throw std::runtime_error("graph contains a self dependency after functionalization");
            }
            predecessors.insert(producer_index);
        }
        in_degree[consumer_index] = static_cast<int>(predecessors.size());
        for (size_t predecessor : predecessors) {
            successors[predecessor].insert(consumer_index);
        }
    }

    std::priority_queue<size_t, std::vector<size_t>, std::greater<size_t>> ready;
    for (size_t i = 0; i < node_count; ++i) {
        if (in_degree[i] == 0) ready.push(i);
    }

    std::vector<GraphNode*> order;
    order.reserve(node_count);
    while (!ready.empty()) {
        const size_t index = ready.top();
        ready.pop();
        order.push_back(graph.nodes[index].get());
        for (size_t successor : successors[index]) {
            if (--in_degree[successor] == 0) ready.push(successor);
        }
    }

    if (order.size() != node_count) {
        throw std::runtime_error("graph contains a cycle or an unresolved dependency");
    }

    std::unordered_map<Tensor*, TensorLife> life;
    std::unordered_set<Tensor*> consumed;
    for (size_t i = 0; i < order.size(); ++i) {
        GraphNode* node = order[i];
        for (Tensor* output : node->outputs) {
            TensorLife& item = life[output];
            if (item.start == -1) item.start = static_cast<int>(i);
            item.end = static_cast<int>(i);
        }
        for (Tensor* input : node->inputs) {
            TensorLife& item = life[input];
            if (item.start == -1) item.start = static_cast<int>(i);
            item.end = static_cast<int>(i);
            consumed.insert(input);
        }
    }

    for (const auto& tensor_ptr : graph.tensors) {
        Tensor* tensor = tensor_ptr.get();
        const bool has_producer = producer.count(tensor) != 0;
        const bool has_consumer = consumed.count(tensor) != 0;
        if (!has_producer && tensor->data == nullptr) {
            throw std::runtime_error(
                "graph input tensor must be bound with create_tensor_from_ptr before compiling");
        }
        if (has_producer && !has_consumer && tensor->data == nullptr &&
            !graph.is_mutation_version(tensor)) {
            throw std::runtime_error(
                "graph output tensor must be bound with create_tensor_from_ptr before compiling");
        }
    }

    const bool debug_graph = env_flag("LLM_DEBUG_GRAPH");
    if (debug_graph) {
        std::cout << "\n=== Graph order and tensor liveness ===\n";
        for (size_t i = 0; i < order.size(); ++i) {
            std::cout << "node[" << i << "] @" << order[i] << "\n";
        }
        for (const auto& [tensor, item] : life) {
            std::cout << "Tensor@" << tensor << " : [" << item.start
                      << ", " << item.end << "]\n";
        }
    }

    std::unordered_map<Tensor*, size_t> tensor_group;
    std::vector<StorageGroup> groups;
    groups.reserve(life.size());

    auto create_external_group = [&](Tensor* tensor) -> size_t {
        StorageGroup group;
        group.external = true;
        group.external_data = tensor->data;
        group.external_owner = tensor;
        group.owner_owns_data = tensor->owns_data;
        group.size = align_up_64(tensor->bytes());
        auto life_it = life.find(tensor);
        group.end = life_it == life.end() ? -1 : life_it->second.end;
        groups.push_back(group);
        const size_t id = groups.size() - 1;
        tensor_group[tensor] = id;
        return id;
    };

    for (const auto& [tensor, item] : life) {
        if (tensor->data != nullptr) create_external_group(tensor);
    }

    size_t peak_memory = 0;
    std::vector<FreeBlock> free_blocks;

    for (size_t step = 0; step < order.size(); ++step) {
        GraphNode* node = order[step];
        const int current_step = static_cast<int>(step);
        std::unordered_map<size_t, size_t> alias_input_for_output;
        for (const InplaceAliasCandidate& candidate : node->inplace_alias_candidates()) {
            if (candidate.output_index >= node->outputs.size() ||
                candidate.input_index >= node->inputs.size()) {
                throw std::runtime_error("node returned an invalid in-place alias candidate");
            }
            alias_input_for_output.emplace(candidate.output_index, candidate.input_index);
        }

        for (size_t output_index = 0; output_index < node->outputs.size(); ++output_index) {
            Tensor* output = node->outputs[output_index];
            if (tensor_group.count(output) != 0) continue;

            bool aliased = false;
            auto alias_it = alias_input_for_output.find(output_index);
            if (alias_it != alias_input_for_output.end()) {
                Tensor* input = node->inputs[alias_it->second];
                auto input_group_it = tensor_group.find(input);
                if (input_group_it != tensor_group.end() &&
                    life[input].end == current_step &&
                    tensors_are_alias_compatible(input, output)) {
                    StorageGroup& group = groups[input_group_it->second];
                    if (!group.released && group.end <= current_step) {
                        tensor_group[output] = input_group_it->second;
                        group.end = std::max(group.end, life[output].end);
                        aliased = true;
                    }
                }
            }
            if (aliased) continue;

            StorageGroup group;
            group.size = align_up_64(output->bytes());
            group.offset = allocate_first_fit(group.size, free_blocks, peak_memory);
            group.end = life[output].end;
            groups.push_back(group);
            tensor_group[output] = groups.size() - 1;
        }

        std::unordered_set<size_t> input_groups;
        for (Tensor* input : node->inputs) {
            auto group_it = tensor_group.find(input);
            if (group_it == tensor_group.end()) {
                throw std::runtime_error("tensor storage is unavailable before its consumer executes");
            }
            input_groups.insert(group_it->second);
        }
        for (size_t group_id : input_groups) {
            StorageGroup& group = groups[group_id];
            if (!group.external && !group.released && group.end == current_step) {
                free_blocks.push_back({group.offset, group.size});
                group.released = true;
            }
        }
        coalesce_free_blocks(free_blocks);
    }

    if (graph.arena_buffer != nullptr) {
        throw std::runtime_error("computation graph has already been compiled");
    }
    if (peak_memory > 0) {
        if (!g_memory_pool) {
            throw std::runtime_error("global memory pool is not initialized");
        }
        graph.arena_buffer = g_memory_pool->allocate(peak_memory);
        if (!graph.arena_buffer) {
            throw std::runtime_error("failed to allocate graph arena");
        }
        graph.arena_size = peak_memory;
    }

    for (const auto& [tensor, group_id] : tensor_group) {
        const StorageGroup& group = groups[group_id];
        if (group.external) {
            tensor->data = group.external_data;
            tensor->owns_data = group.owner_owns_data && group.external_owner == tensor;
        } else {
            tensor->data = static_cast<char*>(graph.arena_buffer) + group.offset;
            tensor->owns_data = false;
        }
    }

    if (debug_graph) {
        std::cout << "[Memory Planner] Peak Workspace Required: "
                  << peak_memory / 1024.0 / 1024.0 << " MB\n";
    }
    return order;
}

Status GraphRuntime::run(const std::vector<GraphNode*>& plan) {
    for (GraphNode* node : plan) {
        if (!node) return Status::INVALID_ARGUMENT;
        Status status = node->forward();
        if (status != Status::SUCCESS) return status;
    }
    return Status::SUCCESS;
}

} // namespace llm_engine
