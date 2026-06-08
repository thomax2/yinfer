#include "llm_engine/graph/compiler.h"

#include <unordered_map>
#include <unordered_set>
#include <queue>
#include <iostream>
#include <algorithm>
#include <cstdlib>
#include <string>

namespace llm_engine {

namespace {
bool env_flag(const char* name) {
    const char* v = std::getenv(name);
    if (!v) return false;
    return std::string(v) == "1" ||
           std::string(v) == "true" ||
           std::string(v) == "TRUE" ||
           std::string(v) == "on" ||
           std::string(v) == "ON";
}
} // namespace

struct TensorLife {
    int start = -1;
    int end = -1;
};

struct MemoryBlock {
    size_t offset;
    size_t size;
};

std::vector<GraphNode*> GraphCompiler::compile(ComputationGraph& graph) {

    std::vector<GraphNode*> order;
    for (auto& node : graph.nodes) {
        order.push_back(node.get());
    }
    /* 
        目前：order 中的节点顺序就是一个合法的执行顺序，保证了每个节点的输入在它之前被计算出来了。
        但只是个一维的，不能并行计算。
    */


    // ===== 2. 生命周期分析（核心新增） =====
    std::unordered_map<Tensor*, TensorLife> life;

    for(int i = 0; i < order.size(); i++) {
        auto* node = order[i];

        for(auto* t : node->outputs) {
            if(life[t].start == -1)
                life[t].start = i; // 第一次被生产出来
            life[t].end = i;   // 每次被生产出来都更新结束位置
        }

        for(auto* t : node->inputs) {
            if(life[t].start == -1)
                life[t].start = i; // 第一次被使用
            life[t].end = i;   // 每次被使用都更新结束位置
        }
    }

    bool debug_graph = env_flag("LLM_DEBUG_GRAPH");
    // ===== 3. 打印生命周期（debug用） =====
    if (debug_graph) {
        std::cout << "\n=== Tensor Liveness ===\n";
    }

    for(auto& kv : life) {
        auto* t = kv.first;
        auto& l = kv.second;

        if (debug_graph) {
            std::cout << "Tensor@" << t
                      << " : [" << l.start
                      << ", " << l.end << "]\n";
        }
    }

    // ===== 3.1 自动推导并保护 Input 和 Output (核心新增保护逻辑) =====
    std::unordered_set<Tensor*> all_produced;
    std::unordered_set<Tensor*> all_consumed;

    // 收集所有被生产过和被消费过的 Tensor
    for(const auto& node : graph.nodes) {
        for(auto* t : node->outputs) {
            all_produced.insert(t);
        }
        for(auto* t : node->inputs) {
            all_consumed.insert(t);
        }
    }

    std::unordered_set<Tensor*> protected_tensors;

    // 遍历图中所有的 Tensor，进行身份判定
    for(const auto& t_ptr : graph.tensors) {
        Tensor* t = t_ptr.get();

        bool is_produced = all_produced.count(t) > 0;
        bool is_consumed = all_consumed.count(t) > 0;

        // 规则 1：基于拓扑的边界推导 (没有生产者，或没有消费者)
        if (!is_produced || !is_consumed) {
            protected_tensors.insert(t);
        }

        // 规则 2：只要是用户显式绑定的外部内存，强制保护！
        if (t->data != nullptr && t->owns_data == false) {
            protected_tensors.insert(t);
        }

        // 统一安全检查：边界张量必须绑了物理内存！
        if (protected_tensors.count(t) > 0 && t->data == nullptr) {
            throw std::runtime_error("Fatal: The boundary tensor (Input/Output) must be bound to external memory via create_tensor_from_ptr before compiling!");
        }
    }


    // ===================================================================
    // 4. 工业级内存规划 (Static Memory Arena Planner)
    // ===================================================================
    
    // 辅助函数：64 字节 cache-line 对齐
    auto align_size = [](size_t size) -> size_t {
        return (size + 63) & ~63;
    };

    size_t peak_memory = 0; 
    std::unordered_map<Tensor*, size_t> tensor_offsets; // 每个张量在内存池中的起始位置
    std::unordered_set<Tensor*> freed_tensors;          // 记录释放，绝不 erase

    struct FreeBlock { size_t offset; size_t size; };
    std::vector<FreeBlock> free_blocks;                 // 空闲块列表。记录当前可用的内存碎片（起始位置和大小），用于内存复用。

    // 模拟时间线推演：从第 0 步走到最后一步
    for (int i = 0; i < order.size(); i++) {
        GraphNode* node = order[i];
        
        // 先分配 Outputs，在inputs分配中检查是否分配，没分配才分配，防止读写踩踏 (In-place hazard)
        for (auto* t : node->outputs) {
            
            // 1. 受保护的边界张量，直接跳过 (不参与 Arena 排版)
            if (protected_tensors.count(t) > 0) {
                continue; 
            }

            // 2. 防止 In-place 算子多分配
            if (tensor_offsets.count(t) > 0) {
                continue; 
            }

            size_t req_size = align_size(t->bytes());
            bool allocated = false;
            
            // First-Fit 寻找空闲块
            for (auto it = free_blocks.begin(); it != free_blocks.end(); ++it) {
                if (it->size >= req_size) {
                    tensor_offsets[t] = it->offset; 
                    if (it->size == req_size) {
                        free_blocks.erase(it); 
                    } else {
                        it->offset += req_size; 
                        it->size -= req_size;
                    }
                    allocated = true;
                    break;
                }
            }
            
            // 扩建边界
            if (!allocated) {
                tensor_offsets[t] = peak_memory;
                peak_memory += req_size;
            }
        }

        // 【B】Outputs 分配完后，再回收这一步死掉的 Inputs
        for (auto* t : node->inputs) {
            if (life[t].end == i && protected_tensors.count(t) == 0) {
                if (tensor_offsets.count(t) && freed_tensors.count(t) == 0) {
                    size_t offset = tensor_offsets[t];
                    size_t size = align_size(t->bytes()); 
                    free_blocks.push_back({offset, size});
                    
                    freed_tensors.insert(t); // 标记为已释放
                }
            }
        }

        // 碎片合并：排序并合并相邻空闲块
        if (!free_blocks.empty()) {
            std::sort(free_blocks.begin(), free_blocks.end(), [](const FreeBlock& a, const FreeBlock& b) {
                return a.offset < b.offset;
            });
            for (size_t fb = 0; fb + 1 < free_blocks.size(); ) {
                if (free_blocks[fb].offset + free_blocks[fb].size == free_blocks[fb+1].offset) {
                    free_blocks[fb].size += free_blocks[fb+1].size;
                    free_blocks.erase(free_blocks.begin() + fb + 1);
                } else {
                    fb++;
                }
            }
        }
    }

    if (debug_graph) {
        std::cout << "\n[Memory Planner] Peak Workspace Required: " << peak_memory / 1024.0 / 1024.0 << " MB\n";
    }

    // ===================================================================
    // 5. 零分配运行态准备 (Zero-Allocation Binding)
    // ===================================================================
    
    // 一次性申请连续大内存
    if (peak_memory > 0) {
        // (注意：这里假设你在 ComputationGraph 中加了 arena_size 和 arena_buffer 成员)
        graph.arena_size = peak_memory;
        graph.arena_buffer = g_memory_pool->allocate(peak_memory);
    }

    // 开始遍历图中的所有张量，分配推演好的位置
    for (const auto& t_ptr : graph.tensors) {
        Tensor* t = t_ptr.get();
        
        // 只要在档案本里，说明它是需要复用 Arena 的临时变量
        if (tensor_offsets.count(t) > 0) {
            
            // 如果它之前私自占用了内存（比如 Eager 模式遗留），乖乖退还给池子
            if (t->data != nullptr && t->owns_data) {
                g_memory_pool->free_block(t->data);
            }

            // 确定位置，管理所有权
            size_t offset = tensor_offsets[t];
            t->data = static_cast<char*>(graph.arena_buffer) + offset;
            t->owns_data = false; 
        }
    }

    // 完美竣工！
    return order;
}

Status GraphRuntime::run(const std::vector<GraphNode*>& plan){
    for(auto* node : plan) {
        auto s = node->forward();
        if(s != Status::SUCCESS) {
            return s;
        }
    }
    return Status::SUCCESS;}
}
