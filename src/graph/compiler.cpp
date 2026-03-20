#include "llm_engine/graph/compiler.h"

#include <unordered_map>
#include <unordered_set>
#include <queue>
#include <iostream>
#include <algorithm>

namespace llm_engine {

struct TensorLife {
    int start = -1;
    int end = -1;
};

struct MemoryBlock {
    size_t offset;
    size_t size;
};

std::vector<GraphNode*> GraphCompiler::compile(ComputationGraph& graph) {

    // ===== 1. 拓扑排序（你原来的逻辑） =====
    std::unordered_map<GraphNode*, int> in_degree;
    std::unordered_map<Tensor*, std::vector<GraphNode*>> producers;
    std::queue<GraphNode*> q;
    std::vector<GraphNode*> order;

    // 入度初始为0
    for(auto &node : graph.nodes) {
        in_degree[node.get()] = 0;
    }

    /*
    1. 第一层的输入是没有生产者的，所以它的入度为0，可以第一批就被取出
    2. 第一层的输出被加入到生产者中，而第一层的输出正好是第二层的输入，所以第二层的输入都是有生成者的，
       就有入度了，所以要等第一层的节点全被取出之后，才能被取出
    */
    // 第一趟：先收集所有生产者信息
    for(auto &node : graph.nodes) {
        for(auto* t : node->outputs) {
            producers[t].push_back(node.get());
        }
    }

    // 第二趟：基于完整的生产者信息计算入度
    for(auto &node : graph.nodes) {
        for(auto* t : node->inputs) {
            if(producers.count(t) > 0) {
                in_degree[node.get()]++;
            }
        }
    }


    for(auto& n : in_degree) {
        if(n.second ==0)
            q.push(n.first);
    }


    while(!q.empty()) {
        auto* node = q.front();
        q.pop();
        order.push_back(node);

        // 找到当前节点的输出对应的生产者，入度减1，如果入度为0了，就加入队列
        for(auto* t : node->outputs) {
            for(auto& next : graph.nodes) {
                /*
                    这里一个用 * 指针，因为 outputs 和 inputs 元素都是 Tensor*，获得 Tensor* 

                    一个用 & 引用，因为nodes容器中就是 unique_ptr<GraphNode>，所以 next 是
                    unique_ptr<GraphNode>&，而 next.get() 才是 GraphNode*。
                */ 
                for(auto* in : next->inputs) {
                    if(in == t) {
                        in_degree[next.get()]--;
                        if(in_degree[next.get()] == 0)
                            q.push(next.get());
                    }
                }
            }
        }
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

    // ===== 3. 打印生命周期（debug用） =====
    std::cout << "\n=== Tensor Liveness ===\n";

    for(auto& kv : life) {
        auto* t = kv.first;
        auto& l = kv.second;

        std::cout << "Tensor@" << t
                  << " : [" << l.start
                  << ", " << l.end << "]\n";
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

        if (!is_produced) {
            // 从未被生产过：说明它是图的输入（外部数据）或模型权重
            protected_tensors.insert(t);
        }
        
        if (!is_consumed) {
            // 从未被消费过：说明它是图的最终输出结果，或者是个没用的游离节点
            protected_tensors.insert(t);
        }
    }


    // ===================================================================
    // 4. 工业级内存规划 (Static Memory Arena Planner)
    // ===================================================================
    
    // 辅助函数：16 字节对齐
    auto align_size = [](size_t size) -> size_t {
        return (size + 15) & ~15;
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
            
            // 1. 受保护的全局变量，单独分配真实内存
            if (protected_tensors.count(t) > 0) {
                if (t->data == nullptr) {
                    t->data = g_memory_pool->allocate(t->bytes());
                    std::memset(t->data, 0, t->bytes());
                    t->owns_data = true; 
                }
                continue;
            }

            //【终极修复】：防止 In-place 算子 多分配
            // 如果这个 Tensor 之前已经分配过房间了，绝对不能再分配！
            if (tensor_offsets.count(t) > 0)
                continue; 
                
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

    std::cout << "\n[Memory Planner] Peak Workspace Required: " << peak_memory / 1024.0 / 1024.0 << " MB\n";

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