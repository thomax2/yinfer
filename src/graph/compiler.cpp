#include "llm_engine/graph/compiler.h"

#include <unordered_map>
#include <unordered_set>
#include <queue>
#include <iostream>

namespace llm_engine {

struct TensorLife {
    int start = -1;
    int end = -1;
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

    // ===== 3.5 自动推导并保护 Input 和 Output (核心新增保护逻辑) =====
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


    // ===== 4. 执行期释放策略（关键优化） =====
    // 思路：当执行到 step i 时，把所有 end == i 的 tensor 释放

    // 我们把释放逻辑嵌入 execution plan
    // 用一个 side table 记录

    std::unordered_map<int, std::vector<Tensor*>> free_table;

    for(auto& kv : life) {
        Tensor* t = kv.first;
        int end_step = kv.second.end;

        // 【关键过滤】只有不在保护名单里的 Tensor，才会被加入释放计划
        if (protected_tensors.count(t) == 0) {
            free_table[end_step].push_back(t);  // 结束位置 end，存放需要释放的 tensor 指针。
        }
    }


    // ===== 5. 包装执行计划（加释放hook） =====

    std::vector<GraphNode*> optimized_plan;

    for(int i = 0; i < order.size(); i++) {

        optimized_plan.push_back(order[i]);

        // 插入一个“释放节点”
        if(free_table.count(i)) {       // 是否存在 key = i 的条目
            auto tensors = free_table[i];

            /*
                用 lambda 包装一个 fake node
                函数内部的局部类
            */ 
            class FreeNode : public GraphNode {         // 创建一个 node，专门负责释放 tensor
            public:
                std::vector<Tensor*> tensors;

                FreeNode(const std::vector<Tensor*>& ts)
                    : tensors(ts) {}

                Status forward() override {
                    for(auto* t : tensors) {
                        if(t->data && t->owns_data) {

                            g_memory_pool->free_block(t->data);
                            t->data = nullptr;

                            // Debug
                            // std::cout << "FREE Tensor@" << t << "\n";
                        }
                    }
                    return Status::SUCCESS;
                }
                bool is_temporary() const override { return true; }
            };

            optimized_plan.push_back(new FreeNode(tensors));
        }
    }

    return optimized_plan;
}

Status GraphRuntime::run(const std::vector<GraphNode*>& plan){
    for(auto* node : plan) {
        auto s = node->forward();
        if(s != Status::SUCCESS) {
            return s;
        }
    }

    // 👇 执行完后释放临时节点
    for(auto* node : plan) {
        if(node->is_temporary())
            delete node;
    }

    return Status::SUCCESS;
}

}