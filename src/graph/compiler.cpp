#include "llm_engine/compiler.h"

#include <unordered_map>
#include <queue>

namespace llm_engine {


std::vector<GraphNode*> GraphCompiler::compile(ComputationGraph& graph) {
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
    for(auto &node : graph.nodes) {
        for(auto* t : node->inputs) {
            if(producers.count(t) > 0) {
                in_degree[node.get()]++;
            }
        }
        for(auto* t : node->outputs) {
            producers[t].push_back(node.get());
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
                        if(in_degree[next.get()] == 0) {
                            q.push(next.get());
                        }
                    }
                }
            }
        }
    }

    /* 
        order 中的节点顺序就是一个合法的执行顺序，保证了每个节点的输入在它之前被计算出来了。
        但只是个一维的，不能并行计算。 
    */
    return order;
}

Status GraphRuntime::run(const std::vector<GraphNode*>& plan){
    for(auto* node : plan) {
        auto s = node->forward();
        if(s != Status::SUCCESS) {
            return s;
        }
    }
    return Status::SUCCESS;
}

}