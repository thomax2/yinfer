#pragma once

#include <memory>
#include <vector>

#include "llm_engine/tensor.h"
#include "llm_engine/status.h"

namespace llm_engine {

class GraphNode{
public:
    std::vector<Tensor*> inputs;
    std::vector<Tensor*> outputs;

    virtual Status forward() = 0;
    virtual ~GraphNode() = default;
};


class MatmulNode : public GraphNode {
public:
    MatmulNode(Tensor* A, Tensor* B, Tensor* C);

    Status forward() override;  // 重写基类的纯虚函数，在cpp中完成具体的实现。
};

class AddNode : public GraphNode {
public:
    AddNode(Tensor* A, Tensor* B, Tensor* C);
    
    Status forward() override;
};

class ComputationGraph {
public:
    std::vector<std::unique_ptr<GraphNode>> nodes;
    std::vector<std::unique_ptr<Tensor>> tensors;

public:
    
    Tensor* create_tensor(
        const std::vector<int>& shape,
        DataType dtype = DataType::FP32
    );

    MatmulNode* add_matmul(
        Tensor* A,
        Tensor* B,
        Tensor* C
    );

    AddNode* add_add(
        Tensor* A,
        Tensor* B,
        Tensor* C
    );
};

}