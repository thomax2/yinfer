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
    virtual bool is_temporary() const { return false; }
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

class RMSNormNode : public GraphNode {
public:
    float eps;
    RMSNormNode(Tensor* X, Tensor* Weight, Tensor* Y, float eps);
    Status forward() override;
};

class SwiGLUNode : public GraphNode {
public:
    SwiGLUNode(Tensor* Gate, Tensor* Up, Tensor* Y);
    Status forward() override;
};

class RoPENode : public GraphNode {
public:
    // 注意：你写的 RoPE 是 in-place 原地修改的，所以输入输出都是 X
    RoPENode(Tensor* X, Tensor* Cos, Tensor* Sin);
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

    Tensor* create_tensor_from_ptr(
        const std::vector<int>& shape,
        void* data,
        DataType dtype = DataType::FP32
    );

    MatmulNode*     add_matmul(Tensor* A, Tensor* B, Tensor* C);
    AddNode*        add_add(Tensor* A, Tensor* B, Tensor* C);

    RMSNormNode*    add_rmsnorm(Tensor* X, Tensor* Weight, Tensor* Y, float eps);
    SwiGLUNode*     add_swiglu(Tensor* Gate, Tensor* Up, Tensor* Y);
    RoPENode*       add_rope(Tensor* X, Tensor* Cos, Tensor* Sin);
};

}