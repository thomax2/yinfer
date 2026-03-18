#include "llm_engine/graph/graph.h"
#include "backends/cpu/reference/math_ref.h"

namespace llm_engine {

using namespace reference;

// 类名::函数名(参数列表) : 初始化列表
MatmulNode::MatmulNode(Tensor* A, Tensor* B, Tensor* C) {
    inputs = {A, B};
    outputs = {C};
}

Status MatmulNode::forward() {
    return matmul_ref(*inputs[0], *inputs[1], *outputs[0]);
}

AddNode::AddNode(Tensor* A, Tensor* B, Tensor* C) {
    inputs = {A, B};
    outputs = {C};
}

Status AddNode::forward() {
    return add_ref(*inputs[0], *inputs[1], *outputs[0]);
}

Tensor* ComputationGraph::create_tensor(
    const std::vector<int>& shape,
    DataType dtype
) {
    // 创建一个 unique_ptr<Tensor> ,所有权属于 tensors, 当ComputationGraph被销毁时，所有的Tensor也会被自动销毁。
    tensors.push_back(std::make_unique<Tensor>(shape, dtype));
    return tensors.back().get();
}

MatmulNode* ComputationGraph::add_matmul(
    Tensor* A,
    Tensor* B,
    Tensor* C
) {
    nodes.push_back(std::make_unique<MatmulNode>(A, B, C));

    // static_cast 把通用的基类指针，变回具体的子类指针。
    return static_cast<MatmulNode*>(nodes.back().get());
}

AddNode* ComputationGraph::add_add(
    Tensor* A,
    Tensor* B,
    Tensor* C
) {
    nodes.push_back(std::make_unique<AddNode>(A, B, C));
    return static_cast<AddNode*>(nodes.back().get());
}



}