#include "llm_engine/graph/graph.h"
#include "backends/cpu/reference/math_ref.h"
#include "backends/cpu/arm_neon/neon_ops.h"
#include "backends/cpu/arm_neon/kernel_common.h"

namespace llm_engine {

using namespace reference;
using namespace arm_neon;
/*                                   定义 node                                          */
// 类名::函数名(参数列表) : 初始化列表
MatmulNode::MatmulNode(Tensor* A, Tensor* B, Tensor* C) {
    inputs = {A, B};
    outputs = {C};
}

Status MatmulNode::forward() {
    Tensor* A = inputs[0];
    Tensor* B = inputs[1];
    Tensor* C = outputs[0];

    int M = A->shape[0];
    int K = A->shape[1];
    int N = B->shape[1];

    // 1. 计算需要的 Workspace 大小
    int mp = (M + MR - 1) / MR;
    int np = (N + NR - 1) / NR;
    size_t ws_size = (mp * MR * K + np * NR * K) * sizeof(float);

    // 2. 从全局内存池临时申请 Workspace
    float* workspace = (float*)g_memory_pool->allocate(ws_size);

    // 3. 执行 NEON 矩阵乘法
    Status status = arm_neon::matmul_neon(*A, *B, *C, workspace, false, nullptr);

    // 4. 执行完毕，立刻释放临时内存，防止显存/内存泄漏！
    g_memory_pool->free_block(workspace);

    return status;
}


AddNode::AddNode(Tensor* A, Tensor* B, Tensor* C) {
    inputs = {A, B};
    outputs = {C};
}

Status AddNode::forward() {
    // 调用你手写的 NEON add 算子
    add_neon(*inputs[0], *inputs[1], *outputs[0]);
    return Status::SUCCESS;
}

// --- RMSNorm ---
RMSNormNode::RMSNormNode(Tensor* X, Tensor* Weight, Tensor* Y, float eps) : eps(eps) {
    inputs = {X, Weight};
    outputs = {Y};
}

Status RMSNormNode::forward() {
    int n = inputs[0]->shape.back(); // 获取最后一个维度
    arm_neon::rmsnorm_neon(
        inputs[0]->ptr<float>(),
        inputs[1]->ptr<float>(),
        outputs[0]->ptr<float>(),
        n, eps
    );
    return Status::SUCCESS;
}

// --- SwiGLU ---
SwiGLUNode::SwiGLUNode(Tensor* Gate, Tensor* Up, Tensor* Y) {
    inputs = {Gate, Up};
    outputs = {Y};
}

Status SwiGLUNode::forward() {
    int n = inputs[0]->size();
    arm_neon::swiglu_neon(
        inputs[0]->ptr<float>(),
        inputs[1]->ptr<float>(),
        outputs[0]->ptr<float>(),
        n
    );
    return Status::SUCCESS;
}

// --- RoPE ---
RoPENode::RoPENode(Tensor* X, Tensor* Cos, Tensor* Sin) {
    inputs = {X, Cos, Sin};
    outputs = {X}; // In-place 修改
}

Status RoPENode::forward() {
    int n = inputs[0]->shape.back();
    arm_neon::rope_neon(
        inputs[0]->ptr<float>(),
        inputs[1]->ptr<float>(),
        inputs[2]->ptr<float>(),
        n
    );
    return Status::SUCCESS;
}


/*                                       创建 tensor                                         */
Tensor* ComputationGraph::create_tensor(
    const std::vector<int>& shape,
    DataType dtype
) {
    // 创建一个 unique_ptr<Tensor> ,所有权属于 tensors, 当ComputationGraph被销毁时，所有的Tensor也会被自动销毁。
    tensors.push_back(std::make_unique<Tensor>(shape, dtype));
    return tensors.back().get();
}

Tensor* ComputationGraph::create_tensor_from_ptr(
    const std::vector<int>& shape,
    void* data,
    DataType dtype
) {
    auto t = std::make_unique<Tensor>(shape, dtype);
    // 假设你的 Tensor 类里有 data 和 owns_data 成员
    t->data = data; 
    t->owns_data = false; // 极其关键：告诉引擎不要去 free 它！
    tensors.push_back(std::move(t));
    return tensors.back().get();
}

/*                                      添加 node                                          */
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

RMSNormNode* ComputationGraph::add_rmsnorm(
    Tensor* X,
    Tensor* Weight, 
    Tensor* Y, 
    float eps
) {
    nodes.push_back(std::make_unique<RMSNormNode>(X, Weight, Y, eps));
    return static_cast<RMSNormNode*>(nodes.back().get());
}

SwiGLUNode* ComputationGraph::add_swiglu(Tensor* Gate,
    Tensor* Up, 
    Tensor* Y
) {
    nodes.push_back(std::make_unique<SwiGLUNode>(Gate, Up, Y));
    return static_cast<SwiGLUNode*>(nodes.back().get());
}

RoPENode* ComputationGraph::add_rope(Tensor* X, 
    Tensor* Cos,
    Tensor* Sin
) {
    nodes.push_back(std::make_unique<RoPENode>(X, Cos, Sin));
    return static_cast<RoPENode*>(nodes.back().get());
}


}