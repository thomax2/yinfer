#pragma once

#include "llm_engine/tensor.h"
#include "llm_engine/status.h"

namespace llm_engine {
namespace reference {

Status add_ref(
    const Tensor& A,
    const Tensor& B,
    Tensor& C
);

Status matmul_ref(
    const Tensor& A,
    const Tensor& B,
    Tensor& C
);

void softmax_ref(
    const float* input, 
    float* output, 
    int batch_size, 
    int seq_len
);

void bmm_ref(const float* A, const float* B, float* C, 
             int batch, int m, int n, int k, 
             bool transA, bool transB);


// 【新增】基础的单次通用矩阵乘法 (GEMM)
// 计算 C = A * B
// m: C 的行数 (如果 A 不转置，也是 A 的行数)
// n: C 的列数 (如果 B 不转置，也是 B 的列数)
// k: A 的列数 / B 的行数 (中间维度)
void gemm_ref(const float* A, const float* B, float* C, 
              int m, int n, int k, 
              bool transA, bool transB);

} // reference
} // llm_engine