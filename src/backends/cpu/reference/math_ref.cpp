#include "math_ref.h"

#include <limits>
#include <cmath>

namespace llm_engine {
namespace reference {

Status add_ref(
    const Tensor& A,
    const Tensor& B,
    Tensor& C
) {

    if (A.size() != B.size() || A.size() != C.size())
        return Status::SHAPE_MISMATCH;

    float* a = (float*)A.data;
    float* b = (float*)B.data;
    float* c = (float*)C.data;

    for (size_t i = 0; i < A.size(); ++i) {
        c[i] = a[i] + b[i];
    }

    return Status::SUCCESS;
}


Status matmul_ref(
    const Tensor& A,
    const Tensor& B,
    Tensor& C
) {

    if (A.shape.size() != 2 || B.shape.size() != 2)
        return Status::INVALID_ARGUMENT;

    int M = A.shape[0];
    int K = A.shape[1];
    int K2 = B.shape[0];
    int N = B.shape[1];

    if (K != K2)
        return Status::SHAPE_MISMATCH;

    float* a = (float*)A.data;
    float* b = (float*)B.data;
    float* c = (float*)C.data;

    for (int i = 0; i < M; i++) {

        for (int j = 0; j < N; j++) {

            float sum = 0;

            for (int k = 0; k < K; k++) {

                sum += a[i*K + k] * b[k*N + j];

            }

            c[i*N + j] = sum;

        }
    }

    return Status::SUCCESS;
}

/*
    input: [batch_size, seq_len]
    output: [batch_size, seq_len]

    减去最大值原因：

    最大值太大会导致 exp 结果溢出float最大值，而减去最大值后，指数项 <= 0，exp 结果在 (0, 1] 之间，绝不会溢出。
    exp(x_i) / Σ exp(x_j) = exp(x_i - M) / Σ exp(x_j - M)
*/
void softmax_ref(const float* input, float* output, int batch_size, int seq_len) {
    for (int b = 0; b < batch_size; ++b) {
        // 定位到当前处理的这一行
        const float* in_row = input + b * seq_len;
        float* out_row = output + b * seq_len;

        // 1. 寻找当前行的最大值 (Safe 技巧核心)
        float max_val = std::numeric_limits<float>::lowest();
        for (int i = 0; i < seq_len; ++i) {
            if (in_row[i] > max_val) {
                max_val = in_row[i];
            }
        }

        // 2. 减去最大值，计算 exp，并累加求和
        float sum_exp = 0.0f;
        for (int i = 0; i < seq_len; ++i) {
            // in_row[i] - max_val 确保指数项 <= 0，exp 结果在 (0, 1] 之间，绝不会溢出
            out_row[i] = std::exp(in_row[i] - max_val);
            sum_exp += out_row[i];
        }

        // 3. 归一化，得到最终的概率分布
        for (int i = 0; i < seq_len; ++i) {
            out_row[i] /= sum_exp;
        }
    }
}

void gemm_ref(const float* A, const float* B, float* C, 
              int m, int n, int k, 
              bool transA, bool transB) {
    
    // 遍历结果矩阵 C 的每一行
    for (int i = 0; i < m; ++i) {
        // 遍历结果矩阵 C 的每一列
        for (int j = 0; j < n; ++j) {
            float sum = 0.0f;
            
            // 遍历中间维度 k 进行点积累加
            for (int p = 0; p < k; ++p) {
                // 读取 A 矩阵的元素
                // 如果 transA 为 true，相当于 A 是 [k, m] 的形状，坐标为 (p, i)
                // 如果 transA 为 false，A 是 [m, k] 的形状，坐标为 (i, p)
                float a_val = transA ? A[p * m + i] : A[i * k + p];
                
                // 读取 B 矩阵的元素
                // 如果 transB 为 true，相当于 B 是 [n, k] 的形状，坐标为 (j, p)
                // 如果 transB 为 false，B 是 [k, n] 的形状，坐标为 (p, j)
                float b_val = transB ? B[j * k + p] : B[p * n + j];
                
                sum += a_val * b_val;
            }
            // 将计算结果写入 C 矩阵的 (i, j) 位置
            C[i * n + j] = sum;
        }
    }
}

void bmm_ref(const float* A, const float* B, float* C, 
             int batch, int m, int n, int k, 
             bool transA, bool transB) {
    
    // 计算单个 Batch 的元素数量 (步长)
    // 如果 A 被转置了，那么它在内存里存的其实是 k * m，但元素总量还是 m * k
    int stride_A = m * k;
    int stride_B = k * n;
    int stride_C = m * n;

    for (int b = 0; b < batch; ++b) {
        // 定位到当前 Batch 的指针
        const float* a_ptr = A + b * stride_A;
        const float* b_ptr = B + b * stride_B;
        float* c_ptr = C + b * stride_C;

        // 这里调用你原本写好的普通单次 gemm_ref。
        // （注意：你需要确保你的 gemm_ref 内部正确处理了 transB=true 的情况，
        // 即读取 B 矩阵时的索引从 B[i * N + j] 变为 B[j * K + i]）
        gemm_ref(a_ptr, b_ptr, c_ptr, m, n, k, transA, transB);
    }
}

} // reference
} // llm_engine