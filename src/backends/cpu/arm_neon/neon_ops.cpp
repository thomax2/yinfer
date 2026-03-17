#include "neon_ops.h"
#include "pack.h"
#include "gemm.h"
#include "kernel_common.h"
#include <arm_neon.h>

#include <vector>

namespace llm_engine {
namespace arm_neon {

/*
C = A * B + C
*/
Status matmul_neon(
    const Tensor& A,
    const Tensor& B,
    Tensor& C,
    float* workspace
) {
    // 目前仅支持FP32的矩阵乘法
    if (A.dtype != DataType::FP32 || B.dtype != DataType::FP32 || C.dtype != DataType::FP32)
        return Status::INVALID_ARGUMENT;

    // 目前仅支持CPU设备
    if (A.device != DeviceType::CPU || B.device != DeviceType::CPU || C.device != DeviceType::CPU)
        return Status::UNSUPPORTED_DEVICE;

    int M = A.shape[0];
    int K = A.shape[1];
    int N = B.shape[1];

    const float* a = A.ptr<float>();
    const float* b = B.ptr<float>();
    float* c = C.ptr<float>();

    int mp = (M + MR - 1) / MR;
    int np = (N + NR - 1) / NR;

    float* A_pack = workspace;
    float* B_pack = workspace + mp * MR * K;

    pack_A(a, A_pack, M, K, K);
    pack_B(b, B_pack, K, N, N);

    for(int i = 0; i < mp; i++) {
        for(int j = 0; j < np; j++) {
            const float* Ap = A_pack + i*MR*K;
            const float* Bp = B_pack + j*NR*K;

            // 计算当前块实际需要写入的行数和列数
            int actual_m = std::min(MR, M - i * MR);
            int actual_n = std::min(NR, N - j * NR);

            if (actual_m == MR && actual_n == NR) {
                // 完整块，直接安全写入 C
                float* Cp = c + i*MR*N + j*NR;
                gemm_kernel_8x12_neon(Ap, Bp, Cp, K, N);
            } else {
                // 边缘块，使用临时 Buffer
                float temp_C[MR * NR] = {0}; // 大小固定为 MR * NR
                // 注意：此时传给内核的 stride (N) 需要变成 NR，因为是在写 temp_C
                gemm_kernel_8x12_neon(Ap, Bp, temp_C, K, NR); 

                // 只把有效的部分拷贝回真实的 C 中
                for (int r = 0; r < actual_m; r++) {
                    for (int c_col = 0; c_col < actual_n; c_col++) {
                        c[(i * MR + r) * N + (j * NR + c_col)] = temp_C[r * NR + c_col];
                    }
                }
            }
        }
    }
    return Status::SUCCESS;
}

void add_neon(const Tensor& A, const Tensor& B, Tensor& C) {
    const float* a_ptr = A.ptr<float>();
    const float* b_ptr = B.ptr<float>();
    float* c_ptr = C.ptr<float>();
    
    int total_elements = A.size();
    int i = 0;

    // 主循环：每次处理 4 个 float32 (刚好塞满一个 128-bit NEON 寄存器)
    for (; i <= total_elements - 4; i += 4) {
        // 从内存加载 4 个 float
        float32x4_t va = vld1q_f32(a_ptr + i);
        float32x4_t vb = vld1q_f32(b_ptr + i);
        
        // 向量加法
        float32x4_t vc = vaddq_f32(va, vb);
        
        // 写回内存
        vst1q_f32(c_ptr + i, vc);
    }

    // 处理尾部 (Tail/Fringe)
    // 如果元素总数不是 4 的倍数，剩下的一两个数字用普通 C++ 算完
    for (; i < total_elements; ++i) {
        c_ptr[i] = a_ptr[i] + b_ptr[i];
    }
}

}
}