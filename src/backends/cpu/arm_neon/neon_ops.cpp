#include "neon_ops.h"
#include "pack.h"
#include "gemm.h"
#include "kernel_common.h"

#include <vector>

namespace llm_engine {
namespace arm_neon {

/*
C = A * B + C
*/
Status matmul_neon(
    const Tensor& A,
    const Tensor& B,
    Tensor& C
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

    std::vector<float> A_pack(mp * MR * K);
    std::vector<float> B_pack(np * NR * K);

    pack_A(a, A_pack.data(), M, K, K);
    pack_B(b, B_pack.data(), K, N, N);

    for(int i = 0; i < mp; i++) {
        for(int j = 0; j < np; j++) {
            const float* Ap = A_pack.data() + i*MR*K;
            const float* Bp = B_pack.data() + j*NR*K;

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

}
}