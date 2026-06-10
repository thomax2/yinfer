#include "neon_ops.h"
#include "pack.h"
#include "gemm.h"
#include "kernel_common.h"
#include <arm_neon.h>

#include <algorithm>
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
    float* workspace,
    bool transB,
    const float* bias
) {
    // 目前仅支持FP32的矩阵乘法
    if (A.dtype != DataType::FP32 || B.dtype != DataType::FP32 || C.dtype != DataType::FP32)
        return Status::INVALID_ARGUMENT;

    // 目前仅支持CPU设备
    if (A.device != DeviceType::CPU || B.device != DeviceType::CPU || C.device != DeviceType::CPU)
        return Status::UNSUPPORTED_DEVICE;

    int M = A.shape[0];
    int K = A.shape[1];

    // ⚠️ 关键：根据 transB 确定 N 和 ldb
    int N, ldb;
    if (!transB) {
        // B 是 K×N, 行优先
        N = B.shape[1];
        ldb = N;  // B 的行宽
        // 可选：验证 B.shape[0] == K
    } else {
        // B 是 N×K, 行优先 (用于计算 Bᵀ)
        N = B.shape[0];  // ⚠️ N 来自 B 的行数
        ldb = K;         // ⚠️ B 的行宽是 K
        // 可选：验证 B.shape[1] == K
    }

    const float* a = A.ptr<float>();
    const float* b = B.ptr<float>();
    float* c = C.ptr<float>();

    int mp = (M + MR - 1) / MR;
    int np = (N + NR - 1) / NR;

    float* A_pack = workspace;
    float* B_pack = workspace + mp * MR * K;

    pack_A(a, A_pack, M, K, K);
    if (!transB) {
        pack_B(b, B_pack, K, N, ldb);
    } else {
        pack_B_trans(b, B_pack, K, N, ldb);
    }

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

    if (bias) {
        // C: [M, N]
        for (int m = 0; m < M; ++m) {
            int n = 0;
            for (; n <= N - 4; n += 4) {
                float32x4_t v = vld1q_f32(c + m*N + n);
                float32x4_t b = vld1q_f32(bias + n);
                v = vaddq_f32(v, b);
                vst1q_f32(c + m*N + n, v);
            }
            for (; n < N; ++n) {
                c[m*N + n] += bias[n];
            }
        }
    }

    return Status::SUCCESS;
}

Status gemv_neon_transposed(
    const Tensor& A, 
    const Tensor& B_T, 
    Tensor& C,
    const float* bias
) {
    if (A.dtype != DataType::FP32 || B_T.dtype != DataType::FP32 || C.dtype != DataType::FP32)
        return Status::INVALID_ARGUMENT;

    int K = A.shape[1];
    int N = B_T.shape[0]; // B_T 是 [N, K]

    const float* x = A.ptr<float>();
    const float* W = B_T.ptr<float>();
    float* y = C.ptr<float>();

    for (int n = 0; n < N; n++) {
        const float* w = W + n * K; // 定位到 W 的第 n 行首地址

        float32x4_t acc = vdupq_n_f32(0.0f);
        int k = 0;
        
        // 核心：连续内存访问，纯享 NEON 加速
        for (; k <= K - 4; k += 4) {
            float32x4_t xv = vld1q_f32(x + k);
            float32x4_t wv = vld1q_f32(w + k);
            acc = vfmaq_f32(acc, wv, xv);  // acc += wv * xv
        }

        // 规约求和
        float sum = vaddvq_f32(acc);

        // 处理尾部
        for (; k < K; k++) {
            sum += x[k] * w[k];
        }

        if (bias) sum += bias[n];
        
        y[n] = sum;
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

Status bmm_neon(
    const Tensor& A, // [B, M, K]
    const Tensor& B, // [B, K, N] 或 [B, N, K]
    Tensor& C,       // [B, M, N]
    float* workspace,
    bool transB
) {
    int batch = A.shape[0];
    int M = A.shape[1];
    int K = A.shape[2];
    int N = transB ? B.shape[1] : B.shape[2];
    int stride_A = M * K;
    int stride_B = transB ? (N * K) : (K * N);
    int stride_C = M * N;
    
    for (int b = 0; b < batch; b++) {
        Tensor A_b;
        A_b.shape = {M, K};
        A_b.data = (void*)(A.ptr<float>() + b * stride_A);
        A_b.dtype = A.dtype; 
        A_b.device = A.device;

        Tensor B_b;
        B_b.shape = transB ? std::vector<int>{N, K} : std::vector<int>{K, N};
        B_b.data = (void*)(B.ptr<float>() + b * stride_B);
        B_b.dtype = B.dtype;
        B_b.device = B.device;

        Tensor C_b;
        C_b.shape = {M, N};
        C_b.data = (void*)(C.ptr<float>() + b * stride_C);
        C_b.dtype = C.dtype;
        C_b.device = C.device;

        // ⚠️ 将 transB 传给底层的 matmul_neon
        Status status = matmul_neon(A_b, B_b, C_b, workspace, transB);
        if (status != Status::SUCCESS) {
            return status;
        }
    }

    return Status::SUCCESS;
}

Status matmul_f16_neon(
    const Tensor& A,
    const Tensor& B,
    Tensor& C,
    fp16_t* workspace,
    bool transB,
    const fp16_t* bias
) {
    if (A.dtype != DataType::FP16 || B.dtype != DataType::FP16 || C.dtype != DataType::FP16)
        return Status::INVALID_ARGUMENT;
    if (!workspace)
        return Status::INVALID_ARGUMENT;
    if (A.device != DeviceType::CPU || B.device != DeviceType::CPU || C.device != DeviceType::CPU)
        return Status::UNSUPPORTED_DEVICE;

    int M = A.shape[0];
    int K = A.shape[1];
    int N = transB ? B.shape[0] : B.shape[1];

    const fp16_t* a = A.ptr<fp16_t>();
    const fp16_t* b = B.ptr<fp16_t>();
    fp16_t* c = C.ptr<fp16_t>();

    // Correctness-first FP32 accumulation. Inputs/outputs stay FP16, but K
    // reductions are held in float32x4_t registers to avoid the NaNs seen with
    // the old pure-FP16 accumulator.
    (void)workspace;

    auto reduce_f32x8 = [](float32x4_t lo, float32x4_t hi) -> float {
        return vaddvq_f32(vaddq_f32(lo, hi));
    };

    auto dot_contiguous_f16_fp32 = [&](const fp16_t* lhs, const fp16_t* rhs) -> float {
        float32x4_t acc0 = vdupq_n_f32(0.0f);
        float32x4_t acc1 = vdupq_n_f32(0.0f);

        int k = 0;
        for (; k <= K - 8; k += 8) {
            // Load 8 FP16 values from each contiguous vector, widen to two
            // FP32 vectors, then accumulate in FP32.
            float16x8_t av = vld1q_f16(lhs + k);
            float16x8_t bv = vld1q_f16(rhs + k);
            acc0 = vfmaq_f32(acc0,
                             vcvt_f32_f16(vget_low_f16(av)),
                             vcvt_f32_f16(vget_low_f16(bv)));
            acc1 = vfmaq_f32(acc1,
                             vcvt_f32_f16(vget_high_f16(av)),
                             vcvt_f32_f16(vget_high_f16(bv)));
        }

        float sum = reduce_f32x8(acc0, acc1);
        for (; k < K; ++k) {
            sum += (float)lhs[k] * (float)rhs[k];
        }
        return sum;
    };

    if (transB) {
        // B is logically transposed: each output column reads one contiguous
        // row of B, so vectorize over K and compute four columns per pass.
        for (int m = 0; m < M; ++m) {
            const fp16_t* a_row = a + (size_t)m * K;
            int n = 0;
            for (; n <= N - 4; n += 4) {
                float32x4_t acc00 = vdupq_n_f32(0.0f);
                float32x4_t acc01 = vdupq_n_f32(0.0f);
                float32x4_t acc10 = vdupq_n_f32(0.0f);
                float32x4_t acc11 = vdupq_n_f32(0.0f);
                float32x4_t acc20 = vdupq_n_f32(0.0f);
                float32x4_t acc21 = vdupq_n_f32(0.0f);
                float32x4_t acc30 = vdupq_n_f32(0.0f);
                float32x4_t acc31 = vdupq_n_f32(0.0f);

                const fp16_t* b0 = b + (size_t)(n + 0) * K;
                const fp16_t* b1 = b + (size_t)(n + 1) * K;
                const fp16_t* b2 = b + (size_t)(n + 2) * K;
                const fp16_t* b3 = b + (size_t)(n + 3) * K;

                int k = 0;
                for (; k <= K - 8; k += 8) {
                    float16x8_t av = vld1q_f16(a_row + k);
                    float32x4_t alo = vcvt_f32_f16(vget_low_f16(av));
                    float32x4_t ahi = vcvt_f32_f16(vget_high_f16(av));

                    float16x8_t bv0 = vld1q_f16(b0 + k);
                    acc00 = vfmaq_f32(acc00, alo, vcvt_f32_f16(vget_low_f16(bv0)));
                    acc01 = vfmaq_f32(acc01, ahi, vcvt_f32_f16(vget_high_f16(bv0)));

                    float16x8_t bv1 = vld1q_f16(b1 + k);
                    acc10 = vfmaq_f32(acc10, alo, vcvt_f32_f16(vget_low_f16(bv1)));
                    acc11 = vfmaq_f32(acc11, ahi, vcvt_f32_f16(vget_high_f16(bv1)));

                    float16x8_t bv2 = vld1q_f16(b2 + k);
                    acc20 = vfmaq_f32(acc20, alo, vcvt_f32_f16(vget_low_f16(bv2)));
                    acc21 = vfmaq_f32(acc21, ahi, vcvt_f32_f16(vget_high_f16(bv2)));

                    float16x8_t bv3 = vld1q_f16(b3 + k);
                    acc30 = vfmaq_f32(acc30, alo, vcvt_f32_f16(vget_low_f16(bv3)));
                    acc31 = vfmaq_f32(acc31, ahi, vcvt_f32_f16(vget_high_f16(bv3)));
                }

                float sum0 = reduce_f32x8(acc00, acc01);
                float sum1 = reduce_f32x8(acc10, acc11);
                float sum2 = reduce_f32x8(acc20, acc21);
                float sum3 = reduce_f32x8(acc30, acc31);
                for (; k < K; ++k) {
                    float av = (float)a_row[k];
                    sum0 += av * (float)b0[k];
                    sum1 += av * (float)b1[k];
                    sum2 += av * (float)b2[k];
                    sum3 += av * (float)b3[k];
                }

                if (bias) {
                    sum0 += (float)bias[n + 0];
                    sum1 += (float)bias[n + 1];
                    sum2 += (float)bias[n + 2];
                    sum3 += (float)bias[n + 3];
                }
                c[(size_t)m * N + n + 0] = (fp16_t)sum0;
                c[(size_t)m * N + n + 1] = (fp16_t)sum1;
                c[(size_t)m * N + n + 2] = (fp16_t)sum2;
                c[(size_t)m * N + n + 3] = (fp16_t)sum3;
            }

            for (; n < N; ++n) {
                float sum = dot_contiguous_f16_fp32(a_row, b + (size_t)n * K);
                if (bias) sum += (float)bias[n];
                c[(size_t)m * N + n] = (fp16_t)sum;
            }
        }
    } else {
        // B is KxN row-major. For each output row, compute eight contiguous
        // columns at once so every B load is a 128-bit contiguous FP16 access.
        for (int m = 0; m < M; ++m) {
            const fp16_t* a_row = a + (size_t)m * K;
            int n = 0;
            for (; n <= N - 8; n += 8) {
                float32x4_t acc0 = vdupq_n_f32(0.0f);
                float32x4_t acc1 = vdupq_n_f32(0.0f);

                for (int k = 0; k < K; ++k) {
                    float32x4_t av = vdupq_n_f32((float)a_row[k]);
                    float16x8_t bv = vld1q_f16(b + (size_t)k * N + n);
                    acc0 = vfmaq_f32(acc0, av, vcvt_f32_f16(vget_low_f16(bv)));
                    acc1 = vfmaq_f32(acc1, av, vcvt_f32_f16(vget_high_f16(bv)));
                }

                if (bias) {
                    float16x8_t bv = vld1q_f16(bias + n);
                    acc0 = vaddq_f32(acc0, vcvt_f32_f16(vget_low_f16(bv)));
                    acc1 = vaddq_f32(acc1, vcvt_f32_f16(vget_high_f16(bv)));
                }

                float16x8_t out = vcombine_f16(vcvt_f16_f32(acc0), vcvt_f16_f32(acc1));
                vst1q_f16(c + (size_t)m * N + n, out);
            }

            for (; n < N; ++n) {
                float sum = bias ? (float)bias[n] : 0.0f;
                for (int k = 0; k < K; ++k) {
                    sum += (float)a_row[k] * (float)b[(size_t)k * N + n];
                }
                c[(size_t)m * N + n] = (fp16_t)sum;
            }
        }
    }

    return Status::SUCCESS;
}

void add_f16_neon(const Tensor& A, const Tensor& B, Tensor& C) {
    const fp16_t* a = A.ptr<fp16_t>();
    const fp16_t* b = B.ptr<fp16_t>();
    fp16_t* c = C.ptr<fp16_t>();
    int total = A.size();

    int i = 0;
    for (; i <= total - 8; i += 8) {
        float16x8_t va = vld1q_f16(a + i);
        float16x8_t vb = vld1q_f16(b + i);
        vst1q_f16(c + i, vaddq_f16(va, vb));
    }
    for (; i < total; ++i) {
        c[i] = (fp16_t)((float)a[i] + (float)b[i]);
    }
}

}
}
