#pragma once

#include "llm_engine/tensor.h"
#include "kernel_common.h"

namespace llm_engine {
namespace arm_neon {

/*
A: 原矩阵首地址
A_pack: 打包后矩阵首地址
M: A 的行数
K: A 的列数
lda: A 的行跨度（即每行元素之间的距离，单位为元素个数），通常等于矩阵的列，
    是大矩阵中的小块时，lda是大矩阵的列数
*/
void pack_A(
    const float* A,
    float* A_pack,
    int M,
    int K,
    int lda
);

void pack_B(
    const float* B,
    float* B_pack,
    int K,
    int N,
    int ldb
);

void pack_B_trans(
    const float* B,
    float* B_pack,
    int K,
    int N,
    int ldb
);

void pack_weight_for_linear_decode(
    const float* W,
    float* W_pack,
    int K,
    int N
);

void pack_A_f16(
    const fp16_t* A,
    fp16_t* A_pack,
    int M,
    int K,
    int lda
);

void pack_B_f16(
    const fp16_t* B,
    fp16_t* B_pack,
    int K,
    int N,
    int ldb
);

void pack_B_trans_f16(
    const fp16_t* B,
    fp16_t* B_pack,
    int K,
    int N,
    int ldb
);

void pack_B_gptq_w8a16(
    const int8_t* qweight_kn,
    const fp16_t* scales_gn,
    const int8_t* zeros_gn,
    int8_t* qweight_pack,
    fp16_t* scales_pack,
    int8_t* zeros_pack,
    int K,
    int N,
    int group_size
);

}
}
