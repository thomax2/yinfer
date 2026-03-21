#include "pack.h"
#include <cstring>

namespace llm_engine {
namespace arm_neon {


    
/*
A: 原矩阵首地址
A_pack: 打包后矩阵首地址
M: A 的行数
K: A 的列数
lda: A 的行跨度（即每行元素之间的距离，单位为元素个数），通常等于矩阵的列，
    是大矩阵中的小块时，lda是大矩阵的列数

针对左矩阵，按 MR 行一组（Panel）进行分块，块内按列优先存放数据。
行不足部分补0，将矩阵 A 的行数 M 强行向上对齐到 MR 的整数倍。
*/

void pack_A(
    const float* A,
    float* A_pack,
    int M,
    int K,
    int lda
) {
    int mp = (M + MR - 1) / MR;
    for(int i = 0; i < mp; i++) {
        for(int k = 0; k < K; k++) {
            for(int j = 0; j < MR; j++) {
                int row = i * MR + j;
                *A_pack++ = (row < M) ? A[row * lda + k] : 0.0f;
            }
        }
    }
}


/*
    针对右矩阵，按 NR 列一组（Panel）进行分块，块内按行优先。
    
    这里有个视角的变化，可以方便后面理解：
    原B的维度是 KxN，打包后是 (NR*np)xK（按行优先来看），每NR行其实就是原B的NR列，并且在panel内是按行优先存储的。

    原始 B (K=4, N=10, NR=4):

    行\列  0   1   2   3 | 4   5   6   7 | 8   9
    ┌─────────────────┼─────────────────┼──────
 0  │ b00 b01 b02 b03 | b04 b05 b06 b07 | b08 b09
 1  │ b10 b11 b12 b13 | b14 b15 b16 b17 | b18 b19
 2  │ b20 b21 b22 b23 | b24 b25 b26 b27 | b28 b29
 3  │ b30 b31 b32 b33 | b34 b35 b36 b37 | b38 b39

    块0 (列 0~3):       
    ┌────────────┐       
   │ b00 b01 b02 b03 │  
   │ b10 b11 b12 b13 │  
   │ b20 b21 b22 b23 │  
   │ b30 b31 b32 b33 │  
    └────────────┘       

    块1 (列 4~7):      
    ┌────────────┐       
  │ b04 b05 b06 b07 │ 
  │ b14 b15 b16 b17 │ 
  │ b24 b25 b26 b27 │ 
  │ b34 b35 b36 b37 │ 
    └────────────┘       

     块2 (列 8~9+填充):
    ┌────────────┐
   │ b08 b09 0  0 │ 
   │ b18 b19 0  0 │
   │ b28 b29 0  0 │
   │ b38 b39 0  0 │
    └────────────┘
*/
void pack_B(
    const float* B,
    float* B_pack,
    int K,
    int N,
    int ldb
) {
    int np = (N + NR - 1) / NR;
    for(int j = 0; j < np; j++) {
        for(int k = 0; k < K; k++) {
            for(int i = 0; i < NR; i++) {
                int col = j*NR + i;
                *B_pack++ = (col < N) ? B[k*ldb + col]: 0.0f;
            }
        }
    }
}

/*
    传入B的行列K*N, B_trans 相当于是切B的行，而不是列，行优先改为列优先
    K*N->N*K
    在 matmul_neon 中，保证了传入 K 是A的列，N在trans情况下是B的行
    转置下的pack_B，就是按 NR 行一组（Panel）进行分块，块内按列优先存放数据。
*/
void pack_B_trans(
    const float* B,      // N×K, 行优先
    float* B_pack,
    int K,               // 内积维度 (与 pack_B 保持一致!)
    int N,               // 输入行数，输出列数
    int ldb              // B 的行宽 (=K)
) {
    int np = (N + NR - 1) / NR;
    
    for(int j = 0; j < np; j++) {           // Panel 索引
        for(int k = 0; k < K; k++) {        // 内积维度
            for(int i = 0; i < NR; i++) {   // Panel 内列
                int row = j*NR + i;         // 逻辑行索引 (0~N-1)
                // 关键映射: Bᵀ(k, row) = B(row, k)
                *B_pack++ = (row < N) ? B[row * ldb + k] : 0.0f;
            }
        }
    }
}

}


}