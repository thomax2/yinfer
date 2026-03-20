# ARM NEON Matmul 性能测试报告

**测试环境：**
- **可执行文件：** ./benchmarks/bench_gemm

## 系统配置
- **CPU:** 8 核 @ 1500 MHz
- **缓存配置:**
  - L1 Data: 32 KiB (x8)
  - L1 Instruction: 32 KiB (x8)
  - L2 Unified: 64 KiB (x8)
  - L3 Unified: 1024 KiB (x1)
- **负载均值:** 1.89, 1.54, 1.27

> **警告:** CPU scaling 已启用，基准测试的真实时间测量可能存在噪音并产生额外开销。

## 基准测试结果

| Benchmark | Time (ns) | CPU (ns) | Iterations |
| :--- | :--- | :--- | :--- |
| **BM_matmul_ref/128** | 12,578,522 | 12,562,656 | 55 |
| **BM_matmul_ref/512** | 4,862,411,929 | 4,851,424,139 | 1 |
| **BM_matmul_ref/1024** | 141,320,483,074 | 141,159,281,658 | 1 |
| **BM_matmul_neon/128** | 739,242 | 738,306 | 948 |
| **BM_matmul_neon/512** | 47,041,362 | 46,953,122 | 15 |
| **BM_matmul_neon/1024** | 346,384,826 | 345,869,604 | 2 |

## 性能对比分析 (Ref vs NEON)

通过对比普通实现 (`ref`) 与 NEON 优化实现 (`neon`) 的耗时，可以看出显著的性能提升：

| 矩阵尺寸 | Ref 耗时 (ms) | NEON 耗时 (ms) | 加速比 |
| :--- | :--- | :--- | :--- |
| 128 | 12.58 | 0.74 | **~17.0x** |
| 512 | 4862.41 | 47.04 | **~103.4x** |
| 1024 | 141320.48 | 346.38 | **~408.2x** |

## 结论
完成 `arm_neon` matmul 的测试。测试结果显示，使用 NEON 指令集优化后，矩阵乘法性能得到了极大的提升，且随着矩阵尺寸增大，优化效果愈发显著。

---

# 图执行器单元测试报告

**测试可执行文件：** `./build/tests/test_graph`

## 测试概要

| 测试套件 | 测试数量 | 通过 | 失败 |
| :--- | :--- | :--- | :--- |
| InplaceMemoryTest | 4 | 4 | 0 |
| CompilerTest | 1 | 1 | 0 |
| **总计** | **5** | **5** | **0** |

## 测试详情

### InplaceMemoryTest

| 测试用例 | 描述 | 状态 |
| :--- | :--- | :--- |
| `InplaceWithSubsequentUse_External` | 原地操作 + 后续使用（外部绑定） | ✅ PASS |
| `InplaceAsFinalOutput_External` | 原地操作作为最终输出（外部绑定） | ✅ PASS |
| `InternalArenaRecycling` | 内部内存池回收验证 | ✅ PASS |
| `StressMultipleInplace_External` | 多重原地操作压力测试（外部绑定） | ✅ PASS |

**内存规划器输出摘要：**
- 所有测试中峰值工作区需求均为 0 MB（`InternalArenaRecycling` 峰值约为 3.05e-05 MB）
- 内存复用机制正常工作

### CompilerTest

| 测试用例 | 描述 | 状态 |
| :--- | :--- | :--- |
| `ShouldThrowIfBoundaryNotBound` | 编译器检测未绑定的边界张量并抛出异常 | ✅ PASS |

## 结论

- 内存原地操作（inplace）功能正常，包括外部绑定场景和内部内存池回收机制
- 编译器能够正确识别未绑定的边界张量并抛出异常，保证计算图的完整性
- 所有单元测试均通过，系统运行稳定