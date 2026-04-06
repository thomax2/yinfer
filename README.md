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


## Attention 算子单元测试报告

**测试可执行文件：** `./build/tests/test_attention`

### 测试概要

| 测试套件 | 测试数量 | 通过 | 失败 |
| :--- | :--- | :--- | :--- |
| AttentionTest | 1 | 1 | 0 |

### 测试详情

| 测试用例 | 描述 | 状态 |
| :--- | :--- | :--- |
| `PyTorchAlignment` | 与 PyTorch 黄金输出对齐测试（含 KV Cache 副作用验证） | ✅ PASS |

### 测试说明

- **测试配置：** `hidden_dim=64, num_q_heads=4, num_kv_heads=2, head_dim=16`（GQA 分组查询注意力）
- **验证内容：**
  1. `attention_neon` 算子的最终输出与 PyTorch 参考实现输出对齐（容忍误差 `1e-4`）
  2. KV Cache 中写入的 K 张量与 PyTorch 导出的黄金数据对齐（容忍误差 `1e-5`）
- **测试数据：** 从 Python 脚本生成的二进制文件加载（`data/*.bin`）

### 测试环境

- **数据路径解析：** 自动搜索 `TEST_DATA_DIR` 环境变量指定的目录、源码 `tests/` 目录，以及构建目录的相对路径
- **内存资源：**
  - MemoryPool：256 MB
  - Workspace：2 MB
  - KVCache：`batch=1, max_seq_len=128`

### 结论

- `arm_neon::attention_neon` 实现与 PyTorch 参考实现完全对齐
- KV Cache 的写入逻辑正确，支持增量式生成场景（首个 token 生成）
- 所有单元测试通过，Attention 模块可投入实际推理使用

## Qwen Block 单元测试报告

**测试可执行文件：** `./build/tests/test_qwen_block`

### 测试概要

| 测试套件 | 测试数量 | 通过 | 失败 |
| :--- | :--- | :--- | :--- |
| QwenBlockTest | 1 | 1 | 0 |

### 测试详情

| 测试用例 | 描述 | 状态 |
| :--- | :--- | :--- |
| `PyTorchAlignment` | Qwen Block 完整前向计算与 PyTorch 黄金输出对齐测试 | ✅ PASS |

### 测试说明

- **测试配置：**
  - `hidden_dim=64, num_q_heads=4, num_kv_heads=2, head_dim=16`
  - `intermediate_size=128, rms_norm_eps=1e-6`
  - `max_seq_len=128, num_tokens=1`
- **验证内容：**
  - Qwen Block 完整前向计算结果与 PyTorch 导出的黄金输出对齐（容忍误差 `2e-3`）
  - 包含 RMS Norm、GQA 注意力、FFN（SwiGLU）等全部子模块
- **测试数据：** 从 Python 脚本生成的二进制文件加载（`data/*.bin`）

### 测试环境

- **数据路径解析：** 自动搜索 `TEST_DATA_DIR` 环境变量、源码 `tests/` 目录及构建目录相对路径
- **内存资源：**
  - MemoryPool：256 MB
  - Workspace：4 MB
  - KVCache：`batch=1, max_seq_len=128`

### 结论

- `arm_neon::qwen_block_neon` 完整实现与 PyTorch 参考实现完全对齐
- Qwen Block 所有子模块（RMS Norm、GQA 注意力、RoPE、SwiGLU FFN）集成正确
- 单元测试通过，Qwen Block 模块可投入实际推理使用

---

## Qwen Block 图集成测试报告

**测试可执行文件：** `./build/tests/test_graph_qwen_block`

### 测试概要

| 测试套件 | 测试数量 | 通过 | 失败 |
| :--- | :--- | :--- | :--- |
| GraphIntegrationTest | 1 | 1 | 0 |

### 测试详情

| 测试用例 | 描述 | 状态 |
| :--- | :--- | :--- |
| `QwenBlockNode` | 将 Qwen Block 封装为图节点，经编译执行后对齐 PyTorch 输出 | ✅ PASS |

### 测试说明

- **测试配置：** 同上（`hidden_dim=64, num_q_heads=4, num_kv_heads=2, head_dim=16`）
- **验证内容：**
  1. 将 15 个外部数据指针（权重、偏置、cos/sin 等）注册为图中的 Tensor
  2. 通过 `add_qwen_block` 接口创建图节点
  3. 图编译器生成执行计划并执行
  4. 最终输出与 PyTorch 黄金输出对齐（容忍误差 `2e-3`）
- **内存规划器输出：** 峰值工作区需求为 **0 MB**（所有张量均为外部绑定，无额外内存分配）

### 测试环境

- **数据路径解析：** 同前
- **内存资源：**
  - MemoryPool：256 MB
  - Workspace：由图执行器动态管理

### 结论

- Qwen Block 节点可正确集成到计算图中
- 图编译器能正确处理外部绑定的张量，无需额外工作区内存
- 编译-执行流程完整，Qwen Block 图集成测试通过