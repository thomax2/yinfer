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


# Decode 阶段性能优化报告
在之前的测试中，Qwen 模型虽然能够正确运行并产出结果，但在逐 token 生成（Decode 阶段）时的推理速度较慢（约 0.45 tok/s）。针对这一瓶颈，我们针对单 token 推理路径进行了深度的算子与内存布局优化。
## 优化内容
本次优化核心思想是**“计算与数据布局解耦”**，针对 Decode 阶段 `num_tokens == 1` 的特征，绕开通用矩阵乘法的额外开销，直接消费预打包的权重数据：
- **新增 Decode 专属 Kernel：** 实现 `linear_decode_prepacked_neon()`，在单 token 推理时直接处理 `[ceil(N/12), K, 12]` 布局的 packed 权重，极大提升了访存效率。
- **权重预打包机制：** 修改 `pack.h / pack.cpp`，新增 `pack_weight_for_linear_decode()`；在 `model.h / model.cpp` 中新增 block packed 权重与 `lm_head_pack`，在构造时分配内存，并在 `load_weights()` 后统一调用 `prepack_all_weights()` 完成离线打包。
- **计算图与调用链适配：** 修改 graph/block 调用链，使 packed 权重能够从 `build_graph()` 依次传递至 `QwenBlockNode` 及底层 `qwen_block_neon()`。
- **动态路由分流：** 修改 `attention_neon.cpp` 和 `ffn_neon.cpp`，当检测到 `num_tokens == 1` 时走新增的 decode linear 路径，其他情况（Prefill 阶段）保留原 `matmul_neon()` 路径。
## 性能对比分析
优化前后均运行相同的 Qwen 模型生成任务，结果如下：
| 运行阶段 | 生成 Token 数 | 总耗时 (秒) | 推理速度 | 加速比 |
| :--- | :--- | :--- | :--- | :--- |
| **优化前 (run1)** | 19 | 42.27 | 0.449 tok/s | - |
| **优化后 (run2)** | 18 | 3.36 | 5.354 tok/s | **~11.9x** |
> **注：** 优化后 Decode 速度提升约 **11.9 倍**，彻底解决了单 token 生成缓慢的问题。
## 硬件资源使用分析
基于 `run1.csv` 与 `run2.csv` 的系统监控数据，优化前后硬件资源使用特征发生了显著变化：
### 1. 内存占用 (RAM)
| 指标 | 优化前 (run1) | 优化后 (run2) | 变化说明 |
| :--- | :--- | :--- | :--- |
| **进程 RSS 峰值** | ~3161 MB | ~4503 MB | 增加 ~1342 MB |
| **系统可用内存** | ~3166 MB | ~1245 MB | 相应减少 |
**分析：** 内存占用的增加符合预期。由于引入了针对 decode 阶段的预打包权重（`[ceil(N/12), K, 12]` 布局），模型在加载时会同时保留原始权重与预打包权重，这属于典型的**“以空间换时间”**优化策略。
### 2. CPU 与运行时间
| 指标 | 优化前 (run1) | 优化后 (run2) | 变化说明 |
| :--- | :--- | :--- | :--- |
| **高负载持续时间** | 约 50 秒 | 约 15 秒 | 计算耗时大幅缩短 |
| **进程状态** | 长时间处于 `R` (Running) | 短暂 `R` 后迅速回到 `S` (Sleep) | CPU 释放更迅速 |
**分析：** 优化前进程长时间占满单核 CPU（`proc_cpu_pct` 接近 100%），优化后 CPU 密集计算阶段高度集中且迅速完成，说明 NEON 专属 Kernel 的计算效率远高于通用路径。
## 结论
- 针对 Decode 阶段的 NEON 算子定制与权重预打包策略取得了巨大成功，推理速度从 0.45 tok/s 提升至 5.35 tok/s。
- 虽然预打包权重带来了约 1.3 GB 的额外内存开销，但换取了接近 12 倍的性能提升，在内存资源允许的设备上具有极高的应用价值。
- `num_tokens == 1` 的条件分支路由成功，未影响 Prefill 阶段的逻辑，系统运行稳定。
