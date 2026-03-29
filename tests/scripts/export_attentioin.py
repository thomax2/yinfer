import torch
import numpy as np
import os

def export_tensor(tensor, filename):
    # 将 PyTorch Tensor 展平并保存为 float32 的二进制文件
    tensor.detach().cpu().numpy().astype(np.float32).tofile(filename)

def main():
    # 1. 设置 Qwen2.5 风格的配置 (缩小版)
    bsz = 1
    hidden_dim = 64
    num_q_heads = 4
    num_kv_heads = 2
    head_dim = 16
    
    os.makedirs("tests/data", exist_ok=True)

    # 2. 生成随机输入和权重
    torch.manual_seed(42)
    hidden_states = torch.randn(bsz, 1, hidden_dim) # 当前步的输入
    
    w_q = torch.randn(hidden_dim, num_q_heads * head_dim)
    w_k = torch.randn(hidden_dim, num_kv_heads * head_dim)
    w_v = torch.randn(hidden_dim, num_kv_heads * head_dim)
    w_o = torch.randn(num_q_heads * head_dim, hidden_dim)

    # 为了聚焦测试 Matmul 和 GQA 逻辑，暂时将 RoPE 的 cos 设为 1，sin 设为 0 (相当于不旋转)
    cos = torch.ones(num_q_heads * head_dim)
    sin = torch.zeros(num_q_heads * head_dim)

    # 导出输入
    export_tensor(hidden_states, "tests/data/hidden_states.bin")
    export_tensor(w_q, "tests/data/w_q.bin")
    export_tensor(w_k, "tests/data/w_k.bin")
    export_tensor(w_v, "tests/data/w_v.bin")
    export_tensor(w_o, "tests/data/w_o.bin")
    export_tensor(cos, "tests/data/cos.bin")
    export_tensor(sin, "tests/data/sin.bin")

    # 3. 模拟 PyTorch 端的前向传播 (当前为第 0 步)
    # Q, K, V 投影
    q = hidden_states @ w_q
    k = hidden_states @ w_k
    v = hidden_states @ w_v

    # 形状变换 [1, 1, heads, dim] -> [1, heads, 1, dim]
    q = q.view(bsz, 1, num_q_heads, head_dim).transpose(1, 2)
    k = k.view(bsz, 1, num_kv_heads, head_dim).transpose(1, 2)
    v = v.view(bsz, 1, num_kv_heads, head_dim).transpose(1, 2)

    # 因为是第 0 步，Cache 里只有当前 token
    k_cache = k
    v_cache = v

    # GQA 广播: 把 2 个 KV 头复制成 4 个，与 Q 头对齐
    num_rep = num_q_heads // num_kv_heads
    k_rep = torch.repeat_interleave(k_cache, num_rep, dim=1)
    v_rep = torch.repeat_interleave(v_cache, num_rep, dim=1)

    # Score = Q * K^T / sqrt(d)
    scores = torch.matmul(q, k_rep.transpose(-2, -1)) / (head_dim ** 0.5)
    
    # Softmax
    probs = torch.nn.functional.softmax(scores, dim=-1)
    
    # Output = Score * V
    out = torch.matmul(probs, v_rep)

    # 最后的 O 投影
    out = out.transpose(1, 2).contiguous().view(bsz, 1, -1)
    final_out = out @ w_o

    # 4. 导出黄金结果
    export_tensor(final_out, "tests/data/golden_attn_out.bin")
    export_tensor(k_cache, "tests/data/golden_k_cache.bin")
    export_tensor(v_cache, "tests/data/golden_v_cache.bin")
    print("Golden data generated successfully in tests/data/ !")

if __name__ == "__main__":
    main()