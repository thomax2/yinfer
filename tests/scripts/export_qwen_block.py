import torch
import torch.nn.functional as F
import os

def export_tensor(tensor, filename):
    tensor.detach().cpu().float().contiguous().numpy().tofile(filename)

def rmsnorm(x, weight, eps=1e-6):
    variance = x.pow(2).mean(-1, keepdim=True)
    return (x * torch.rsqrt(variance + eps)) * weight

def main():
    os.makedirs("../data", exist_ok=True)
    
    hidden_dim, num_q_heads, num_kv_heads, head_dim = 64, 4, 2, 16
    intermediate_size = 128
    
    torch.manual_seed(42)
    hidden_states = torch.randn(1, hidden_dim) * 0.1
    
    norm1_w = torch.ones(hidden_dim)
    w_q = torch.randn(num_q_heads * head_dim, hidden_dim)
    w_k = torch.randn(num_kv_heads * head_dim, hidden_dim)
    w_v = torch.randn(num_kv_heads * head_dim, hidden_dim)
    w_o = torch.randn(hidden_dim, num_q_heads * head_dim)
    
    # ✅ 1. 新增：随机生成 Q, K, V 的 Bias
    b_q = torch.randn(num_q_heads * head_dim) * 0.1
    b_k = torch.randn(num_kv_heads * head_dim) * 0.1
    b_v = torch.randn(num_kv_heads * head_dim) * 0.1
    
    cos = torch.ones(num_q_heads * head_dim) 
    sin = torch.zeros(num_q_heads * head_dim)
    
    norm2_w = torch.ones(hidden_dim)
    w_gate = torch.randn(intermediate_size, hidden_dim)
    w_up = torch.randn(intermediate_size, hidden_dim)
    w_down = torch.randn(hidden_dim, intermediate_size)

    # 2. 导出所有输入和权重
    export_tensor(hidden_states, "../data/block_hidden_states.bin")
    export_tensor(norm1_w, "../data/block_norm1_w.bin")
    export_tensor(w_q.t(), "../data/block_w_q.bin")
    export_tensor(w_k.t(), "../data/block_w_k.bin")
    export_tensor(w_v.t(), "../data/block_w_v.bin")
    export_tensor(w_o.t(), "../data/block_w_o.bin")
    
    # ✅ 2. 新增：导出 Bias (一维向量直接导出，无需转置)
    export_tensor(b_q, "../data/block_b_q.bin")
    export_tensor(b_k, "../data/block_b_k.bin")
    export_tensor(b_v, "../data/block_b_v.bin")
    
    export_tensor(cos, "../data/block_cos.bin")
    export_tensor(sin, "../data/block_sin.bin")
    export_tensor(norm2_w, "../data/block_norm2_w.bin")
    export_tensor(w_gate.t(), "../data/block_w_gate.bin")
    export_tensor(w_up.t(), "../data/block_w_up.bin")
    export_tensor(w_down.t(), "../data/block_w_down.bin")

    # 3. Qwen Block 前向模拟
    # [Pre-Norm]
    residual = hidden_states
    x_norm1 = rmsnorm(hidden_states, norm1_w)
    
    # [Attention 简化计算] 
    # ⚠️ 记得之后替换为你包含 RoPE 的完整版 Attention
    # ✅ 3. 新增：在矩阵乘法后加上对应的 Bias
    q = x_norm1 @ w_q.t() + b_q
    k = x_norm1 @ w_k.t() + b_k
    v = x_norm1 @ w_v.t() + b_v
    
    attn_scores = (q @ k.t()) / (head_dim ** 0.5)
    attn_probs = F.softmax(attn_scores, dim=-1)
    attn_out = attn_probs @ v
    attn_out = attn_out @ w_o.t()
    
    # [Add]
    hidden_states = residual + attn_out
    
    # [Pre-Norm]
    residual = hidden_states
    x_norm2 = rmsnorm(hidden_states, norm2_w)
    
    # [SwiGLU FFN] (FFN 没有 Bias)
    gate = x_norm2 @ w_gate.t()
    up = x_norm2 @ w_up.t()
    ffn_out = (F.silu(gate) * up) @ w_down.t()
    
    # [Add]
    hidden_states = residual + ffn_out

    # 4. 导出期望结果
    export_tensor(hidden_states, "../data/golden_block_out.bin")
    print("Qwen Block golden data exported!")

if __name__ == "__main__":
    main()