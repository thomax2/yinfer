import torch
import torch.nn.functional as F
import os

def export_tensor(tensor, filename):
    # 转为 float32 并按 C++ 中紧凑的 C-Contiguous 内存格式保存
    tensor.detach().cpu().float().contiguous().numpy().tofile(filename)

def main():
    # 确保 data 目录存在
    os.makedirs("../data", exist_ok=True)
    
    # 1. 参数配置 (与 C++ 测试对齐)
    num_tokens = 1
    hidden_dim = 64
    intermediate_size = 128
    
    # 2. 随机生成输入
    torch.manual_seed(42)
    hidden_states = torch.randn(num_tokens, hidden_dim)
    
    # 3. 随机生成权重
    # 注意：PyTorch nn.Linear 的权重形状是 [out_features, in_features]
    # 但由于 C++ 张量我们定义的是 [in_features, out_features] 用于标准矩阵乘法，
    # 我们在导出时主动对其转置 .t()，使得 C++ 读取时内存排布一致。
    w_gate = torch.randn(intermediate_size, hidden_dim)
    w_up = torch.randn(intermediate_size, hidden_dim)
    w_down = torch.randn(hidden_dim, intermediate_size)
    
    # 4. 前向传播 (PyTorch 基准计算)
    # FFN(x) = (SiLU(x * W_gate) * (x * W_up)) * W_down
    # 注意用 F.linear 的等价矩阵乘法方式计算
    gate = hidden_states @ w_gate.t()
    up = hidden_states @ w_up.t()
    
    # SwiGLU 激活：SiLU(gate) * up
    act = F.silu(gate) * up
    
    # 向下投影
    ffn_out = act @ w_down.t()
    
    # 5. 导出 Bin 文件
    export_tensor(hidden_states, "../data/ffn_hidden_states.bin")
    
    # 导出权重，转置为 [in_features, out_features] 格式供 C++ 直接使用
    export_tensor(w_gate.t(), "../data/ffn_w_gate.bin")
    export_tensor(w_up.t(), "../data/ffn_w_up.bin")
    export_tensor(w_down.t(), "../data/ffn_w_down.bin")
    
    # 导出最终黄金输出
    export_tensor(ffn_out, "../data/golden_ffn_out.bin")
    
    print("✅ FFN Golden data exported successfully to ../data/")

if __name__ == "__main__":
    main()