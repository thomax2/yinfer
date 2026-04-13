import os
import torch
from modelscope import AutoModelForCausalLM, AutoTokenizer

def export_tensor(tensor, filename, transpose=False):
    t = tensor.detach().cpu().float()
    if transpose and len(t.shape) == 2:
        t = t.t() 
    t.contiguous().numpy().tofile(filename)
    # 不打印太多，免得刷屏
    # print(f"Exported {filename}, shape: {t.shape}")

def main():
    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    print(f"正在加载 {model_id} ...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        torch_dtype="auto",
        device_map="cpu"
    )

    state_dict = model.state_dict()
    config = model.config
    num_layers = config.num_hidden_layers
    
    save_dir = "qwen_0.5b_full_bins"
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"模型加载成功！共有 {num_layers} 层 Transformer Block。")
    print("正在导出...")

    # ==========================================
    # 1. 导出 Embedding 层 (查表操作，不需要转置)
    # ==========================================
    export_tensor(state_dict["model.embed_tokens.weight"], 
                  f"{save_dir}/embed_tokens_w.bin", transpose=False)
    print("✅ Embedding 层导出完成")

    # ==========================================
    # 2. 循环导出所有的 Transformer Blocks
    # ==========================================
    for i in range(num_layers):
        prefix = f"model.layers.{i}"
        
        # [Norm 1]
        export_tensor(state_dict[f"{prefix}.input_layernorm.weight"], 
                      f"{save_dir}/layer{i}_norm1_w.bin", transpose=False)
        
        # [Attention Q, K, V, O] - 注意矩阵要转置
        export_tensor(state_dict[f"{prefix}.self_attn.q_proj.weight"], 
                      f"{save_dir}/layer{i}_w_q.bin", transpose=True)
        export_tensor(state_dict[f"{prefix}.self_attn.k_proj.weight"], 
                      f"{save_dir}/layer{i}_w_k.bin", transpose=True)
        export_tensor(state_dict[f"{prefix}.self_attn.v_proj.weight"], 
                      f"{save_dir}/layer{i}_w_v.bin", transpose=True)
        export_tensor(state_dict[f"{prefix}.self_attn.o_proj.weight"], 
                      f"{save_dir}/layer{i}_w_o.bin", transpose=True)
        
        # [Attention Bias]
        if f"{prefix}.self_attn.q_proj.bias" in state_dict:
            export_tensor(state_dict[f"{prefix}.self_attn.q_proj.bias"], 
                          f"{save_dir}/layer{i}_b_q.bin", transpose=False)
            export_tensor(state_dict[f"{prefix}.self_attn.k_proj.bias"], 
                          f"{save_dir}/layer{i}_b_k.bin", transpose=False)
            export_tensor(state_dict[f"{prefix}.self_attn.v_proj.bias"], 
                          f"{save_dir}/layer{i}_b_v.bin", transpose=False)
                          
        # [Norm 2]
        export_tensor(state_dict[f"{prefix}.post_attention_layernorm.weight"], 
                      f"{save_dir}/layer{i}_norm2_w.bin", transpose=False)
        
        # [FFN Weights] - 注意矩阵要转置
        export_tensor(state_dict[f"{prefix}.mlp.gate_proj.weight"], 
                      f"{save_dir}/layer{i}_w_gate.bin", transpose=True)
        export_tensor(state_dict[f"{prefix}.mlp.up_proj.weight"], 
                      f"{save_dir}/layer{i}_w_up.bin", transpose=True)
        export_tensor(state_dict[f"{prefix}.mlp.down_proj.weight"], 
                      f"{save_dir}/layer{i}_w_down.bin", transpose=True)
                      
        if (i + 1) % 4 == 0:
            print(f"✅ 已导出 {i + 1}/{num_layers} 层...")

    # ==========================================
    # 3. 导出模型最后的输出层 (Final Norm & LM Head)
    # ==========================================
    # 最后的 RMSNorm
    export_tensor(state_dict["model.norm.weight"], 
                  f"{save_dir}/final_norm_w.bin", transpose=False)
    
    # 最后的分类器矩阵 lm_head，它是一个超大的 2D 矩阵，需要做一次矩阵乘法，所以要转置
    export_tensor(state_dict["lm_head.weight"], 
                  f"{save_dir}/lm_head_w.bin", transpose=True)
    # 给 gemv 用（关键）
    export_tensor(state_dict["lm_head.weight"],
              "lm_head_w_T.bin", transpose=False)  # [N, K]

    print("✅ 输出层导出完成")

    print("\n🎉 恭喜！整个模型的所有必要权重已全部导出到 qwen_0.5b_full_bins 文件夹中！")

if __name__ == "__main__":
    main()
