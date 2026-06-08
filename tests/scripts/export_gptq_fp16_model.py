#!/usr/bin/env python3
"""Export Qwen2.5 GPTQ-Int8 checkpoints for the RK3588 FP16+W8A16 engine."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM


NR = 16
GROUP_SIZE_DEFAULT = 128


def align_up(x: int, a: int) -> int:
    return (x + a - 1) // a * a


def save_f16(path, tensor, name):
    if tensor.dtype in (torch.int8, torch.uint8, torch.int16, torch.int32, torch.int64):
        raise ValueError(
            f"{name}: expected floating tensor for FP16 export, got {tensor.dtype}. "
            "Do not cast quantized integer tensors to fp16."
        )
    tensor.detach().cpu().to(torch.float16).contiguous().numpy().tofile(path)

def to_s8_array(t: torch.Tensor, name: str) -> np.ndarray:
    cpu = t.detach().cpu()
    if cpu.dtype == torch.uint8:
        return (cpu.to(torch.int16).numpy() - 128).astype(np.int8)
    if cpu.dtype == torch.int8:
        return cpu.numpy().astype(np.int8, copy=False)
    if cpu.dtype in (torch.int16, torch.int32, torch.int64):
        arr = cpu.numpy()
        if arr.min() < -128 or arr.max() > 127:
            raise ValueError(f"{name}: integer values are outside signed int8 range")
        return arr.astype(np.int8)
    raise ValueError(f"{name}: expected uint8/int8/integer GPTQ tensor, got {cpu.dtype}")


def unpack_int32_to_u8(arr: np.ndarray, pack_factor: int, name: str) -> np.ndarray:
    if pack_factor <= 0 or 32 % pack_factor != 0:
        raise ValueError(f"{name}: invalid pack_factor={pack_factor}")
    bits = 32 // pack_factor
    if bits != 8:
        raise ValueError(f"{name}: this exporter currently expects GPTQ int8, inferred bits={bits}")

    packed = arr.astype(np.uint32, copy=False)
    out_shape = (packed.shape[0] * pack_factor, packed.shape[1])
    out = np.empty(out_shape, dtype=np.uint8)
    mask = np.uint32((1 << bits) - 1)
    for i in range(pack_factor):
        out[i::pack_factor, :] = ((packed >> np.uint32(bits * i)) & mask).astype(np.uint8)
    return out


def unpack_packed_qweight(t: torch.Tensor, K: int, N: int, name: str) -> np.ndarray:
    cpu = t.detach().cpu()
    arr = cpu.numpy()

    if arr.size == K * N:
        return orient_matrix(to_s8_array(t, name), K, N, name)

    if arr.ndim != 2:
        raise ValueError(f"{name}: expected 2D qweight, got shape={arr.shape} dtype={cpu.dtype}")

    if arr.shape[1] == N and K % arr.shape[0] == 0:
        pack_factor = K // arr.shape[0]
        q_u8 = unpack_int32_to_u8(arr, pack_factor, name)[:K, :]
    elif arr.shape[0] == N and K % arr.shape[1] == 0:
        pack_factor = K // arr.shape[1]
        q_u8 = unpack_int32_to_u8(arr.T, pack_factor, name)[:K, :]
    else:
        raise ValueError(
            f"{name}: unsupported packed qweight shape={arr.shape}, expected "
            f"({K // 4}, {N}) for int8 AutoGPTQ or an unpacked ({K}, {N}) matrix"
        )

    return (q_u8.astype(np.int16) - 128).astype(np.int8)


def unpack_packed_qzeros(t: torch.Tensor, groups: int, N: int, name: str) -> np.ndarray:
    cpu = t.detach().cpu()
    arr = cpu.numpy()

    if arr.size == groups * N:
        return orient_matrix(to_s8_array(t, name), groups, N, name)

    if arr.ndim != 2:
        raise ValueError(f"{name}: expected 2D qzeros, got shape={arr.shape} dtype={cpu.dtype}")

    if arr.shape[0] == groups and N % arr.shape[1] == 0:
        pack_factor = N // arr.shape[1]
        z_u8 = unpack_int32_to_u8(arr.T, pack_factor, name).T[:, :N]
    elif arr.shape[1] == groups and N % arr.shape[0] == 0:
        pack_factor = N // arr.shape[0]
        z_u8 = unpack_int32_to_u8(arr, pack_factor, name).T[:, :N]
    else:
        raise ValueError(
            f"{name}: unsupported packed qzeros shape={arr.shape}, expected "
            f"({groups}, {N // 4}) for int8 AutoGPTQ or an unpacked ({groups}, {N}) matrix"
        )

    # AutoGPTQ stores zero - 1 in qzeros.
    z_u8 = (z_u8.astype(np.uint16) + 1).astype(np.uint8)
    return (z_u8.astype(np.int16) - 128).astype(np.int8)


def orient_matrix(arr: np.ndarray, rows: int, cols: int, name: str) -> np.ndarray:
    if arr.shape == (rows, cols):
        return np.ascontiguousarray(arr)
    if arr.shape == (cols, rows):
        return np.ascontiguousarray(arr.T)
    raise ValueError(f"{name}: expected shape {(rows, cols)} or {(cols, rows)}, got {arr.shape}")


def pack_gptq_w8a16(
    qweight_kn: np.ndarray,
    scales_gn: np.ndarray,
    zeros_gn: np.ndarray,
    K: int,
    N: int,
    group_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    npanels = (N + NR - 1) // NR
    k_pad = align_up(K, 8)
    num_groups = (K + group_size - 1) // group_size

    q_pack = np.zeros((npanels, k_pad, NR), dtype=np.int8)
    s_pack = np.zeros((npanels, num_groups, NR), dtype=np.float16)
    z_pack = np.zeros((npanels, num_groups, NR), dtype=np.int8)

    for panel in range(npanels):
        for k in range(K):
            for lane in range(NR):
                n = panel * NR + lane
                if n < N:
                    q_pack[panel, k, lane] = qweight_kn[k, n]

        for g in range(num_groups):
            for lane in range(NR):
                n = panel * NR + lane
                if n < N:
                    s_pack[panel, g, lane] = scales_gn[g, n]
                    z_pack[panel, g, lane] = zeros_gn[g, n]

    return q_pack, s_pack, z_pack


def dump_debug_keys(sd: dict[str, torch.Tensor], prefix: str) -> None:
    print(f"[DEBUG] tensors near '{prefix}':")
    for key, value in sd.items():
        if prefix in key:
            print(f"  {key}: shape={tuple(value.shape)} dtype={value.dtype}")


def require(sd: dict[str, torch.Tensor], key: str) -> torch.Tensor:
    if key not in sd:
        stem = ".".join(key.split(".")[:-1])
        dump_debug_keys(sd, stem)
        raise KeyError(f"missing required tensor: {key}")
    return sd[key]


def export_gptq_linear(
    sd: dict[str, torch.Tensor],
    hf_prefix: str,
    out_prefix: Path,
    K: int,
    N: int,
    group_size: int,
) -> None:
    num_groups = (K + group_size - 1) // group_size
    q_kn = unpack_packed_qweight(require(sd, hf_prefix + ".qweight"), K, N, hf_prefix + ".qweight")

    s_raw = require(sd, hf_prefix + ".scales").detach().cpu().to(torch.float16).numpy()
    s_gn = orient_matrix(s_raw, num_groups, N, hf_prefix + ".scales")

    if hf_prefix + ".qzeros" in sd:
        z_gn = unpack_packed_qzeros(sd[hf_prefix + ".qzeros"], num_groups, N, hf_prefix + ".qzeros")
    else:
        z_gn = np.zeros_like(s_gn, dtype=np.int8)

    q_pack, s_pack, z_pack = pack_gptq_w8a16(q_kn, s_gn, z_gn, K, N, group_size)
    q_pack.tofile(str(out_prefix) + ".qweight.s8pack.bin")
    s_pack.tofile(str(out_prefix) + ".scales.f16pack.bin")
    z_pack.tofile(str(out_prefix) + ".qzeros.s8pack.bin")

    gidx_key = hf_prefix + ".g_idx"
    if gidx_key in sd:
        require(sd, gidx_key).detach().cpu().to(torch.int32).contiguous().numpy().tofile(
            str(out_prefix) + ".g_idx.i32.bin"
        )


def export_fp16_linear_as_w8a16(
    tensor: torch.Tensor,
    out_prefix: Path,
    K: int,
    N: int,
    group_size: int,
    name: str,
) -> None:
    if tensor.dtype in (torch.int8, torch.uint8, torch.int16, torch.int32, torch.int64):
        raise ValueError(f"{name}: expected floating lm_head.weight, got {tensor.dtype}")

    w = tensor.detach().cpu().to(torch.float16).numpy()
    if w.shape == (N, K):
        w_nk = np.ascontiguousarray(w)
    elif w.shape == (K, N):
        w_nk = np.ascontiguousarray(w.T)
    else:
        raise ValueError(f"{name}: expected shape {(N, K)} or {(K, N)}, got {w.shape}")

    num_groups = (K + group_size - 1) // group_size
    q_kn = np.zeros((K, N), dtype=np.int8)
    scales_gn = np.zeros((num_groups, N), dtype=np.float16)
    zeros_gn = np.zeros((num_groups, N), dtype=np.int8)

    for g in range(num_groups):
        k0 = g * group_size
        k1 = min(K, k0 + group_size)
        block = w_nk[:, k0:k1].astype(np.float32, copy=False)
        max_abs = np.max(np.abs(block), axis=1)
        scale = np.where(max_abs > 0.0, max_abs / 127.0, 1.0).astype(np.float32)
        q_block = np.round(block / scale[:, None])
        q_block = np.clip(q_block, -127, 127).astype(np.int8)

        q_kn[k0:k1, :] = q_block.T
        scales_gn[g, :] = scale.astype(np.float16)

    q_pack, s_pack, z_pack = pack_gptq_w8a16(q_kn, scales_gn, zeros_gn, K, N, group_size)
    q_pack.tofile(str(out_prefix) + ".qweight.s8pack.bin")
    s_pack.tofile(str(out_prefix) + ".scales.f16pack.bin")
    z_pack.tofile(str(out_prefix) + ".qzeros.s8pack.bin")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="HF model id or local GPTQ checkpoint directory")
    parser.add_argument("--out", required=True, help="Output directory")
    parser.add_argument("--group-size", type=int, default=GROUP_SIZE_DEFAULT)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map="cpu",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    sd = model.state_dict()

    hidden = int(config.hidden_size)
    layers = int(config.num_hidden_layers)
    q_heads = int(config.num_attention_heads)
    kv_heads = int(config.num_key_value_heads)
    head_dim = hidden // q_heads
    intermediate = int(config.intermediate_size)
    vocab = int(config.vocab_size)

    save_f16(
        out_dir / "embed_tokens_w.f16.bin",
        require(sd, "model.embed_tokens.weight"),
        "model.embed_tokens.weight",
    )
    save_f16(
        out_dir / "final_norm_w.f16.bin",
        require(sd, "model.norm.weight"),
        "model.norm.weight",
    )

    for i in range(layers):
        base = f"model.layers.{i}"
        save_f16(
            out_dir / f"layer{i}_norm1_w.f16.bin",
            require(sd, base + ".input_layernorm.weight"),
            base + ".input_layernorm.weight",
        )
        save_f16(
            out_dir / f"layer{i}_norm2_w.f16.bin",
            require(sd, base + ".post_attention_layernorm.weight"),
            base + ".post_attention_layernorm.weight",
        )
        save_f16(
            out_dir / f"layer{i}_b_q.f16.bin",
            require(sd, base + ".self_attn.q_proj.bias"),
            base + ".self_attn.q_proj.bias",
        )
        save_f16(
            out_dir / f"layer{i}_b_k.f16.bin",
            require(sd, base + ".self_attn.k_proj.bias"),
            base + ".self_attn.k_proj.bias",
        )
        save_f16(
            out_dir / f"layer{i}_b_v.f16.bin",
            require(sd, base + ".self_attn.v_proj.bias"),
            base + ".self_attn.v_proj.bias",
        )

        export_gptq_linear(sd, base + ".self_attn.q_proj", out_dir / f"layer{i}_q_proj", hidden, q_heads * head_dim, args.group_size)
        export_gptq_linear(sd, base + ".self_attn.k_proj", out_dir / f"layer{i}_k_proj", hidden, kv_heads * head_dim, args.group_size)
        export_gptq_linear(sd, base + ".self_attn.v_proj", out_dir / f"layer{i}_v_proj", hidden, kv_heads * head_dim, args.group_size)
        export_gptq_linear(sd, base + ".self_attn.o_proj", out_dir / f"layer{i}_o_proj", q_heads * head_dim, hidden, args.group_size)
        export_gptq_linear(sd, base + ".mlp.gate_proj", out_dir / f"layer{i}_gate_proj", hidden, intermediate, args.group_size)
        export_gptq_linear(sd, base + ".mlp.up_proj", out_dir / f"layer{i}_up_proj", hidden, intermediate, args.group_size)
        export_gptq_linear(sd, base + ".mlp.down_proj", out_dir / f"layer{i}_down_proj", intermediate, hidden, args.group_size)

    if "lm_head.qweight" in sd:
        export_gptq_linear(sd, "lm_head", out_dir / "lm_head", hidden, vocab, args.group_size)
        lm_head_source = "gptq_checkpoint"
    elif "lm_head.weight" in sd:
        print(
            "[WARN] lm_head GPTQ tensors are absent; quantizing fp16 lm_head.weight "
            "to W8A16 groupwise int8 during export."
        )
        export_fp16_linear_as_w8a16(
            sd["lm_head.weight"],
            out_dir / "lm_head",
            hidden,
            vocab,
            args.group_size,
            "lm_head.weight",
        )
        lm_head_source = "export_side_groupwise_int8_from_fp16"
    else:
        dump_debug_keys(sd, "lm_head")
        raise RuntimeError("missing both lm_head.qweight and lm_head.weight")

    meta = {
        "hidden_dim": hidden,
        "num_layers": layers,
        "num_q_heads": q_heads,
        "num_kv_heads": kv_heads,
        "head_dim": head_dim,
        "intermediate_size": intermediate,
        "vocab_size": vocab,
        "group_size": args.group_size,
        "lm_head_source": lm_head_source,
        "dtype": "fp16_activation_gptq_int8_weight",
    }
    (out_dir / "export_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[OK] exported FP16+GPTQ-Int8 weights to {out_dir}")


if __name__ == "__main__":
    main()
