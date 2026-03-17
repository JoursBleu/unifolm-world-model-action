#!/usr/bin/env python3
"""
GPU Benchmark: 测试各层 Linear/SDPA 的实际性能
      risks associated with Your exercise of permissions this  under
"""

import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


def sync_and_time():
    torch.cuda.synchronize()
    return time.time()


# ============================================================
# 硬件参数 (从 rocminfo/amd-smi 获取)
# ============================================================
# gfx1201: 64 CU, 2 SIMDs/CU, wavefront=32, max_freq=2.35GHz
# 理论 FP32: 64 * 2 * 32 * 2 * 2.35 = 19.3 TFLOPs
# 理论 FP16 (packed): ~38.6 TFLOPs
THEORETICAL_FP32_TFLOPS = 19.3
THEORETICAL_FP16_TFLOPS = 38.6

# ============================================================
# 模型参数
# ============================================================
video_length = 16
h_orig, w_orig = 320, 512
vae_downsample = 8
h_latent = h_orig // vae_downsample  # 40
w_latent = w_orig // vae_downsample  # 64

# UNet 各层分辨率 (channel_mult = [1, 2, 4, 4])
resolutions = [
    (h_latent, w_latent),  # level 0: 40x64 = 2560
    (h_latent // 2, w_latent // 2),  # level 1: 20x32 = 640
    (h_latent // 4, w_latent // 4),  # level 2: 10x16 = 160
    (h_latent // 8, w_latent // 8),  # level 3: 5x8 = 40
]
dims = [320, 640, 1280, 1280]
num_attn_layers = [4, 4, 4, 4]  # 每层的 attention block 数量
num_heads = 8
dim_head = 64
mult = 4  # FeedForward 扩展倍数


def benchmark_linear(in_dim, out_dim, seq_len, batch_size, device, dtype=torch.float32, num_runs=100, warmup=20):
    """Benchmark Linear layer, return (ms, TFLOPs achieved)"""
    linear = nn.Linear(in_dim, out_dim, bias=False).to(device, dtype=dtype)
    x = torch.randn(batch_size, seq_len, in_dim, device=device, dtype=dtype)

    # Warmup
    for _ in range(warmup):
        _ = linear(x)
    torch.cuda.synchronize()

    # Benchmark
    start = time.time()
    for _ in range(num_runs):
        _ = linear(x)
    torch.cuda.synchronize()
    elapsed_ms = (time.time() - start) / num_runs * 1000

    # FLOPs: 2 * batch * seq * in_dim * out_dim (matmul = multiply + add)
    flops = 2 * batch_size * seq_len * in_dim * out_dim
    tflops = flops / (elapsed_ms / 1000) / 1e12

    return elapsed_ms, tflops, flops


def benchmark_sdpa(batch, num_heads, seq_len, dim_head, device, dtype=torch.float32, num_runs=100, warmup=20):
    """Benchmark scaled_dot_product_attention"""
    q = torch.randn(batch, num_heads, seq_len, dim_head, device=device, dtype=dtype)
    k = torch.randn(batch, num_heads, seq_len, dim_head, device=device, dtype=dtype)
    v = torch.randn(batch, num_heads, seq_len, dim_head, device=device, dtype=dtype)

    # Warmup
    for _ in range(warmup):
        _ = F.scaled_dot_product_attention(q, k, v)
    torch.cuda.synchronize()

    # Benchmark
    start = time.time()
    for _ in range(num_runs):
        _ = F.scaled_dot_product_attention(q, k, v)
    torch.cuda.synchronize()
    elapsed_ms = (time.time() - start) / num_runs * 1000

    # SDPA FLOPs: 4 * batch * heads * seq^2 * dim_head (Q@K^T + softmax + @V)
    flops = 4 * batch * num_heads * seq_len * seq_len * dim_head
    tflops = flops / (elapsed_ms / 1000) / 1e12

    return elapsed_ms, tflops, flops


def benchmark_rearrange(x, pattern, num_runs=100, warmup=20, **kwargs):
    """Benchmark einops rearrange"""
    for _ in range(warmup):
        _ = rearrange(x, pattern, **kwargs)
    torch.cuda.synchronize()

    start = time.time()
    for _ in range(num_runs):
        _ = rearrange(x, pattern, **kwargs)
    torch.cuda.synchronize()
    elapsed_ms = (time.time() - start) / num_runs * 1000
    return elapsed_ms


def main():
    device = "cuda"
    dtype = torch.float32  # 可改为 torch.float16 测试

    print("=" * 100)
    print("GPU Benchmark: Linear / SDPA / Rearrange")
    print("=" * 100)

    # 设备信息
    if torch.cuda.is_available():
        print(f"Device: {torch.cuda.get_device_name(0)}")
        print(f"Device count: {torch.cuda.device_count()}")
    print(f"Dtype: {dtype}")
    print(f"Theoretical FP32: {THEORETICAL_FP32_TFLOPS:.1f} TFLOPs")
    print(f"Theoretical FP16: {THEORETICAL_FP16_TFLOPS:.1f} TFLOPs")

    print(f"\n原始分辨率: {h_orig}x{w_orig}")
    print(f"VAE latent: {h_latent}x{w_latent} = {h_latent * w_latent} tokens/frame")
    print(f"Video length: {video_length} frames")

    print("\n各层分辨率和 token 数:")
    for i, ((h, w), dim) in enumerate(zip(resolutions, dims)):
        print(f"  Level {i}: {h}x{w} = {h * w:4d} tokens/frame, dim={dim}")

    # ============================================================
    # Benchmark: FeedForward (GEGLU + Linear)
    # ============================================================
    print("\n" + "=" * 100)
    print("Benchmark: FeedForward (GEGLU proj + Linear out)")
    print("=" * 100)

    total_geglu_ms = 0.0
    total_linear_ms = 0.0
    total_geglu_flops = 0
    total_linear_flops = 0

    for i, ((h, w), dim) in enumerate(zip(resolutions, dims)):
        spatial_tokens = h * w
        inner_ff = dim * mult
        batch = video_length
        n_layers = num_attn_layers[i] * 2  # spatial + temporal 各有 FF

        # GEGLU: dim -> inner_ff * 2
        ms_g, tflops_g, flops_g = benchmark_linear(dim, inner_ff * 2, spatial_tokens, batch, device, dtype)
        # Linear out: inner_ff -> dim
        ms_l, tflops_l, flops_l = benchmark_linear(inner_ff, dim, spatial_tokens, batch, device, dtype)

        total_geglu_ms += ms_g * n_layers
        total_linear_ms += ms_l * n_layers
        total_geglu_flops += flops_g * n_layers
        total_linear_flops += flops_l * n_layers

        print(f"Level {i} (dim={dim}, {h}x{w}={spatial_tokens} tokens, batch={batch}):")
        print(f"  GEGLU proj:  {ms_g:7.3f} ms, {tflops_g:5.2f} TFLOPs ({tflops_g / THEORETICAL_FP32_TFLOPS * 100:5.1f}% FP32)")
        print(f"  Linear out:  {ms_l:7.3f} ms, {tflops_l:5.2f} TFLOPs ({tflops_l / THEORETICAL_FP32_TFLOPS * 100:5.1f}% FP32)")
        print(f"  x {n_layers} layers: {(ms_g + ms_l) * n_layers:7.2f} ms")

    total_ff_ms = total_geglu_ms + total_linear_ms
    avg_ff_tflops = (total_geglu_flops + total_linear_flops) / (total_ff_ms / 1000) / 1e12

    print(f"\n--- FeedForward 汇总 ---")
    print(f"总计 GEGLU:      {total_geglu_ms:7.1f} ms = {total_geglu_ms / 1000:.3f} s")
    print(f"总计 Linear out: {total_linear_ms:7.1f} ms = {total_linear_ms / 1000:.3f} s")
    print(f"总计 FF:         {total_ff_ms:7.1f} ms = {total_ff_ms / 1000:.3f} s")
    print(f"平均 TFLOPs:     {avg_ff_tflops:.2f} ({avg_ff_tflops / THEORETICAL_FP32_TFLOPS * 100:.1f}% FP32)")

    # ============================================================
    # Benchmark: CrossAttention QKV projections
    # ============================================================
    print("\n" + "=" * 100)
    print("Benchmark: CrossAttention (to_q/k/v + to_out)")
    print("=" * 100)

    total_qkv_ms = 0.0
    total_out_ms = 0.0
    inner_dim = num_heads * dim_head  # 8 * 64 = 512

    for i, ((h, w), dim) in enumerate(zip(resolutions, dims)):
        spatial_tokens = h * w
        batch = video_length
        n_layers = num_attn_layers[i] * 2 * 2  # spatial + temporal, self + cross

        # to_q/k/v: dim -> inner_dim (512)
        ms_qkv, tflops_qkv, _ = benchmark_linear(dim, inner_dim, spatial_tokens, batch, device, dtype)
        # to_out: inner_dim -> dim
        ms_out, tflops_out, _ = benchmark_linear(inner_dim, dim, spatial_tokens, batch, device, dtype)

        total_qkv_ms += ms_qkv * 3 * n_layers  # q, k, v
        total_out_ms += ms_out * n_layers

        print(f"Level {i} (dim={dim}, tokens={spatial_tokens}, batch={batch}):")
        print(f"  to_q/k/v:  {ms_qkv:7.3f} ms x 3, {tflops_qkv:5.2f} TFLOPs ({tflops_qkv / THEORETICAL_FP32_TFLOPS * 100:5.1f}% FP32)")
        print(f"  to_out:    {ms_out:7.3f} ms, {tflops_out:5.2f} TFLOPs ({tflops_out / THEORETICAL_FP32_TFLOPS * 100:5.1f}% FP32)")
        print(f"  x {n_layers} attn: {(ms_qkv * 3 + ms_out) * n_layers:7.2f} ms")

    print(f"\n--- CrossAttention Linear 汇总 ---")
    print(f"总计 to_qkv:  {total_qkv_ms:7.1f} ms = {total_qkv_ms / 1000:.3f} s")
    print(f"总计 to_out:  {total_out_ms:7.1f} ms = {total_out_ms / 1000:.3f} s")

    # ============================================================
    # Benchmark: SDPA
    # ============================================================
    print("\n" + "=" * 100)
    print("Benchmark: Scaled Dot-Product Attention (SDPA)")
    print("=" * 100)

    total_sdpa_ms = 0.0

    for i, ((h, w), dim) in enumerate(zip(resolutions, dims)):
        spatial_tokens = h * w
        batch = video_length
        n_layers = num_attn_layers[i] * 2 * 2  # spatial + temporal, self + cross

        ms_sdpa, tflops_sdpa, _ = benchmark_sdpa(batch, num_heads, spatial_tokens, dim_head, device, dtype)
        total_sdpa_ms += ms_sdpa * n_layers

        print(f"Level {i} (tokens={spatial_tokens}, batch={batch}, heads={num_heads}):")
        print(f"  SDPA: {ms_sdpa:7.3f} ms, {tflops_sdpa:5.2f} TFLOPs ({tflops_sdpa / THEORETICAL_FP32_TFLOPS * 100:5.1f}% FP32)")
        print(f"  x {n_layers} attn: {ms_sdpa * n_layers:7.2f} ms")

    print(f"\n--- SDPA 汇总 ---")
    print(f"总计 SDPA: {total_sdpa_ms:7.1f} ms = {total_sdpa_ms / 1000:.3f} s")

    # ============================================================
    # Benchmark: Rearrange
    # ============================================================
    print("\n" + "=" * 100)
    print("Benchmark: Rearrange (einops)")
    print("=" * 100)

    b, c, t, h, w = 1, 1280, video_length, 10, 16
    x1 = torch.randn(b, c, t, h, w, device=device, dtype=dtype)
    ms_r1 = benchmark_rearrange(x1, "b c t h w -> (b t) (h w) c")
    print(f"rearrange 'b c t h w -> (b t) (h w) c': {ms_r1:.3f} ms")

    x2 = torch.randn(b * t, h * w, c, device=device, dtype=dtype)
    ms_r2 = benchmark_rearrange(x2, "(b t) (h w) c -> b c t h w", b=b, t=t, h=h, w=w)
    print(f"rearrange back: {ms_r2:.3f} ms")

    n_rearranges = 100
    print(f"估计 {n_rearranges} 次 rearrange: {(ms_r1 + ms_r2) * n_rearranges / 2:.1f} ms")

    # ============================================================
    # 汇总
    # ============================================================
    print("\n" + "=" * 100)
    print("汇总对比")
    print("=" * 100)

    total_linear_all = total_ff_ms + total_qkv_ms + total_out_ms
    total_attention = total_sdpa_ms
    total_all = total_linear_all + total_attention

    print(f"FeedForward (GEGLU + Linear): {total_ff_ms:.1f} ms = {total_ff_ms / 1000:.3f} s")
    print(f"CrossAttention Linear (QKV + Out): {total_qkv_ms + total_out_ms:.1f} ms = {(total_qkv_ms + total_out_ms) / 1000:.3f} s")
    print(f"SDPA: {total_sdpa_ms:.1f} ms = {total_sdpa_ms / 1000:.3f} s")
    print(f"总计: {total_all:.1f} ms = {total_all / 1000:.3f} s")

    print(f"\nLinear 占比: {total_linear_all / total_all * 100:.1f}%")
    print(f"SDPA 占比:   {total_sdpa_ms / total_all * 100:.1f}%")

    print("\n" + "=" * 100)
    print("对比模型内测量值:")
    print("=" * 100)
    print("模型内测量 ff ≈ 1.28s/step")
    print(f"Benchmark 估算 ff ≈ {total_ff_ms / 1000:.3f}s")
    print("\n模型内测量 total attention (spatial+temporal) ≈ 2.77s/step")
    print(f"Benchmark 估算 total ≈ {total_all / 1000:.3f}s")


if __name__ == "__main__":
    main()
