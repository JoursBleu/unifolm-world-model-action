#!/usr/bin/env python3
"""
Linear 层 GPU Benchmark (快速版)
"""
import time
import torch
import torch.nn as nn

device = "cuda"
dtype = torch.float32

# 理论峰值 (gfx1201: 64 CU, 2 SIMDs/CU, wavefront=32, 2.35GHz)
THEORETICAL_FP32_TFLOPS = 19.3


def benchmark(in_dim, out_dim, batch, seq_len, name="", warmup=5, runs=5):
    M = batch * seq_len
    linear = nn.Linear(in_dim, out_dim, bias=False).to(device, dtype)
    x = torch.randn(M, in_dim, device=device, dtype=dtype)

    for _ in range(warmup):
        _ = linear(x)
    torch.cuda.synchronize()

    start = time.time()
    for _ in range(runs):
        _ = linear(x)
    torch.cuda.synchronize()
    ms = (time.time() - start) / runs * 1000

    flops = 2 * M * in_dim * out_dim
    tflops = flops / (ms / 1000) / 1e12
    util = tflops / THEORETICAL_FP32_TFLOPS * 100

    print(f"{name:30s} | ({in_dim:5d} -> {out_dim:5d}) x {M:7d} | {ms:7.3f} ms | {tflops:5.2f} TFLOPs | {util:5.1f}%")
    return ms, tflops


if __name__ == "__main__":
    print("=" * 100)
    print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
    print(f"Dtype: {dtype}")
    print(f"Theoretical FP32: {THEORETICAL_FP32_TFLOPS} TFLOPs")
    print("=" * 100)

    # 预热 GPU 0.3 秒让其升频
    print("预热 GPU 0.3 秒...")
    linear = nn.Linear(4096, 4096, bias=False).to(device, dtype)
    x = torch.randn(4096, 4096, device=device, dtype=dtype)
    start = time.time()
    while time.time() - start < 0.3:
        _ = linear(x)
    torch.cuda.synchronize()
    del linear, x
    print("预热完成\n")

    print(f"{'Name':30s} | {'Shape':28s} | {'Time':9s} | {'TFLOPs':10s} | {'Util':6s}")
    print("-" * 100)

    # FeedForward GEGLU (level 0: 40x64=2560 tokens, batch=16)
    benchmark(320, 2560, 16, 2560, "FF GEGLU L0 (320->2560)")
    benchmark(1280, 320, 16, 2560, "FF Linear L0 (1280->320)")

    # FeedForward (level 2: 160 tokens)
    benchmark(1280, 10240, 16, 160, "FF GEGLU L2 (1280->10240)")
    benchmark(5120, 1280, 16, 160, "FF Linear L2 (5120->1280)")

    print("-" * 100)

    # CrossAttention to_qkv (dim -> 512)
    benchmark(320, 512, 16, 2560, "Attn to_q L0 (320->512)")
    benchmark(1280, 512, 16, 160, "Attn to_q L2 (1280->512)")

    # CrossAttention to_out (512 -> dim)
    benchmark(512, 320, 16, 2560, "Attn to_out L0 (512->320)")
    benchmark(512, 1280, 16, 160, "Attn to_out L2 (512->1280)")

    print("=" * 100)
