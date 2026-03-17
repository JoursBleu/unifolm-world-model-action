import time

import torch
import torch.nn as nn
from einops import rearrange


# 实际模型参数
# height=320, width=512, video_length=16
# VAE 下采样 8x: h=40, w=64
# 然后 UNet 各层下采样

print("=" * 100)
print("分析单独测试 vs 模型内测试的差异")
print("=" * 100)

# 实际配置
video_length = 16
h_orig, w_orig = 320, 512
vae_downsample = 8
h_latent = h_orig // vae_downsample  # 40
w_latent = w_orig // vae_downsample  # 64

print(f"\n原始分辨率: {h_orig}x{w_orig}")
print(f"VAE latent: {h_latent}x{w_latent} = {h_latent * w_latent} tokens/frame")
print(f"Video length: {video_length} frames")

# UNet 各层分辨率 (按 channel_mult = [1, 2, 4, 4])
# 下采样: /2, /4, /8 (最深层)
resolutions = [
    (h_latent, w_latent),  # level 0: 40x64 = 2560
    (h_latent // 2, w_latent // 2),  # level 1: 20x32 = 640
    (h_latent // 4, w_latent // 4),  # level 2: 10x16 = 160
    (h_latent // 8, w_latent // 8),  # level 3: 5x8 = 40
]
dims = [320, 640, 1280, 1280]
num_attn_layers = [4, 4, 4, 0]  # Level 3 无 attention (不在 attention_resolutions)  # 估计每层的 attention 数量

print("\n各层分辨率和 token 数:")
for i, ((h, w), dim) in enumerate(zip(resolutions, dims)):
    print(f"  Level {i}: {h}x{w} = {h * w:4d} tokens/frame, dim={dim}")

# 自动检测设备
if torch.cuda.is_available():
    device = "cuda"
    def sync(): torch.cuda.synchronize()
else:
    device = "cpu"
    def sync(): pass

print(f"\n使用设备: {device}")
if device == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

num_heads = 8
mult = 4


def benchmark_linear(in_dim, out_dim, seq_len, batch_size=1, num_runs=50, warmup=10):
    linear = nn.Linear(in_dim, out_dim, bias=False).to(device)
    x = torch.randn(batch_size, seq_len, in_dim, device=device)

    for _ in range(warmup):
        _ = linear(x)
    sync()

    start = time.time()
    for _ in range(num_runs):
        _ = linear(x)
    sync()
    elapsed = (time.time() - start) / num_runs * 1000
    
    # 计算 FLOPs
    flops = 2 * batch_size * seq_len * in_dim * out_dim
    tflops = flops / (elapsed / 1000) / 1e12
    return elapsed, tflops


print("\n" + "=" * 100)
print("Benchmark: 各层 FeedForward 实际 shape")
print("=" * 100)

total_geglu_time = 0.0
total_linear_out_time = 0.0

for i, ((h, w), dim) in enumerate(zip(resolutions, dims)):
    spatial_tokens = h * w
    inner_ff = dim * mult  # 4x 扩展

    # GEGLU: dim -> inner_ff * 2
    # 实际调用时 batch = video_length (16)
    batch = video_length
    seq_len = spatial_tokens

    t_geglu, tflops_geglu = benchmark_linear(dim, inner_ff * 2, seq_len, batch_size=batch)
    t_linear, tflops_linear = benchmark_linear(inner_ff, dim, seq_len, batch_size=batch)

    # 每层有多少个 attention block? (spatial + temporal 各有 FeedForward)
    n_layers = num_attn_layers[i] * 2  # spatial + temporal 各有 FF

    layer_total = (t_geglu + t_linear) * n_layers
    total_geglu_time += t_geglu * n_layers
    total_linear_out_time += t_linear * n_layers

    print(f"Level {i} ({dim}d, {h}x{w}={spatial_tokens} tokens, batch={batch}):")
    print(f"  GEGLU:      {t_geglu:7.2f} ms x {n_layers} = {t_geglu * n_layers:7.2f} ms ({tflops_geglu:.2f} TFLOPs)")
    print(f"  Linear out: {t_linear:7.2f} ms x {n_layers} = {t_linear * n_layers:7.2f} ms ({tflops_linear:.2f} TFLOPs)")
    print(f"  Total:      {layer_total:7.2f} ms")

print(f"\n总计 GEGLU:      {total_geglu_time:.1f} ms = {total_geglu_time / 1000:.3f} s")
print(f"总计 Linear out: {total_linear_out_time:.1f} ms = {total_linear_out_time / 1000:.3f} s")
print(
    f"总计 FF:         {total_geglu_time + total_linear_out_time:.1f} ms = "
    f"{(total_geglu_time + total_linear_out_time) / 1000:.3f} s"
)

# 对比模型内测量
print("\n" + "=" * 100)
print("对比模型内测量值:")
print("=" * 100)
print("模型内测量 ff ≈ 1.28s/step")
print(f"Benchmark 估算 ff ≈ {(total_geglu_time + total_linear_out_time) / 1000:.3f}s")

# 理论峰值对比 (gfx1201: 64 CU x 2 SIMD x 32 lanes x 2 FMA x 2.35 GHz ≈ 19.3 TFLOPs FP32)
print("\n" + "=" * 100)
print("理论峰值对比 (gfx1201 / Radeon 9700 XT):")
print("=" * 100)
print("理论 FP32 峰值: 64 CU x 2 SIMD x 32 lanes x 2 FMA x 2.35 GHz ≈ 19.3 TFLOPs")
print("理论 FP16 峰值: ~38.6 TFLOPs (2x FP32)")
print(f"实测吞吐:       ~2.7 TFLOPs (约 14% FP32 峰值)")
print("注: 小矩阵/带宽受限场景下这是正常水平")

# 测量 rearrange 开销
print("\n" + "=" * 100)
print("测量 rearrange 开销:")
print("=" * 100)


def benchmark_rearrange(x, pattern_out, **kwargs):
    for _ in range(10):
        _ = rearrange(x, pattern_out, **kwargs)
    sync()

    start = time.time()
    for _ in range(100):
        _ = rearrange(x, pattern_out, **kwargs)
    sync()
    elapsed = (time.time() - start) / 100
    return elapsed * 1000


# SpatialTransformer 的 rearrange
b, c, t, h, w = 1, 1280, 16, 10, 16
x1 = torch.randn(b, c, t, h, w, device=device)

t1 = benchmark_rearrange(x1, "b c t h w -> (b t) (h w) c")
print(f"rearrange 'b c t h w -> (b t) (h w) c': {t1:.3f} ms")

x2 = torch.randn(b * t, h * w, c, device=device)

t2 = benchmark_rearrange(x2, "(b t) (h w) c -> b c t h w", t=t, h=h, w=w)
print(f"rearrange back: {t2:.3f} ms")

# 大量 rearrange 会累积
n_rearranges = 100  # 估计每步有很多 rearrange
print(f"\n估计 {n_rearranges} 次 rearrange: {(t1 + t2) * n_rearranges / 2:.1f} ms")
