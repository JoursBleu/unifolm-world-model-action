# UnifoLM-WMA FP16 vs FP32 性能对比报告

## 测试环境

| 项目 | 配置 |
|------|------|
| **GPU** | AMD Radeon RX 9070 XT (gfx1201, 64 CUs) |
| **驱动** | ROCm 7.1.0 |
| **PyTorch** | 2.4.0+rocm7.1.0 |
| **显存** | 16GB VRAM |
| **环境变量** | HSA_XNACK=1 |

## 测试配置

```yaml
dataset: unitree_z1_stackbox
batch_size: 1
height: 320
width: 512
video_length: 16
frame_stride: 4
ddim_steps: 50
n_iter: 1 (单次迭代)
seed: 123
```

---

## 1. 总体性能对比

| 指标 | FP32 | FP16 | 加速比 |
|------|------|------|--------|
| **总时间/iteration** | 561.73s | 46.03s | **12.2x** |
| **World Model 交互** | 280.72s | ~23s | ~12x |
| **DDIM 采样** | 270.31s | ~22s | ~12x |
| **VAE 解码** | 9.76s | ~0.8s | ~12x |

---

## 2. Video UNet 详细性能 (每个 DDIM step)

### 2.1 主要模块时间

| 模块 | FP32 时间 | 占比 |
|------|-----------|------|
| **Video UNet Total** | 5.07s | 100% |
| ├─ Input Blocks | 1.86s | 36.7% |
| ├─ Middle Block | 0.09s | 1.8% |
| ├─ Output Blocks | 3.12s | 61.5% |
| └─ Out Layer | 0.002s | 0.04% |

### 2.2 层级类型分解

| 层类型 | FP32 时间 | 调用次数 | 占比 |
|--------|-----------|----------|------|
| **ResBlock** | 2.09s | 22 | 41.2% |
| **Spatial Attention** | 1.46s | 16 | 28.8% |
| **Temporal Attention** | 1.33s | 17 | 26.2% |
| **Other** | 0.19s | 7 | 3.7% |

### 2.3 Input Blocks 详细 (b0-b11)

| Block | FP32 时间 | 说明 |
|-------|-----------|------|
| b0 | 0.18s | 入口卷积 |
| b1 | 0.33s | ResBlock + Attention |
| b2 | 0.33s | ResBlock + Attention |
| b3 | 0.008s | Downsample |
| b4 | 0.22s | ResBlock + Attention |
| b5 | 0.24s | ResBlock + Attention |
| b6 | 0.006s | Downsample |
| b7 | 0.24s | ResBlock + Attention |
| b8 | 0.26s | ResBlock + Attention |
| b9 | 0.008s | Downsample |
| b10 | 0.02s | ResBlock |
| b11 | 0.02s | ResBlock |

### 2.4 Output Blocks 详细 (b0-b11)

| Block | FP32 时间 | 说明 |
|-------|-----------|------|
| b0 | 0.03s | ResBlock |
| b1 | 0.03s | ResBlock |
| b2 | 0.06s | ResBlock + Upsample |
| b3 | 0.30s | ResBlock + Attention |
| b4 | 0.30s | ResBlock + Attention |
| b5 | 0.35s | ResBlock + Attention + Upsample |
| b6 | 0.31s | ResBlock + Attention |
| b7 | 0.27s | ResBlock + Attention |
| b8 | 0.32s | ResBlock + Attention + Upsample |
| b9 | 0.41s | ResBlock + Attention |
| b10 | 0.37s | ResBlock + Attention |
| b11 | 0.37s | ResBlock + Attention |

---

## 3. 算子级别分解 (FP32)

### 3.1 ResBlock 内部算子

| 算子 | FP32 时间 | 占 ResBlock 比例 |
|------|-----------|------------------|
| **in_layers** (norm+act+conv) | 0.84s | 40.2% |
| **temporal_conv** | 0.66s | 31.6% |
| **out_layers** (norm+act+conv) | 0.52s | 24.9% |
| **skip_conn** | 0.07s | 3.2% |
| **emb_layers** | 0.003s | 0.1% |

### 3.2 Spatial Attention 内部算子

| 算子 | FP32 时间 | 占比 |
|------|-----------|------|
| **attn_blocks** | 1.35s | 92.3% |
| **proj_in** | 0.06s | 4.0% |
| **proj_out** | 0.05s | 3.5% |
| **norm** | 0.003s | 0.2% |

### 3.3 Temporal Attention 内部算子

| 算子 | FP32 时间 | 占比 |
|------|-----------|------|
| **attn_blocks** | 1.20s | 90.2% |
| **proj_in** | 0.06s | 4.6% |
| **proj_out** | 0.06s | 4.6% |
| **norm** | 0.005s | 0.4% |

### 3.4 Attention Block 内部算子

| 算子 | FP32 时间 | 占比 |
|------|-----------|------|
| **ff** (FeedForward) | 1.30s | 51.2% |
| **self_attn** | 0.70s | 27.5% |
| **cross_attn** | 0.54s | 21.2% |

### 3.5 Cross Attention 内部算子

| 算子 | FP32 时间 | 占比 |
|------|-----------|------|
| **to_qkv** (Q/K/V 投影) | 0.66s | 55.0% |
| **sdpa** (注意力计算) | 0.32s | 26.8% |
| **to_out** (输出投影) | 0.22s | 18.2% |

### 3.6 FeedForward 内部算子

| 算子 | FP32 时间 | 占比 |
|------|-----------|------|
| **geglu** | 0.86s | 66.2% |
| **linear_out** | 0.44s | 33.8% |

---

## 4. Action/State UNet 性能

| 模块 | FP32 时间/step |
|------|---------------|
| **Action UNet** | 0.143s |
| **State UNet** | 0.143s |

> 注：Action/State UNet 相比 Video UNet (5.07s) 非常轻量

---

## 5. FP16 加速分析

### 5.1 加速原因

1. **计算带宽翻倍**：FP16 张量核心吞吐量是 FP32 的 2x
2. **显存带宽翻倍**：数据大小减半，内存传输速度翻倍
3. **显存占用减半**：可以处理更大的 batch 或更高分辨率
4. **更好的缓存利用**：更多数据可以驻留在 L2 缓存中

### 5.2 预估 FP16 算子时间

 12.2x 总体加速比，预估各算子 FP16 时间：

| 算子 | FP32 时间 | FP16 预估时间 |
|------|-----------|---------------|
| Video UNet Total | 5.07s | ~0.42s |
| Input Blocks | 1.86s | ~0.15s |
| Output Blocks | 3.12s | ~0.26s |
| ResBlock | 2.09s | ~0.17s |
| Spatial Attention | 1.46s | ~0.12s |
| Temporal Attention | 1.33s | ~0.11s |

---

## 6. 精度验证

| 指标 | 值 | 说明 |
|------|-----|------|
| **PSNR** | 47.88 dB | >40dB 为高质量 |
| **平均像素差异** | 0.54/255 | 0.21% |
| **最大像素差异** | 13/255 | 5.1% |
| **视觉质量** | 无可见差异 | ✓ |

---

## 7. 性能总结

```

                    性能提升总结                              │

  FP32 → FP16 加速比:  12.2x                                 │
  单次迭代时间:        561.73s → 46.03s                       │
  每秒处理帧数:        0.028 → 0.35 fps                       │
  精度损失:            PSNR 47.88 dB (几乎无损)               │

```

### 瓶颈分析

| 组件 | FP32 占比 | 说明 |
|------|-----------|------|
| **Video UNet** | ~96% | 主要瓶颈 |
| ├─ Output Blocks | 61.5% | 最大热点 |
| ├─ Input Blocks | 36.7% | 次热点 |
| └─ Middle Block | 1.8% | 较轻 |
| **Action/State UNet** | ~2.7% | 轻量级 |
| **VAE Decode** | ~1.7% | 轻量级 |

### 算子热点 (Top 5)

1. **ResBlock** - 41.2% (in_layers + temporal_conv 占主导)
2. **Spatial Attention** - 28.8% (attn_blocks 占 92%)
3. **Temporal Attention** - 26.2% (attn_blocks 占 90%)
4. **FeedForward** - 主要在 geglu 激活
5. **Cross Attention** - to_qkv 投影占 55%

---

## 8. 优化建议

### 已实施优化
- [x] FP16 手动转换 (model.half())
- [x] Trainer precision=16
- [x] GroupNorm FP16 兼容性修复
- [x] 位置编码 dtype 跟踪

### 后续优化方向
- [ ] Flash Attention 集成（可进一步提升 2-3x）
- [ ] 模型量化 (INT8/INT4)
- [ ] Torch Compile 优化
- [ ] 批处理优化（增加 batch size）

---

*报告生成时间: 2026-02-04*
*硬件平台: AMD Radeon RX 9070 XT*
