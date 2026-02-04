# UnifoLM-WMA FP16 vs FP32 性能对比报告

## 测试环境

| 项目 | AMD Radeon RX 9070 XT | NVIDIA RTX 4090 |
|------|----------------------|-----------------|
| **架构** | RDNA 4 (gfx1201, 64 CUs) | Ada Lovelace (AD102, 128 SMs) |
| **驱动** | ROCm 7.1.0 | CUDA 12.x |
| **PyTorch** | 2.4.0+rocm7.1.0 | 2.4.0+cu124 |
| **显存** | 16GB GDDR6 | 24GB GDDR6X |
| **显存带宽** | 650 GB/s | 1008 GB/s |
| **FP32 峰值** | 48.7 TFLOPs (Vector) | ~83 TFLOPs |
| **FP16 峰值** | 97.3 TFLOPs (Vector) / 195 TFLOPs (WMMA) | ~165 TFLOPs (Tensor Core) |

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
| 指标 | RX 9070 XT (FP32) | RX 9070 XT (FP16) | FP32→FP16 加速 | RTX 4090 (FP16) | 9070 XT vs 4090 |
|------|-------------------|-------------------|----------------|-----------------|-----------------|
| **单次迭代时间** | 561.73s | 46.03s | **12.2x** | 27.0s | 0.59x |
| **Action 生成** | ~272s | ~22s | **12.4x** | 13.0s | 0.59x |
| **World Model 交互** | ~280s | ~23s | **12.2x** | 13.4s | 0.58x |
### 调用树

```
#
 (561.73s FP32 -> 46.03s FP16)
|
+-- Action 生成 (~272s -> ~22s)
|   +-- Video UNet DDIM (50 steps × 5.07s -> 50 × 0.354s = 253s -> 17.7s)
|   +-- Action UNet DDIM (50 steps × 0.143s -> 50 × 0.035s = 7s -> 1.8s)
|   +-- 其他开销 (~12s -> ~2.5s)
|
+-- World Model 交互 (~280s -> ~23s)
    +-- Video UNet DDIM (50 steps × 5.07s -> 50 × 0.354s = 253s -> 17.7s)
    +-- State UNet (50 steps × 0.143s -> 50 × 0.035s = 7s -> 1.8s)
    +-- VAE Decode (~10s -> ~0.8s)
    +-- 其他开销 (~10s -> ~2.7s)

Video UNet 结构 (5.07s/step -> 0.354s/step):
    +-- Input Blocks (1.86s -> 0.130s)
    +-- Middle Block (0.09s -> 0.010s)
    +-- Output Blocks (3.12s -> 0.212s)
```
---
## 2. 精度验证 (FP16 vs FP32)
| 指标 | 值 | 说明 |
|------|-----|------|
| **PSNR** | 47.88 dB | >40dB 为高质量 |
| **平均像素差异** | 0.54/255 | 0.21% |
| **最大像素差异** | 13/255 | 5.1% |
| **视觉质量** | 无可见差异 | ✓ |
---
## 3. 性能总结
| 指标 | RX 9070 XT FP32 | RX 9070 XT FP16 | RTX 4090 FP16 |
|------|-----------------|-----------------|---------------|
| **单次迭代时间** | 561.73s | 46.03s | 27.0s |
| **FP32→FP16 加速比** | - | 12.2x | - |
| **每秒处理帧数** | 0.028 fps | 0.35 fps | 0.59 fps |
| **精度损失 (PSNR)** | - | 47.88 dB | - |
> **结论**: RX 9070 XT FP16 比 FP32 快 **12.2x**，RTX 4090 比 9070 XT FP16 快 **1.7x**
### 瓶颈分析
| 组件 | FP32 时间占比 | 说明 |
|------|---------------|------|
| **Video UNet** | ~96% | 主要瓶颈 |
| - Output Blocks | 61.5% | 最大热点 |
| - Input Blocks | 36.7% | 次热点 |
| - Middle Block | 1.8% | 较轻 |
| **Action/State UNet** | ~2.7% | 轻量级 |
| **VAE Decode** | ~1.7% | 轻量级 |
### 算子热点 (Top 5)
1. **ResBlock** - 41.2% (in_layers + temporal_conv 占主导)
2. **Spatial Attention** - 28.8% (attn_blocks 占 92%)
3. **Temporal Attention** - 26.2% (attn_blocks 占 90%)
4. **FeedForward** - 主要在 geglu 激活
5. **Cross Attention** - to_qkv 投影占 55%
---
## 4. 优化建议
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
## 附录 A: 详细性能 (每个 DDIM step)
### A.1 主要模块时间
| 模块 | FP32 时间 | FP16 时间 | 4090 FP16 |
|------|-----------|-----------|-----------|
| **Video UNet Total** | 5.07s | 0.354s | 0.377s |
| ├─ Input Blocks | 1.86s | 0.130s | 0.155s |
| ├─ Middle Block | 0.09s | 0.010s | 0.008s |
| ├─ Output Blocks | 3.12s | 0.212s | 0.214s |
| └─ Out Layer | 0.002s | 0.0016s | 0.0004s |
### A.2 层级类型分解
| 层类型 | FP32 时间 | FP16 时间 | 4090 FP16 |
|--------|-----------|-----------|-----------|
| **ResBlock** | 2.09s | 0.157s | 0.108s |
| **Spatial Attention** | 1.46s | 0.093s | 0.158s |
| **Temporal Attention** | 1.33s | 0.084s | 0.097s |
| **Other** | 0.19s | 0.015s | 0.011s |
### A.3 Action/State UNet 性能
| 模块 | FP32 时间/step | FP16 时间/step | 4090 FP16 |
|------|---------------|----------------|-----------|
| **Action UNet** | 0.143s | 0.035s | 0.012s |
| **State UNet** | 0.143s | 0.035s | 0.012s |
---
## 附录 B: Input/Output Blocks 详细数据
### B.1 Input Blocks 详细 (b0-b11)
| Block | FP32 时间 | FP16 时间 | 4090 FP16 | 说明 |
|-------|-----------|-----------|-----------|------|
| b0 | 0.18s | 0.013s | 0.013s | 入口卷积 |
| b1 | 0.33s | 0.027s | 0.036s | ResBlock + Attention |
| b2 | 0.33s | 0.027s | 0.036s | ResBlock + Attention |
| b3 | 0.008s | 0.001s | 0.001s | Downsample |
| b4 | 0.22s | 0.015s | 0.018s | ResBlock + Attention |
| b5 | 0.24s | 0.015s | 0.018s | ResBlock + Attention |
| b6 | 0.006s | 0.001s | 0.000s | Downsample |
| b7 | 0.24s | 0.011s | 0.014s | ResBlock + Attention |
| b8 | 0.26s | 0.012s | 0.014s | ResBlock + Attention |
| b9 | 0.008s | 0.001s | 0.001s | Downsample |
| b10 | 0.02s | 0.003s | 0.002s | ResBlock |
| b11 | 0.02s | 0.003s | 0.002s | ResBlock |
### B.2 Output Blocks 详细 (b0-b11)
| Block | FP32 时间 | FP16 时间 | 4090 FP16 | 说明 |
|-------|-----------|-----------|-----------|------|
| b0 | 0.03s | 0.005s | 0.002s | ResBlock |
| b1 | 0.03s | 0.004s | 0.002s | ResBlock |
| b2 | 0.06s | 0.006s | 0.003s | ResBlock + Upsample |
| b3 | 0.30s | 0.014s | 0.015s | ResBlock + Attention |
| b4 | 0.30s | 0.014s | 0.015s | ResBlock + Attention |
| b5 | 0.35s | 0.019s | 0.018s | ResBlock + Attention + Upsample |
| b6 | 0.31s | 0.019s | 0.019s | ResBlock + Attention |
| b7 | 0.27s | 0.018s | 0.018s | ResBlock + Attention |
| b8 | 0.32s | 0.021s | 0.021s | ResBlock + Attention + Upsample |
| b9 | 0.41s | 0.032s | 0.035s | ResBlock + Attention |
| b10 | 0.37s | 0.030s | 0.033s | ResBlock + Attention |
| b11 | 0.37s | 0.030s | 0.033s | ResBlock + Attention |
---
## 附录 C: 算子级别分解
### C.1 ResBlock 内部算子
| 算子 | FP32 时间 | FP16 时间 | 4090 FP16 |
|------|-----------|-----------|-----------|
| **in_layers** (norm+act+conv) | 0.84s | 0.056s | 0.035s |
| **temporal_conv** | 0.66s | 0.059s | 0.044s |
| **out_layers** (norm+act+conv) | 0.52s | 0.035s | 0.022s |
| **skip_conn** | 0.07s | 0.005s | 0.005s |
| **emb_layers** | 0.003s | 0.001s | 0.001s |
### C.2 Spatial Attention 内部算子
| 算子 | FP32 时间 | FP16 时间 | 4090 FP16 |
|------|-----------|-----------|-----------|
| **attn_blocks** | 1.35s | 0.083s | 0.147s |
| **proj_in** | 0.06s | 0.003s | 0.004s |
| **proj_out** | 0.05s | 0.004s | 0.004s |
| **norm** | 0.003s | 0.002s | 0.001s |
### C.3 Temporal Attention 内部算子
| 算子 | FP32 时间 | FP16 时间 | 4090 FP16 |
|------|-----------|-----------|-----------|
| **attn_blocks** | 1.20s | 0.063s | 0.080s |
| **proj_in** | 0.06s | 0.007s | 0.006s |
| **proj_out** | 0.06s | 0.008s | 0.005s |
| **norm** | 0.005s | 0.004s | 0.003s |
### C.4 Attention Block 内部算子
| 算子 | FP32 时间 | FP16 时间 | 4090 FP16 |
|------|-----------|-----------|-----------|
| **ff** (FeedForward) | 1.30s | 0.061s | 0.074s |
| **self_attn** | 0.70s | 0.043s | 0.072s |
| **cross_attn** | 0.54s | 0.040s | 0.079s |
### C.5 Cross Attention 内部算子
| 算子 | FP32 时间 | FP16 时间 | 4090 FP16 |
|------|-----------|-----------|-----------|
| **to_qkv** (Q/K/V 投影) | 0.66s | 0.023s | 0.015s |
| **sdpa** (注意力计算) | 0.32s | 0.032s | 0.013s |
| **to_out** (输出投影) | 0.22s | 0.009s | 0.006s |
### C.6 FeedForward 内部算子
| 算子 | FP32 时间 | FP16 时间 | 4090 FP16 |
|------|-----------|-----------|-----------|
| **geglu** | 0.86s | 0.041s | 0.051s |
| **linear_out** | 0.44s | 0.012s | 0.018s |
---
*报告时间: 2026-02-04*
*硬件平台: AMD Radeon RX 9070 XT, NVIDIA RTX 4090*
